"""
====================================================================
advanced_risk_analytics.py
====================================================================
Terzo modulo del progetto. Si appoggia a bootstrap_yield_curve.py e a
portfolio_risk_stress_test.py (stessa cartella, NON modificati) e
aggiunge le tre analisi che mancavano per chiudere il cerchio
metodologico rispetto a un vero framework di risk management fixed
income:

  A. CURVA FORWARD IMPLICITA + CARRY & ROLL-DOWN
     Cosa guadagno/perdo se i tassi NON si muovono. Dalla curva spot
     bootstrappata si derivano i tassi forward (sottoprodotto diretto
     di z(t)); da questi il carry (yield corrente vs costo di
     funding) e il roll-down (guadagno di prezzo se il titolo scivola
     lungo la curva, a curva invariata, man mano che il tempo passa).

  B. KEY RATE DURATION / PARTIAL DV01
     Duration e Convexity aggregate (script 2) rispondono bene a
     shock PARALLELI, ma nel dashboard pre-shock avevamo gia' scritto
     "qui serve full reval sulla curva intera" per gli scenari non
     paralleli. Qui si shocka un nodo di curva alla volta (usando
     esattamente l'infrastruttura shock_fn gia' presente in
     ContinuousCurve) e si ricava la risk ladder per scadenza: la
     mappa corretta di ESPOSIZIONE per capire perche' lo Scenario B
     (bear steepening) devasta il T-Bond 20Y e lo Scenario A no.

  C. VaR / EXPECTED SHORTFALL
     Un VaR parametrico single-factor sulla sola duration effettiva
     sottostimerebbe/sovrastimerebbe il rischio ignorando che i nodi
     di curva non si muovono tutti insieme. Si costruisce quindi un
     modello a 3 fattori (Level / Slope / Curvature, alla
     Litterman-Scheinkman) sui nodi bootstrappati, e lo si usa sia
     per un VaR parametrico (delta, chiuso in forma analitica) sia
     per un Monte Carlo a FULL REVAL (nessuna linearizzazione: ogni
     scenario riprezza il portafoglio sulla curva shockata nodo per
     nodo, sfruttando i Partial DV01 del punto B come pesi di
     interpolazione lineare tra i nodi).

  D. CORNISH-FISHER VaR (Delta-Gamma analitico)
     Il VaR Gaussiano del punto C e' lineare (ignora la convessita').
     Il Monte Carlo la cattura ma richiede simulazione. Qui si calcola
     Delta e Gamma del portafoglio rispetto ai 3 fattori (differenze
     finite, nessuna simulazione), si derivano in forma chiusa media/
     varianza/skewness/kurtosis del P&L (cumulanti di una forma
     quadratica Gaussiana) e si corregge il quantile normale con
     l'espansione di Cornish-Fisher: un metodo analitico che comunque
     "vede" l'asimmetria da convessita' del barbell.

Eseguito direttamente, stampa le tabelle di sintesi, esporta i CSV e
salva "advanced_risk_dashboard.png".

NOTA METODOLOGICA IMPORTANTE (da dichiarare sempre in sede di
presentazione): le volatilita' storiche dei tre fattori (ASSUMED_VOL_*
piu' sotto) sono valori PLACEHOLDER, calibrati a livelli tipici del
mercato Treasury ma non stimati da una serie storica reale (che non
e' presente nel dataset, il quale e' uno snapshot puntuale WSJ). Per
un uso "live" andrebbero ristimati su una finestra storica di
rendimenti effettivi (es. rolling 1Y di variazioni giornaliere dei
tassi CMT). La struttura del modello (fattoriale, PSD per
costruzione) resta comunque corretta e riutilizzabile.
====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy.stats import norm

from bootstrap_yield_curve import SETTLEMENT, year_frac_act365, NODES_INPUT
from portfolio_risk_stress_test import (
    curve, NODES_BY_NAME, PORTFOLIO, AGGREGATE_NOTIONAL,
    bond_cashflows, price_bond, price_portfolio, bond_ytm,
    DELTA, parallel_shock, effective_risk_metrics,
)

NAVY, GOLD, RED, GREY, LGREY = "#1F4E79", "#C9A24B", "#C00000", "#8C97A8", "#F2F2F2"
LEG_COLORS = {"T-Bill 6M": "#1F4E79", "T-Note 2Y": "#C00000", "T-Bond 20Y": "#E3A324"}

# Nodi di curva (in anni, ACT/365) = gli stessi 12 nodi bootstrappati
KEY_RATE_NODES = [year_frac_act365(SETTLEMENT, n[1]) for n in NODES_INPUT]
KEY_RATE_LABELS = [n[0].replace("T-Bill ", "").replace("T-Note ", "").replace("T-Bond ", "")
                    for n in NODES_INPUT]
N_NODES = len(KEY_RATE_NODES)

fair_value_base, breakdown_base = price_portfolio(detail=True)


# ====================================================================
# A. CURVA FORWARD IMPLICITA + CARRY & ROLL-DOWN
# ====================================================================
def forward_rate(t1: float, t2: float) -> float:
    """Tasso forward implicito (cont. comp.) fra t1 e t2, dalla curva
    spot bootstrappata: f(t1,t2) = [z(t2)*t2 - z(t1)*t1] / (t2 - t1).
    E' un sottoprodotto diretto della curva zero, nessun nuovo input."""
    if t2 <= t1:
        raise ValueError("t2 deve essere maggiore di t1")
    z1, z2 = curve.spot_rate(t1), curve.spot_rate(t2)
    return (z2 * t2 - z1 * t1) / (t2 - t1)


def instantaneous_forward_curve(tenor_grid, step=0.25):
    """Forward a termine costante 'step' (default 3M) lungo la curva,
    per il grafico 'la curva prezza gia' i rialzi/tagli futuri'."""
    fwd = []
    for t in tenor_grid:
        t1 = max(t, 1e-4)
        t2 = t1 + step
        fwd.append(forward_rate(t1, t2))
    return np.array(fwd)


def carry_and_rolldown_table(horizons=(0.25, 0.5)):
    """Carry & roll-down per ciascuna gamba del barbell.

    Carry (annualizzato) = YTM del titolo - tasso di funding (proxy:
    yield del T-Bill piu' corto della curva, 1M), cioe' il guadagno
    "certo" nel tempo se la curva NON si muove.

    Roll-down (per orizzonte h) = variazione di prezzo dovuta al fatto
    che, passato il tempo h, lo stesso titolo si troverebbe a scontare
    i propri flussi futuri sul tratto di curva PIU' CORTO (maturity
    residua T-h) anziche' su quello attuale (T) - a curva invariata.
    Approssimato con la duration effettiva del titolo:
        rolldown_% ~= ModDur(T) * [z(T) - z(T-h)]
    (positivo se la curva e' upward sloping, cioe' z(T) > z(T-h)).
    """
    funding_rate = curve.spot_rate(KEY_RATE_NODES[0])  # T-Bill 1M come proxy di funding/repo
    rows = []
    for pos in PORTFOLIO:
        node = pos["node"]
        name, maturity, coupon, price, freq, kind = NODES_BY_NAME[node]
        T = year_frac_act365(SETTLEMENT, maturity)
        z_T = curve.spot_rate(T)
        ytm = bond_ytm(node)
        pricer = lambda shock_fn, n=node: price_bond(n, shock_fn)
        _, mod_dur, _, _ = effective_risk_metrics(pricer)

        carry_annual_bp = (ytm - funding_rate) * 10000
        row = {"Titolo": node, "Yield spot z(T) (%)": z_T * 100,
               "YTM (%)": ytm * 100, "Funding (T-Bill 1M, %)": funding_rate * 100,
               "Carry annualizzato (bp)": carry_annual_bp}
        for h in horizons:
            T_h = max(T - h, 1e-4)
            z_Th = curve.spot_rate(T_h)
            rolldown_bp = (z_T - z_Th) * 10000
            rolldown_pct = mod_dur * (z_T - z_Th) * 100
            carry_pct_h = carry_annual_bp / 10000 * h * 100
            label = f"{int(h*12)}M"
            row[f"Roll-down {label} (bp curva)"] = rolldown_bp
            row[f"Carry {label} (%)"] = carry_pct_h
            row[f"Roll-down {label} (%)"] = rolldown_pct
            row[f"Total return {label} a curva invariata (%)"] = carry_pct_h + rolldown_pct
        rows.append(row)
    return pd.DataFrame(rows), funding_rate


# ====================================================================
# B. KEY RATE DURATION / PARTIAL DV01
# ====================================================================
def tent_weight(t: float, k: int, nodes) -> float:
    """Peso 'a tenda' (hat function) del nodo k in t: 1 nel nodo k,
    0 nei nodi adiacenti, interpolazione lineare fra i due, flat oltre
    il primo/ultimo nodo. E' la stessa base usata implicitamente da
    un'interpolazione lineare piecewise sui nodi -> per costruzione la
    somma dei pesi su tutti i nodi, per qualunque t, e' sempre 1."""
    n = len(nodes)
    tk = nodes[k]
    t_left = nodes[k - 1] if k > 0 else None
    t_right = nodes[k + 1] if k < n - 1 else None

    if t_left is None:  # primo nodo: flat=1 a sinistra, poi discende
        if t <= tk:
            return 1.0
        if t_right is not None and tk < t <= t_right:
            return (t_right - t) / (t_right - tk)
        return 0.0
    if t_right is None:  # ultimo nodo: sale da 0, poi flat=1 a destra
        if t >= tk:
            return 1.0
        if t_left <= t < tk:
            return (t - t_left) / (tk - t_left)
        return 0.0
    if t_left <= t <= tk:  # nodo interno: tenda triangolare
        return (t - t_left) / (tk - t_left)
    if tk < t <= t_right:
        return (t_right - t) / (t_right - tk)
    return 0.0


def key_rate_shock_fn(k: int, bp: float, nodes=KEY_RATE_NODES):
    return lambda t: bp * tent_weight(t, k, nodes)


def key_rate_duration_table():
    """Per ogni gamba del barbell e per il portafoglio, shocka un nodo
    di curva alla volta (+-1bp, tenda triangolare) e ricava Key Rate
    Duration e Partial DV01. Verifica anche l'additivita' rispetto
    alla duration/DV01 effettiva (parallela) dello script 2."""
    pricers = {pos["node"]: (lambda shock_fn, n=pos["node"]: price_bond(n, shock_fn) * pos["face"] / 100.0)
               for pos in PORTFOLIO}
    pricers["PORTAFOGLIO"] = lambda shock_fn: price_portfolio(shock_fn)

    krd_rows, dv01_rows = {}, {}
    for label, pricer_fn in pricers.items():
        p0 = pricer_fn(None)
        krds, dv01s = [], []
        for k in range(N_NODES):
            p_up = pricer_fn(key_rate_shock_fn(k, +DELTA))
            p_dn = pricer_fn(key_rate_shock_fn(k, -DELTA))
            krds.append((p_dn - p_up) / (2 * p0 * DELTA))
            dv01s.append((p_dn - p_up) / 2)
        krd_rows[label] = krds
        dv01_rows[label] = dv01s

    krd_df = pd.DataFrame(krd_rows, index=KEY_RATE_LABELS).T
    dv01_df = pd.DataFrame(dv01_rows, index=KEY_RATE_LABELS).T
    return krd_df, dv01_df


# ====================================================================
# C. VaR / EXPECTED SHORTFALL (fattoriale Level/Slope/Curvature)
# ====================================================================
# Volatilita' giornaliere ASSUNTE per fattore (bp/giorno) - vedi nota
# metodologica in testa al file: placeholder calibrati su ordini di
# grandezza tipici del mercato Treasury, da ri-stimare su serie
# storiche reali in un utilizzo "live".
ASSUMED_VOL_LEVEL = 2.05  # bp/giorno - shock parallelo di tutta la curva
ASSUMED_VOL_SLOPE = 2.47  # bp/giorno - irripidimento/appiattimento
ASSUMED_VOL_CURVE = 3.33  # bp/giorno - variazione di curvatura (ventre vs estremi) - il piu' alto: la volatilita' 2025-26 e' concentrata nel ventre 2Y-7Y (repricing del sentiero Fed), non sulla parallela
ASSUMED_VOL_IDIO = 1.10   # bp/giorno - rumore idiosincratico per nodo (RMS dei residui, 12 nodi)
# Valori CALIBRATI su dati storici reali Treasury CMT (trailing 1Y, 09/2025-09/2026).
# Vedi calibrate_factor_vols.py per il dettaglio riproducibile. Nome delle costanti
# mantenuto per compatibilita' col resto del modulo, ma non sono piu' placeholder.


def build_factor_loadings(nodes=KEY_RATE_NODES):
    """Loadings alla Litterman-Scheinkman: Level (piatto), Slope
    (crescente in log-maturity, normalizzato in [-1,1]), Curvature
    (gobba centrata sul ventre, ~0 agli estremi). Sigma risultante e'
    somma di 3 matrici rank-1 PSD + una diagonale idiosincratica,
    quindi PSD per costruzione (nessun rischio di matrice non valida)."""
    n = len(nodes)
    level = np.ones(n)
    log_t = np.log(np.array(nodes) + 0.05)
    slope = (log_t - log_t.mean())
    slope = slope / np.max(np.abs(slope))              # in [-1, 1]
    curvature = 1.0 - 4.0 * (0.5 * slope) ** 2          # gobba: 1 al centro, 0 agli estremi
    return level, slope, curvature


def build_covariance_matrix():
    level, slope, curvature = build_factor_loadings()
    n = N_NODES
    Sigma = (ASSUMED_VOL_LEVEL ** 2 * np.outer(level, level) +
             ASSUMED_VOL_SLOPE ** 2 * np.outer(slope, slope) +
             ASSUMED_VOL_CURVE ** 2 * np.outer(curvature, curvature) +
             ASSUMED_VOL_IDIO ** 2 * np.eye(n))
    return Sigma  # in bp^2


def parametric_var_es(dv01_vector, horizons_days=(1, 10), confidences=(0.95, 0.99)):
    """VaR/ES parametrico (delta-normal) multi-fattore: Var(PnL) =
    d^T * Sigma * d, dove d = Partial DV01 ($/bp) per nodo e Sigma la
    covarianza (bp^2) dei tassi ai nodi. Lineare: non cattura la
    convessita' (per questo si affianca il Monte Carlo a full reval)."""
    Sigma = build_covariance_matrix()
    d = np.array(dv01_vector)
    sigma_1d = np.sqrt(d @ Sigma @ d)  # $ per 1 giorno
    rows = []
    for h in horizons_days:
        sigma_h = sigma_1d * np.sqrt(h)
        for c in confidences:
            z = norm.ppf(c)
            var = z * sigma_h
            es = sigma_h * norm.pdf(z) / (1 - c)
            rows.append({"Orizzonte (gg)": h, "Confidenza": f"{c:.0%}",
                         "VaR parametrico ($)": var, "ES parametrico ($)": es})
    return pd.DataFrame(rows), sigma_1d


def monte_carlo_var_es(n_sims=20000, horizons_days=(1, 10), confidences=(0.95, 0.99), seed=42):
    """VaR/ES via Monte Carlo a FULL REVAL: si simulano shock
    congiunti Level/Slope/Curvature + rumore idiosincratico sui 12
    nodi di curva, si interpola linearmente lo shock lungo la curva
    (stessi pesi 'a tenda' del punto B) e si riprezza per intero ogni
    gamba del portafoglio su ciascuno scenario - nessuna
    linearizzazione, quindi la convessita' e la non-parallelicita'
    sono catturate esattamente."""
    rng = np.random.default_rng(seed)
    level, slope, curvature = build_factor_loadings()

    # Pesi di interpolazione (tenda) nodo->cashflow, precalcolati una
    # volta per gamba: interpolated_shock(t) = W[t,:] @ node_shock
    leg_data = []
    for pos in PORTFOLIO:
        cfs = bond_cashflows(pos["node"])
        times = np.array([year_frac_act365(SETTLEMENT, d) for d, _ in cfs])
        amounts = np.array([cf for _, cf in cfs])
        z0 = np.array([curve.spot_rate(t) for t in times])
        W = np.array([[tent_weight(t, k, KEY_RATE_NODES) for k in range(N_NODES)] for t in times])
        leg_data.append({"node": pos["node"], "face": pos["face"],
                          "times": times, "amounts": amounts, "z0": z0, "W": W})

    results = {}
    for h in horizons_days:
        scale = np.sqrt(h)
        f_level = rng.normal(0, ASSUMED_VOL_LEVEL * scale, n_sims)
        f_slope = rng.normal(0, ASSUMED_VOL_SLOPE * scale, n_sims)
        f_curve = rng.normal(0, ASSUMED_VOL_CURVE * scale, n_sims)
        idio = rng.normal(0, ASSUMED_VOL_IDIO * scale, size=(n_sims, N_NODES))
        node_shocks_bp = (np.outer(f_level, level) + np.outer(f_slope, slope) +
                           np.outer(f_curve, curvature) + idio)  # (n_sims, N_NODES)

        portfolio_value = np.zeros(n_sims)
        for leg in leg_data:
            shock_at_cf = (node_shocks_bp @ leg["W"].T) / 10000.0        # (n_sims, n_cf) decimale
            z_shocked = leg["z0"][None, :] + shock_at_cf                # broadcast
            df_matrix = np.exp(-z_shocked * leg["times"][None, :])
            pv_scenario = df_matrix @ leg["amounts"]                    # (n_sims,)
            portfolio_value += pv_scenario * leg["face"] / 100.0

        pnl = portfolio_value - fair_value_base
        for c in confidences:
            var = -np.quantile(pnl, 1 - c)
            es = -pnl[pnl <= np.quantile(pnl, 1 - c)].mean()
            results[(h, c)] = {"var": var, "es": es, "pnl": pnl if h == horizons_days[0] else None}

    rows = [{"Orizzonte (gg)": h, "Confidenza": f"{c:.0%}",
             "VaR Monte Carlo ($)": v["var"], "ES Monte Carlo ($)": v["es"]}
            for (h, c), v in results.items()]
    pnl_1d = results[(horizons_days[0], confidences[0])]["pnl"]
    return pd.DataFrame(rows), pnl_1d


# ====================================================================
# D. CORNISH-FISHER VaR (Delta-Gamma analitico, no simulazione)
# ====================================================================
# Il VaR Gaussiano (sezione C, parametric_var_es) e' lineare: usa solo
# il Delta (Partial DV01) e ignora completamente la convessita'. Il
# Monte Carlo la cattura per intero ma richiede simulazione. Cornish-
# Fisher e' la via di mezzo standard in letteratura (Britten-Jones &
# Schaefer 1999, Zangari 1996): si approssima il P&L con uno sviluppo
# Delta-Gamma nei 3 fattori Level/Slope/Curvature, se ne calcolano
# analiticamente i cumulanti (media, varianza, skewness, kurtosis in
# forma chiusa - nessuna simulazione), e si corregge il quantile
# normale con l'espansione di Cornish-Fisher.
def interp_loading(t: float, loadings, nodes=KEY_RATE_NODES) -> float:
    """Interpola un vettore di loadings definito sui nodi di curva in
    un punto t qualsiasi, con la stessa base 'a tenda' usata per KRD
    e Monte Carlo (garantisce coerenza fra le tre analisi)."""
    return sum(tent_weight(t, k, nodes) * loadings[k] for k in range(len(nodes)))


def factor_shock_fn(a_level: float = 0.0, a_slope: float = 0.0, a_curve: float = 0.0):
    """Shock di curva (decimale) generato da bump congiunti (in bp) dei
    3 fattori Level/Slope/Curvature, interpolati lungo la curva."""
    level, slope, curvature = build_factor_loadings()

    def shock(t):
        lv = interp_loading(t, level)
        sl = interp_loading(t, slope)
        cv = interp_loading(t, curvature)
        return (a_level * lv + a_slope * sl + a_curve * cv) / 10000.0
    return shock


def portfolio_delta_gamma(h_bp: float = 1.0):
    """Delta (3-vettore, $/bp) e Gamma (matrice 3x3, $/bp^2) del
    portafoglio rispetto ai 3 fattori, via differenze finite centrate
    (incluse le derivate incrociate Level-Slope, Level-Curvature,
    Slope-Curvature). Costo: 13 re-pricing del portafoglio completo."""
    def P(a_level, a_slope, a_curve):
        return price_portfolio(factor_shock_fn(a_level, a_slope, a_curve))

    p0 = P(0.0, 0.0, 0.0)
    n = 3
    delta = np.zeros(n)
    gamma = np.zeros((n, n))
    p_plus, p_minus = {}, {}
    for i in range(n):
        args_p, args_m = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        args_p[i], args_m[i] = h_bp, -h_bp
        p_plus[i], p_minus[i] = P(*args_p), P(*args_m)
        delta[i] = (p_plus[i] - p_minus[i]) / (2 * h_bp)
        gamma[i, i] = (p_plus[i] + p_minus[i] - 2 * p0) / (h_bp ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            a_pp, a_pm = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            a_mp, a_mm = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            a_pp[i], a_pp[j] = h_bp, h_bp
            a_pm[i], a_pm[j] = h_bp, -h_bp
            a_mp[i], a_mp[j] = -h_bp, h_bp
            a_mm[i], a_mm[j] = -h_bp, -h_bp
            cross = (P(*a_pp) - P(*a_pm) - P(*a_mp) + P(*a_mm)) / (4 * h_bp ** 2)
            gamma[i, j] = gamma[j, i] = cross
    return p0, delta, gamma


def delta_gamma_cumulants(delta, gamma, factor_vols, idio_var=0.0):
    """Cumulanti esatti (nessuna simulazione) del P&L Delta-Gamma
    Q = delta'X + 0.5 X'Gamma X, con X ~ N(0, diag(factor_vols^2)).
    Diagonalizzando A = D^0.5 Gamma D^0.5 (autovalori lambda_i, e
    proiezione c_i del Delta ruotato), la funzione generatrice dei
    cumulanti di ogni termine c_i*Z+0.5*lambda_i*Z^2 (Z~N(0,1)) da':
        k1_i = 0.5*lambda_i
        k2_i = c_i^2 + 0.5*lambda_i^2
        k3_i = 3*c_i^2*lambda_i + lambda_i^3
        k4_i = 12*c_i^2*lambda_i^2 + 3*lambda_i^4
    sommati sui 3 fattori (indipendenti per costruzione). idio_var
    aggiunge la varianza (solo lineare, Gamma=0) del rumore
    idiosincratico per-nodo, per coerenza col modello Monte Carlo."""
    D_half = np.diag(factor_vols)
    b = D_half @ delta
    A = D_half @ gamma @ D_half
    eigvals, eigvecs = np.linalg.eigh(A)
    c = eigvecs.T @ b
    k1 = 0.5 * np.sum(eigvals)
    k2 = np.sum(c ** 2 + 0.5 * eigvals ** 2) + idio_var
    k3 = np.sum(3 * c ** 2 * eigvals + eigvals ** 3)
    k4 = np.sum(12 * c ** 2 * eigvals ** 2 + 3 * eigvals ** 4)
    return k1, k2, k3, k4


def cornish_fisher_var_es(dv01_vector, horizons_days=(1, 10), confidences=(0.95, 0.99)):
    """VaR Cornish-Fisher (Delta-Gamma analitico) confrontato con la
    Gaussiana pura sullo stesso Delta/Gamma. Include un controllo di
    validita' elementare dell'espansione (deve restare monotona)."""
    p0, delta, gamma = portfolio_delta_gamma()
    idio_var_1d = np.sum((np.array(dv01_vector) * ASSUMED_VOL_IDIO) ** 2)

    rows = []
    for h in horizons_days:
        scale = np.sqrt(h)
        factor_vols_h = np.array([ASSUMED_VOL_LEVEL, ASSUMED_VOL_SLOPE, ASSUMED_VOL_CURVE]) * scale
        k1, k2, k3, k4 = delta_gamma_cumulants(delta, gamma, factor_vols_h, idio_var_1d * h)
        sigma = np.sqrt(k2)
        skew = k3 / sigma ** 3
        excess_kurt = k4 / k2 ** 2

        # controllo di validita' (monotonia) su griglia fine di quantili
        z_grid = np.linspace(-4, 4, 400)
        z_cf_grid = (z_grid + (z_grid**2 - 1) / 6 * skew + (z_grid**3 - 3*z_grid) / 24 * excess_kurt
                     - (2*z_grid**3 - 5*z_grid) / 36 * skew**2)
        valid = np.all(np.diff(z_cf_grid) > 0)

        for c in confidences:
            z = norm.ppf(1 - c)  # coda sinistra (perdita)
            z_cf = (z + (z**2 - 1)/6*skew + (z**3 - 3*z)/24*excess_kurt
                    - (2*z**3 - 5*z)/36*skew**2)
            q_gauss = k1 + sigma * z
            q_cf = k1 + sigma * z_cf
            rows.append({"Orizzonte (gg)": h, "Confidenza": f"{c:.0%}",
                         "Skewness": skew, "Excess Kurtosis": excess_kurt,
                         "VaR Gaussian (Delta-Gamma) ($)": -q_gauss,
                         "VaR Cornish-Fisher ($)": -q_cf,
                         "CF valido": valid})
    return pd.DataFrame(rows), delta, gamma


# ====================================================================
# STAMPA TABELLE + CSV
# ====================================================================
def run_advanced_analytics():
    print("=" * 78)
    print("A. CARRY & ROLL-DOWN (a curva invariata)")
    print("=" * 78)
    cr_df, funding_rate = carry_and_rolldown_table()
    print(f"Tasso di funding (proxy T-Bill 1M): {funding_rate*100:.3f}%\n")
    disp = cr_df.round(3)
    print(disp.to_string(index=False))
    cr_df.to_csv("carry_rolldown.csv", index=False)

    print("\n" + "=" * 78)
    print("B. KEY RATE DURATION (per nodo di curva)")
    print("=" * 78)
    krd_df, dv01_df = key_rate_duration_table()
    print("\n--- Key Rate Duration ---")
    print(krd_df.round(3).to_string())
    print("\n--- Partial DV01 ($/bp) ---")
    print(dv01_df.round(2).to_string())
    krd_sum = krd_df.loc["PORTAFOGLIO"].sum()
    print(f"\nCheck additivita': somma Key Rate Duration = {krd_sum:.3f}y "
          f"(vs. Eff. Mod. Duration parallela script 2 = 6.653y)")
    krd_df.to_csv("key_rate_duration.csv")
    dv01_df.to_csv("partial_dv01.csv")

    print("\n" + "=" * 78)
    print("C. VaR / EXPECTED SHORTFALL")
    print("=" * 78)
    dv01_vector = dv01_df.loc["PORTAFOGLIO"].values
    param_df, sigma_1d = parametric_var_es(dv01_vector)
    mc_df, pnl_1d = monte_carlo_var_es()
    print(f"\nDeviazione standard P&L 1 giorno (parametrica): ${sigma_1d:,.2f}\n")
    print("--- Parametrico (delta-normal, 3 fattori) ---")
    print(param_df.round(2).to_string(index=False))
    print("\n--- Monte Carlo (full reval, 20.000 scenari) ---")
    print(mc_df.round(2).to_string(index=False))
    param_df.to_csv("var_es_parametric.csv", index=False)
    mc_df.to_csv("var_es_montecarlo.csv", index=False)

    print("\n" + "=" * 78)
    print("D. CORNISH-FISHER VaR (Delta-Gamma analitico sui 3 fattori)")
    print("=" * 78)
    cf_df, delta_factors, gamma_factors = cornish_fisher_var_es(dv01_vector)
    print("Delta fattoriale ($/bp) [Level, Slope, Curvature]:", np.round(delta_factors, 2))
    print(f"(Check additivita': Delta Level = {delta_factors[0]:.2f} vs. -somma Partial DV01 "
          f"(shock parallelo) = {-dv01_vector.sum():.2f}, devono coincidere)")
    print()
    print(cf_df.round(4).to_string(index=False))
    if not cf_df["CF valido"].all():
        print("\nATTENZIONE: espansione Cornish-Fisher non monotona in almeno un caso "
              "(skew/kurtosis troppo elevati) - fare riferimento al Monte Carlo.")
    cf_df.to_csv("var_es_cornish_fisher.csv", index=False)

    return cr_df, funding_rate, krd_df, dv01_df, param_df, mc_df, pnl_1d, cf_df


# ====================================================================
# DASHBOARD
# ====================================================================
def render_advanced_dashboard(cr_df, funding_rate, krd_df, dv01_df, param_df, mc_df, pnl_1d, cf_df,
                               filename="advanced_risk_dashboard.png"):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16, 14), dpi=200)
    fig.patch.set_facecolor("white")

    fig.text(0.045, 0.975, "Advanced Risk Analytics — Forward, Key Rate Duration & VaR",
              fontsize=19, weight="bold", color=NAVY)
    fig.text(0.045, 0.952,
              f"Barbell Portfolio · Settlement {SETTLEMENT.strftime('%d/%m/%Y')} · "
              f"Fair Value ${fair_value_base/1e6:.3f}M",
              fontsize=10.5, color=GREY)

    # --- KPI cards ---
    var99_1d = param_df.query("`Orizzonte (gg)`==1 and Confidenza=='99%'")["VaR parametrico ($)"].iloc[0]
    var99_1d_cf = cf_df.query("`Orizzonte (gg)`==1 and Confidenza=='99%'")["VaR Cornish-Fisher ($)"].iloc[0]
    var99_1d_mc = mc_df.query("`Orizzonte (gg)`==1 and Confidenza=='99%'")["VaR Monte Carlo ($)"].iloc[0]
    es99_1d_mc = mc_df.query("`Orizzonte (gg)`==1 and Confidenza=='99%'")["ES Monte Carlo ($)"].iloc[0]
    kpis = [
        ("VaR 99% 1D (GAUSSIAN)", f"${var99_1d:,.0f}", NAVY),
        ("VaR 99% 1D (CORNISH-FISHER)", f"${var99_1d_cf:,.0f}", "#6B3FA0"),
        ("VaR 99% 1D (MONTE CARLO)", f"${var99_1d_mc:,.0f}", RED),
        ("ES 99% 1D (MONTE CARLO)", f"${es99_1d_mc:,.0f}", GOLD),
    ]
    kpi_w, kpi_gap, kpi_y, kpi_h = 0.215, 0.02, 0.895, 0.062
    for i, (label, value, color) in enumerate(kpis):
        x0 = 0.045 + i * (kpi_w + kpi_gap)
        ax = fig.add_axes([x0, kpi_y, kpi_w, kpi_h])
        ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.08",
                                              transform=ax.transAxes, facecolor=LGREY,
                                              edgecolor="#D9D9D9", linewidth=1))
        ax.add_patch(mpatches.Rectangle((0, 0), 0.025, 1, transform=ax.transAxes, facecolor=color, edgecolor="none"))
        ax.text(0.13, 0.62, label, transform=ax.transAxes, fontsize=7.8, color="#555555", weight="bold", va="center")
        ax.text(0.13, 0.28, value, transform=ax.transAxes, fontsize=16, color=color, weight="bold", va="center")

    # --- Panel 1: curva spot vs forward 3M ---
    ax1 = fig.add_axes([0.045, 0.685, 0.42, 0.175])
    t_grid = np.linspace(0.05, 29.5, 300)
    spot_vals = [curve.spot_rate(t) * 100 for t in t_grid]
    fwd_vals = instantaneous_forward_curve(t_grid) * 100
    ax1.plot(t_grid, spot_vals, color=NAVY, linewidth=2.2, label="Spot (bootstrapped)")
    ax1.plot(t_grid, fwd_vals, color=GOLD, linewidth=2.0, linestyle="--", label="Implied 3M forward")
    ax1.set_xscale("symlog", linthresh=1)
    ax1.set_xticks([0.25, 1, 2, 5, 10, 20, 30])
    ax1.set_xticklabels(["3M", "1Y", "2Y", "5Y", "10Y", "20Y", "30Y"])
    ax1.minorticks_off()
    ax1.set_ylabel("Rate (%, cont. comp.)", fontsize=9.3)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.legend(loc="lower right", fontsize=8.3, frameon=False)
    ax1.set_title("A. Spot Curve vs Implied Forward", fontsize=11.5, weight="bold", color=NAVY, loc="left")

    # --- Panel 2: carry & rolldown per leg (orizzonte 3M) ---
    ax2 = fig.add_axes([0.545, 0.685, 0.41, 0.175])
    legs = cr_df["Titolo"].tolist()
    carry_3m = cr_df["Carry 3M (%)"].values
    roll_3m = cr_df["Roll-down 3M (%)"].values
    x = np.arange(len(legs))
    width = 0.6
    ax2.bar(x, carry_3m, width, color=NAVY, label="Carry 3M")
    ax2.bar(x, roll_3m, width, bottom=carry_3m, color=GOLD, label="Roll-down 3M")
    total = carry_3m + roll_3m
    for xi, tot in zip(x, total):
        ax2.annotate(f"{tot:+.2f}%", (xi, tot), textcoords="offset points", xytext=(0, 4),
                     ha="center", fontsize=9, weight="bold", color=NAVY)
    ax2.axhline(0, color=GREY, linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(legs, fontsize=9.3)
    ax2.set_ylabel("Expected 3M return, unchanged curve (%)", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.legend(loc="upper left", fontsize=8.3, frameon=False)
    ax2.set_title("A. Carry & Roll-down by Leg (3M horizon)", fontsize=11.5, weight="bold", color=NAVY, loc="left")

    # --- Panel 3: Key Rate Duration ladder ---
    ax3 = fig.add_axes([0.045, 0.475, 0.90, 0.165])
    node_labels = krd_df.columns.tolist()
    xk = np.arange(len(node_labels))
    n_legs = len(PORTFOLIO)
    bw = 0.8 / n_legs
    for i, pos in enumerate(PORTFOLIO):
        vals = krd_df.loc[pos["node"]].values
        ax3.bar(xk + (i - n_legs/2 + 0.5) * bw, vals, bw,
                color=LEG_COLORS[pos["node"]], label=pos["node"], edgecolor="white", linewidth=0.4)
    ax3.axhline(0, color=GREY, linewidth=0.8)
    ax3.set_xticks(xk)
    ax3.set_xticklabels(node_labels, fontsize=8.8)
    ax3.set_ylabel("Key Rate Duration (years)", fontsize=9)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    ax3.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=3)
    ax3.set_title("B. Key Rate Duration — Risk Ladder by Curve Node",
                  fontsize=11.5, weight="bold", color=NAVY, loc="left")

    # --- Panel 4: distribuzione P&L Monte Carlo + VaR/ES ---
    ax4 = fig.add_axes([0.045, 0.245, 0.42, 0.185])
    ax4.hist(pnl_1d, bins=80, color=NAVY, alpha=0.75, edgecolor="white", linewidth=0.3)
    var99 = mc_df.query("`Orizzonte (gg)`==1 and Confidenza=='99%'")["VaR Monte Carlo ($)"].iloc[0]
    es99 = mc_df.query("`Orizzonte (gg)`==1 and Confidenza=='99%'")["ES Monte Carlo ($)"].iloc[0]
    ax4.axvline(-var99, color=RED, linewidth=2, linestyle="--", label=f"VaR 99% = ${var99:,.0f}")
    ax4.axvline(-es99, color="#7A0000", linewidth=2, linestyle=":", label=f"ES 99% = ${es99:,.0f}")
    ax4.set_xlabel("Portfolio P&L, 1 day ($)", fontsize=9)
    ax4.set_ylabel("Frequency (of 20,000 scenarios)", fontsize=9)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.grid(axis="y", alpha=0.3, linestyle="--")
    ax4.legend(loc="upper left", fontsize=8.3, frameon=False)
    ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax4.set_title("C. P&L Distribution — Monte Carlo Full Reval",
                  fontsize=11.5, weight="bold", color=NAVY, loc="left")

    # --- Panel 5: confronto VaR 99% - Gaussiano vs Cornish-Fisher vs Monte Carlo ---
    ax5 = fig.add_axes([0.545, 0.245, 0.41, 0.185])
    horizons = sorted(param_df["Orizzonte (gg)"].unique())
    methods = [("Gaussian", param_df, "VaR parametrico ($)", NAVY),
               ("Cornish-Fisher", cf_df, "VaR Cornish-Fisher ($)", "#6B3FA0"),
               ("Monte Carlo", mc_df, "VaR Monte Carlo ($)", RED)]
    xh = np.arange(len(horizons))
    bw5 = 0.8 / len(methods)
    for i, (mname, df_m, col, color) in enumerate(methods):
        vals = [df_m.query(f"`Orizzonte (gg)`=={h} and Confidenza=='99%'")[col].iloc[0] for h in horizons]
        bars = ax5.bar(xh + (i - len(methods)/2 + 0.5) * bw5, vals, bw5, color=color, label=mname,
                        edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax5.annotate(f"${v:,.0f}", (b.get_x() + b.get_width()/2, v), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=7.6, color=color, weight="bold")
    ax5.set_xticks(xh)
    ax5.set_xticklabels([f"{h} day" if h == 1 else f"{h} days" for h in horizons], fontsize=9.3)
    ax5.set_ylabel("VaR 99% ($)", fontsize=9)
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax5.spines[["top", "right"]].set_visible(False)
    ax5.grid(axis="y", alpha=0.3, linestyle="--")
    ax5.legend(loc="upper left", fontsize=8.3, frameon=False)
    ax5.set_title("D. VaR 99%: Gaussian vs Cornish-Fisher vs Monte Carlo",
                  fontsize=11.5, weight="bold", color=NAVY, loc="left")

    # --- Box di lettura ---
    ax_note = fig.add_axes([0.045, 0.03, 0.91, 0.175])
    ax_note.axis("off")
    ax_note.add_patch(mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.05",
                                               transform=ax_note.transAxes, facecolor=LGREY,
                                               edgecolor="#D9D9D9", linewidth=1))
    ax_note.add_patch(mpatches.FancyBboxPatch((0, 0), 0.010, 1, boxstyle="round,pad=0,rounding_size=0.05",
                                               transform=ax_note.transAxes, facecolor=NAVY, edgecolor="none"))
    var99_param = param_df.query("`Orizzonte (gg)`==1 and Confidenza=='99%'")["VaR parametrico ($)"].iloc[0]
    long_leg_krd = krd_df.loc["T-Bond 20Y"]
    top_node = long_leg_krd.idxmax()
    skew_1d = cf_df.query("`Orizzonte (gg)`==1 and Confidenza=='99%'")["Skewness"].iloc[0]
    cf_ok = cf_df["CF valido"].all()

    col1 = [
        ("KEY POINTS", NAVY, 10.3, True),
        ("", None, 4, False),
        (f"- VaR 99% 1d: \\${var99_param:,.0f} Gaussian vs \\${var99_1d_cf:,.0f}", "#333333", 9.0, False),
        (f"  Cornish-Fisher vs \\${var99:,.0f} Monte Carlo. P&L skewness", "#333333", 9.0, False),
        (f"  ({skew_1d:+.3f}, positive from the barbell's convexity)", "#333333", 9.0, False),
        ("  pulls the Cornish-Fisher VaR below the Gaussian one:", "#333333", 9.0, False),
        ("  the loss tail is lighter than a pure Normal would", "#333333", 9.0, False),
        ("  suggest.", "#333333", 9.0, False),
        (f"- Cornish-Fisher expansion: {'valid (monotonic)' if cf_ok else 'WARNING, not valid'} "
         f"across the whole tested horizon.", "#333333", 9.0, False),
    ]
    col2 = [
        ("", None, 4, False),
        ("", None, 4, False),
        (f"- T-Bond 20Y's Key Rate Duration is concentrated on node", "#333333", 9.0, False),
        (f"  {top_node} ({long_leg_krd.max():.2f}y): consistent with the $31.8k", "#333333", 9.0, False),
        ("  loss in Scenario B (steepening) from Module 2 - a", "#333333", 9.0, False),
        ("  non-parallel shock that none of the three VaR figures", "#333333", 9.0, False),
        ("  above (all calibrated on an 'average' factor model)", "#333333", 9.0, False),
        ("  is guaranteed to fully capture: hence the value of", "#333333", 9.0, False),
        ("  always pairing this with Module 2's scenario stress test.", "#333333", 9.0, False),
    ]
    for lines, x0 in [(col1, 0.03), (col2, 0.52)]:
        y0 = 0.93
        for text, color, size, bold in lines:
            if text == "":
                y0 -= 0.07
                continue
            ax_note.text(x0, y0, text, transform=ax_note.transAxes, fontsize=size,
                         color=color if color else "#333333", weight="bold" if bold else "normal", va="top")
            y0 -= 0.10

    plt.savefig(filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nDashboard salvata: {filename}")


# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    cr_df, funding_rate, krd_df, dv01_df, param_df, mc_df, pnl_1d, cf_df = run_advanced_analytics()
    render_advanced_dashboard(cr_df, funding_rate, krd_df, dv01_df, param_df, mc_df, pnl_1d, cf_df)

