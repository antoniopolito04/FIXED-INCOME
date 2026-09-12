"""
====================================================================
portfolio_risk_stress_test.py
====================================================================
Costruzione del portafoglio Barbell, pricing esatto sulla curva spot
bootstrappata, metriche di rischio effettive, dashboard pre-shock e
Scenario Analysis (stress test macroeconomico). Consolida in un unico
script coerente quello che nei file caricati era sparso e duplicato
tra "Barbell portfolio construction.py", "Risk_dashboard.py" e
"Stress_testing.py".

Dipende da bootstrap_yield_curve.py (stessa cartella).

Note di conciliazione rispetto ai file originali:
  - Gamba lunga del barbell: T-Bond 20Y (non 30Y). I file caricati
    erano incoerenti fra loro su questo punto; 20Y e' stato scelto
    perche' e' il nodo con il premio a termine piu' marcato della
    curva bootstrappata.
  - Duration/Convexity "ufficiali" del progetto sono quelle EFFETTIVE
    (re-pricing completo del portafoglio su shift della curva spot),
    non quelle a YTM singolo (Newton-Raphson) usate in uno dei file
    caricati: con cash flow scontati sul tasso spot esatto per
    ciascuna scadenza, la duration di Macaulay "chiusa" su un unico
    rendimento non è ben definita per un portafoglio multi-scadenza.
    Il YTM per singolo titolo resta calcolato (Newton-Raphson, bug
    dell'operatore ** corretto) solo come statistica descrittiva.
====================================================================
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from bootstrap_yield_curve import (
    SETTLEMENT, FACE, NODES_INPUT,
    year_frac_act365, coupon_schedule, accrued_interest,
    build_continuous_curve,
)

curve, t_nodes, z_nodes, _ = build_continuous_curve()
NODES_BY_NAME = {n[0]: n for n in NODES_INPUT}

# ====================================================================
# 1. PORTAFOGLIO BARBELL
# ====================================================================
# Barbell "vero": peso concentrato agli estremi, ventre alleggerito,
# per massimizzare il vantaggio di convessità sul bullet equivalente.
AGGREGATE_NOTIONAL = 1_000_000.0

PORTFOLIO = [
    {"node": "T-Bill 6M",  "role": "Liquidita' (short end)",         "weight": 0.30},
    {"node": "T-Note 2Y",  "role": "Fed-sensitive (belly)",          "weight": 0.20},
    {"node": "T-Bond 20Y", "role": "Duration/convexity (long end)",  "weight": 0.50},
]
for pos in PORTFOLIO:
    pos["face"] = AGGREGATE_NOTIONAL * pos["weight"]

# ====================================================================
# 2. PRICING ESATTO (MARK-TO-MARKET) VIA CASH-FLOW MAPPING
# ====================================================================
def bond_cashflows(node_name):
    """Flussi futuri (data, importo) per 100 di nominale."""
    name, maturity, coupon, price, freq, kind = NODES_BY_NAME[node_name]
    if kind == "bill":
        return [(maturity, FACE)]
    _, future_dates = coupon_schedule(maturity, SETTLEMENT, freq)
    coupon_cash = FACE * coupon / freq
    cfs = [(d, coupon_cash) for d in future_dates[:-1]]
    cfs.append((future_dates[-1], coupon_cash + FACE))
    return cfs

def price_bond(node_name, shock_fn=None):
    """Fair Value per 100 di nominale: cash-flow mapping + sconto sul
    tasso spot esatto letto dalla curva continua (CubicSpline)."""
    pv = 0.0
    for d, cf in bond_cashflows(node_name):
        t = year_frac_act365(SETTLEMENT, d)
        pv += cf * curve.discount_factor(t, shock_fn)
    return pv

def price_portfolio(shock_fn=None, detail=False):
    rows, total = [], 0.0
    for pos in PORTFOLIO:
        price_100 = price_bond(pos["node"], shock_fn)
        value = price_100 * pos["face"] / 100.0
        total += value
        rows.append({"node": pos["node"], "face": pos["face"],
                      "price_100": price_100, "value": value})
    if detail:
        return total, pd.DataFrame(rows)
    return total

# ====================================================================
# 3. METRICHE DI RISCHIO
# ====================================================================
# 3a. YTM per singolo titolo (Newton-Raphson) - SOLO statistica
#     descrittiva per la dashboard, non usata per Duration/Convexity.
def bond_ytm(node_name):
    name, maturity, coupon, price, freq, kind = NODES_BY_NAME[node_name]
    dirty_price = price_bond(node_name)
    if freq == 0:
        T = year_frac_act365(SETTLEMENT, maturity)
        return (FACE / dirty_price) ** (1 / T) - 1 if T > 0 else 0.0

    flows = [(year_frac_act365(SETTLEMENT, d), cf) for d, cf in bond_cashflows(node_name)]
    y = 0.04
    for _ in range(200):
        price_y = sum(cf / (1 + y / 2) ** (2 * t) for t, cf in flows)
        dprice_dy = sum(-t * cf / (1 + y / 2) ** (2 * t + 1) for t, cf in flows)
        if abs(price_y - dirty_price) < 1e-9 or dprice_dy == 0:
            break
        y -= (price_y - dirty_price) / dprice_dy
    return y

# 3b. Duration/Convexity EFFETTIVE (re-pricing completo, shift
#     parallelo +-1bp della curva spot) - metriche di rischio ufficiali.
DELTA = 0.0001  # 1 basis point

def parallel_shock(bp):
    return lambda t: bp

def effective_risk_metrics(pricer_fn):
    p0 = pricer_fn(None)
    p_up = pricer_fn(parallel_shock(+DELTA))
    p_dn = pricer_fn(parallel_shock(-DELTA))
    mod_dur = (p_dn - p_up) / (2 * p0 * DELTA)
    convexity = (p_up + p_dn - 2 * p0) / (p0 * DELTA ** 2)
    dv01 = (p_dn - p_up) / 2
    return p0, mod_dur, convexity, dv01

# ====================================================================
# 4. SCENARI DI STRESS MACROECONOMICO
# ====================================================================
def scenario_A(t):
    """Bear Flattening (shock inflazionistico): +70bp fino a 2Y (verso
    il 4.3% dei futures), decadimento lineare a 0 entro il 10Y."""
    if t <= 2.0:
        return 0.0070
    elif t <= 10.0:
        return 0.0070 * (10.0 - t) / (10.0 - 2.0)
    return 0.0

def scenario_B(t):
    """Bear Steepening (crisi del debito): invariato fino a 2Y, rampa
    lineare fino a +60bp al 10Y, flat +60bp da 10Y a 30Y."""
    if t <= 2.0:
        return 0.0
    elif t <= 10.0:
        return 0.0060 * (t - 2.0) / (10.0 - 2.0)
    return 0.0060

SCENARIOS = {
    "Scenario A - Bear Flattening": scenario_A,
    "Scenario B - Bear Steepening": scenario_B,
}

# ====================================================================
# 5. TABELLE DI SINTESI (stampa a video + CSV)
# ====================================================================
def run_analytics():
    fair_value_base, breakdown_base = price_portfolio(detail=True)
    breakdown_base["weight_%"] = breakdown_base["value"] / fair_value_base * 100
    breakdown_base["YTM_%"] = [bond_ytm(n) * 100 for n in breakdown_base["node"]]

    print("=" * 78)
    print("1-2. PORTAFOGLIO BARBELL - FAIR VALUE (mark-to-market, curva spot)")
    print("=" * 78)
    tbl = breakdown_base.copy()
    tbl["face"] = tbl["face"].map(lambda x: f"${x:,.0f}")
    tbl["price_100"] = tbl["price_100"].round(4)
    tbl["value"] = tbl["value"].map(lambda x: f"${x:,.2f}")
    tbl["weight_%"] = tbl["weight_%"].round(2)
    tbl["YTM_%"] = tbl["YTM_%"].round(3)
    print(tbl.to_string(index=False))
    print(f"\nFair Value totale portafoglio: ${fair_value_base:,.2f}")
    print(f"(vs. Notional aggregato:       ${AGGREGATE_NOTIONAL:,.2f})")

    print("\n" + "=" * 78)
    print("3. METRICHE DI RISCHIO EFFETTIVE (Duration & Convexity, re-pricing)")
    print("=" * 78)
    risk_rows = []
    for pos in PORTFOLIO:
        pricer = lambda shock_fn, node=pos["node"], face=pos["face"]: price_bond(node, shock_fn) * face / 100.0
        p0, mdur, conv, dv01 = effective_risk_metrics(pricer)
        risk_rows.append({"Titolo": pos["node"], "Fair Value ($)": p0,
                           "Eff. Mod. Duration": mdur, "Eff. Convexity": conv, "DV01 ($/bp)": dv01})
    p0_pf, mdur_pf, conv_pf, dv01_pf = effective_risk_metrics(lambda s: price_portfolio(s))
    risk_rows.append({"Titolo": "PORTAFOGLIO", "Fair Value ($)": p0_pf,
                       "Eff. Mod. Duration": mdur_pf, "Eff. Convexity": conv_pf, "DV01 ($/bp)": dv01_pf})
    risk_df = pd.DataFrame(risk_rows)
    disp = risk_df.copy()
    disp["Fair Value ($)"] = disp["Fair Value ($)"].map(lambda x: f"${x:,.2f}")
    disp["Eff. Mod. Duration"] = disp["Eff. Mod. Duration"].round(3)
    disp["Eff. Convexity"] = disp["Eff. Convexity"].round(3)
    disp["DV01 ($/bp)"] = disp["DV01 ($/bp)"].map(lambda x: f"${x:,.2f}")
    print(disp.to_string(index=False))
    dv01_sum = risk_df.iloc[:-1]["DV01 ($/bp)"].sum()
    print(f"\nCheck additivita' DV01: somma posizioni = ${dv01_sum:,.2f}   |   portafoglio = ${dv01_pf:,.2f}")

    print("\n" + "=" * 78)
    print("4. SCENARIO ANALYSIS - IMPATTO SUL P&L DI PORTAFOGLIO")
    print("=" * 78)
    scenario_results = []
    for label, shock_fn in SCENARIOS.items():
        fv_shocked, breakdown_shocked = price_portfolio(shock_fn, detail=True)
        pnl = fv_shocked - fair_value_base
        scenario_results.append({"Scenario": label, "Fair Value shocked ($)": fv_shocked,
                                  "P&L ($)": pnl, "P&L (%)": pnl / fair_value_base * 100})
        breakdown_shocked["P&L ($)"] = breakdown_shocked["value"] - breakdown_base["value"]
        print(f"\n--- {label} ---")
        d = breakdown_shocked[["node", "price_100", "value", "P&L ($)"]].copy()
        d["price_100"] = d["price_100"].round(4)
        d["value"] = d["value"].map(lambda x: f"${x:,.2f}")
        d["P&L ($)"] = d["P&L ($)"].map(lambda x: f"${x:,.2f}")
        print(d.to_string(index=False))

    scenario_df = pd.DataFrame(scenario_results)
    print("\n--- Sintesi P&L di portafoglio ---")
    sdf = scenario_df.copy()
    sdf["Fair Value shocked ($)"] = sdf["Fair Value shocked ($)"].map(lambda x: f"${x:,.2f}")
    sdf["P&L ($)"] = sdf["P&L ($)"].map(lambda x: f"${x:,.2f}")
    sdf["P&L (%)"] = sdf["P&L (%)"].round(3)
    print(sdf.to_string(index=False))

    breakdown_base.to_csv("portfolio_fair_value.csv", index=False)
    risk_df.to_csv("risk_metrics_baseline.csv", index=False)
    scenario_df.to_csv("scenario_pnl_results.csv", index=False)
    print("\nCSV esportati nella cartella corrente.")

    return fair_value_base, breakdown_base, risk_df, scenario_df, mdur_pf, conv_pf, dv01_pf

# ====================================================================
# 6. DASHBOARD PRE-SHOCK (stile institutional report)
# ====================================================================
def render_dashboard(breakdown_base, risk_df, mdur_pf, conv_pf, dv01_pf,
                      filename="portfolio_risk_dashboard.png"):
    NAVY, GOLD, RED, GREY, LGREY = "#1F4E79", "#C9A24B", "#C00000", "#8C97A8", "#F2F2F2"
    leg_colors = ["#1F4E79", "#C00000", "#E3A324"]
    labels = breakdown_base["node"].tolist()
    mv = breakdown_base["value"].tolist()
    moddur = risk_df.iloc[:-1]["Eff. Mod. Duration"].tolist()
    convexity = risk_df.iloc[:-1]["Eff. Convexity"].tolist()
    dv01 = risk_df.iloc[:-1]["DV01 ($/bp)"].tolist()
    total_mv = sum(mv)
    ytm_pf = (breakdown_base["value"] * breakdown_base["YTM_%"]).sum() / total_mv

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(15, 9.7), dpi=200)
    fig.patch.set_facecolor("white")

    fig.text(0.045, 0.968, "Portfolio Risk Metrics — Current State (Pre-Shock)",
              fontsize=19, weight="bold", color=NAVY)
    fig.text(0.045, 0.942,
              f"Barbell Portfolio · Settlement {SETTLEMENT.strftime('%d/%m/%Y')} · "
              f"Notional \\${AGGREGATE_NOTIONAL/1e6:.1f}M · Market Value \\${total_mv/1e6:.3f}M",
              fontsize=10.5, color=GREY)

    # --- KPI cards ---
    kpi_w, kpi_gap, kpi_y, kpi_h = 0.215, 0.02, 0.79, 0.11
    kpis = [
        ("MODIFIED DURATION (EFF.)", f"{mdur_pf:.2f}y", NAVY),
        ("CONVEXITY (EFF.)", f"{conv_pf:.1f}", RED),
        ("PORTFOLIO DV01", f"${dv01_pf:,.1f}", GOLD),
        ("YTM (MV-WEIGHTED)", f"{ytm_pf:.2f}%", "#555555"),
    ]
    for i, (label, value, color) in enumerate(kpis):
        x0 = 0.045 + i * (kpi_w + kpi_gap)
        ax = fig.add_axes([x0, kpi_y, kpi_w, kpi_h])
        ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.06",
                                              transform=ax.transAxes, facecolor=LGREY,
                                              edgecolor="#D9D9D9", linewidth=1))
        ax.add_patch(mpatches.Rectangle((0, 0), 0.025, 1, transform=ax.transAxes, facecolor=color, edgecolor="none"))
        ax.text(0.13, 0.62, label, transform=ax.transAxes, fontsize=8.7, color="#555555", weight="bold", va="center")
        ax.text(0.13, 0.28, value, transform=ax.transAxes, fontsize=19, color=color, weight="bold", va="center")

    # --- Tabella per leg ---
    ax_tab = fig.add_axes([0.045, 0.44, 0.40, 0.30])
    ax_tab.axis("off")
    col_labels = ["Leg", "Weight", "MV ($k)", "ModDur", "Convexity", "DV01"]
    weights = breakdown_base["weight_%"].tolist()
    cell_text = [[labels[i], f"{weights[i]:.0f}%", f"{mv[i]/1000:,.0f}",
                  f"{moddur[i]:.2f}", f"{convexity[i]:.1f}", f"${dv01[i]:.1f}"] for i in range(len(labels))]
    cell_text.append(["TOTAL", "100%", f"{total_mv/1000:,.0f}",
                       f"{mdur_pf:.2f}", f"{conv_pf:.1f}", f"${dv01_pf:.1f}"])
    tbl = ax_tab.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.9)
    nrows = len(cell_text)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        if row == 0:
            cell.set_facecolor(NAVY); cell.set_text_props(weight="bold", color="white")
        elif row == nrows:
            cell.set_facecolor(LGREY); cell.set_text_props(weight="bold", color=NAVY)
        else:
            cell.set_facecolor("white")
    ax_tab.set_title("Composition by Leg", fontsize=11.5, weight="bold", color=NAVY, loc="left", pad=10)

    # --- Contributo % a Duration/Convexity ---
    ax_contrib = fig.add_axes([0.545, 0.44, 0.41, 0.30])
    dur_contrib = [mv[i] * moddur[i] / total_mv / mdur_pf * 100 for i in range(len(labels))]
    conv_contrib = [mv[i] * convexity[i] / total_mv / conv_pf * 100 for i in range(len(labels))]
    y_pos = np.arange(len(labels))
    bar_h = 0.35
    ax_contrib.barh(y_pos + bar_h/2, dur_contrib, height=bar_h, color=NAVY, label="% contribution to Duration")
    ax_contrib.barh(y_pos - bar_h/2, conv_contrib, height=bar_h, color=RED, label="% contribution to Convexity")
    for i, (d, c) in enumerate(zip(dur_contrib, conv_contrib)):
        ax_contrib.text(d + 1.5, i + bar_h/2, f"{d:.0f}%", va="center", fontsize=9, color=NAVY, weight="bold")
        ax_contrib.text(c + 1.5, i - bar_h/2, f"{c:.0f}%", va="center", fontsize=9, color=RED, weight="bold")
    ax_contrib.set_yticks(y_pos)
    ax_contrib.set_yticklabels(labels, fontsize=9.5)
    ax_contrib.set_xlim(0, 115)
    ax_contrib.set_xlabel("% of total portfolio", fontsize=9.5)
    ax_contrib.spines[["top", "right"]].set_visible(False)
    ax_contrib.grid(axis="x", linestyle="--", alpha=0.4)
    ax_contrib.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax_contrib.set_title("Risk Concentration: What Drives Duration and Convexity",
                          fontsize=11.5, weight="bold", color=NAVY, loc="left", pad=10)

    # --- Taylor: Duration-only vs Duration+Convexity (shock parallelo) ---
    ax_taylor = fig.add_axes([0.045, 0.065, 0.44, 0.30])
    def price_change_pct(delta_y, use_convexity=True):
        lin = -mdur_pf * delta_y
        return lin + 0.5 * conv_pf * delta_y**2 if use_convexity else lin
    shocks_bp = np.linspace(-250, 250, 200)
    dy = shocks_bp / 10000
    lin_pct = np.array([price_change_pct(d, False) for d in dy]) * 100
    quad_pct = np.array([price_change_pct(d, True) for d in dy]) * 100
    ax_taylor.plot(shocks_bp, lin_pct, color=GREY, linewidth=2, linestyle="--", label="Duration only (linear)")
    ax_taylor.plot(shocks_bp, quad_pct, color=NAVY, linewidth=2.6, label="Duration + Convexity (quadratic)")
    ax_taylor.fill_between(shocks_bp, lin_pct, quad_pct, color=GOLD, alpha=0.25, label="Convexity effect")
    ax_taylor.axhline(0, color="#CCCCCC", linewidth=0.8)
    ax_taylor.axvline(0, color="#CCCCCC", linewidth=0.8)
    for sb in [-100, 100, 200]:
        idx = (np.abs(shocks_bp - sb)).argmin()
        ax_taylor.scatter([sb], [quad_pct[idx]], color=RED, s=28, zorder=5)
        ax_taylor.annotate(f"{quad_pct[idx]:+.2f}%", (sb, quad_pct[idx]), textcoords="offset points",
                            xytext=(0, 9 if sb < 0 else -16), fontsize=8.3, ha="center", color=RED, weight="bold")
    ax_taylor.set_xlabel("Parallel curve shock (bp)", fontsize=9.5)
    ax_taylor.set_ylabel("Portfolio price change (%)", fontsize=9.5)
    ax_taylor.spines[["top", "right"]].set_visible(False)
    ax_taylor.grid(alpha=0.3, linestyle="--")
    ax_taylor.legend(loc="upper right", fontsize=8.3, frameon=False)
    ax_taylor.set_title("Price Estimate: Linear vs Quadratic Approximation (parallel shock)",
                         fontsize=11.5, weight="bold", color=NAVY, loc="left", pad=10)

    # --- Box di commento ---
    ax_note = fig.add_axes([0.545, 0.065, 0.41, 0.30])
    ax_note.axis("off")
    ax_note.add_patch(mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.03",
                                               transform=ax_note.transAxes, facecolor=LGREY,
                                               edgecolor="#D9D9D9", linewidth=1))
    ax_note.add_patch(mpatches.FancyBboxPatch((0, 0), 0.014, 1, boxstyle="round,pad=0,rounding_size=0.03",
                                               transform=ax_note.transAxes, facecolor=NAVY, edgecolor="none"))
    idx100 = np.abs(shocks_bp - 100).argmin()
    long_leg = labels[2]
    note_lines = [
        ("KEY POINTS", NAVY, 10.5, True),
        ("", None, 4, False),
        (f"- Duration {mdur_pf:.2f}y is {dur_contrib[2]:.0f}% driven by the {long_leg}", "#333333", 9.3, False),
        (f"  (only {weights[2]:.0f}% of notional).", "#333333", 9.3, False),
        (f"- Convexity is {conv_contrib[2]:.0f}% concentrated on the same bond:", "#333333", 9.3, False),
        ("  the barbell draws its convexity almost", "#333333", 9.3, False),
        ("  entirely from the long leg.", "#333333", 9.3, False),
        ("- Practical effect: on a +100bp parallel shock, the", "#333333", 9.3, False),
        (f"  actual loss is {quad_pct[idx100]:+.2f}% vs. {lin_pct[idx100]:+.2f}% estimated", "#333333", 9.3, False),
        ("  by duration alone: convexity cushions losses", "#333333", 9.3, False),
        ("  and amplifies gains (asymmetry).", "#333333", 9.3, False),
        ("- Caveat: historical shocks (bear flattening/", "#333333", 9.3, False),
        ("  steepening) are NOT parallel - see Scenario", "#333333", 9.3, False),
        ("  Analysis: needs full reval on the whole curve.", "#333333", 9.3, False),
    ]
    y0 = 0.93
    for text, color, size, bold in note_lines:
        if text == "":
            y0 -= 0.045
            continue
        ax_note.text(0.06, y0, text, transform=ax_note.transAxes, fontsize=size,
                     color=color if color else "#333333", weight="bold" if bold else "normal", va="top")
        y0 -= 0.075

    plt.savefig(filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Dashboard salvata: {filename}")

# ====================================================================
# 7. GRAFICI DELLO STRESS TEST (post-shock)
# ====================================================================
def render_stress_test_charts(fair_value_base, breakdown_base, mdur_pf, conv_pf,
                               filename="scenario_stress_test_dashboard.png"):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
        "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.45,
        "grid.color": "#B0B0B0", "axes.axisbelow": True,
    })
    BLUE, RED, GREY = "#4C72B0", "#C44E52", "#555555"

    breakdown_base_idx = breakdown_base.set_index("node")
    scenario_labels = ["Scenario A\n(Bear Flattening)", "Scenario B\n(Bear Steepening)"]
    scenario_fns = [scenario_A, scenario_B]

    pnl_by_position = {pos["node"]: [] for pos in PORTFOLIO}
    pnl_portfolio, pnl_portfolio_pct = [], []
    for shock_fn in scenario_fns:
        fv_shocked, breakdown_shocked = price_portfolio(shock_fn, detail=True)
        breakdown_shocked = breakdown_shocked.set_index("node")
        pnl_portfolio.append(fv_shocked - fair_value_base)
        pnl_portfolio_pct.append((fv_shocked - fair_value_base) / fair_value_base * 100)
        for node in pnl_by_position:
            pnl_by_position[node].append(breakdown_shocked.loc[node, "value"] - breakdown_base_idx.loc[node, "value"])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.patch.set_facecolor("white")

    # Pannello 1: profilo di shock
    ax = axes[0]
    t_grid = np.linspace(0, 30, 300)
    ax.plot(t_grid, [scenario_A(t) * 10000 for t in t_grid], color=BLUE, linewidth=2.2, label="Scenario A - Bear Flattening")
    ax.plot(t_grid, [scenario_B(t) * 10000 for t in t_grid], color=RED, linewidth=2.2, label="Scenario B - Bear Steepening")
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.axvline(2, color=GREY, linewidth=0.8, linestyle=":")
    ax.axvline(10, color=GREY, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Maturity (years)", fontsize=10)
    ax.set_ylabel("Spot rate shock (bp)", fontsize=10)
    ax.set_title("Spot Curve Shock Profile", fontsize=12.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, edgecolor="#999999")
    ax.set_xlim(0, 30)

    # Pannello 2: P&L per posizione
    ax = axes[1]
    nodes = [pos["node"] for pos in PORTFOLIO]
    x = np.arange(len(nodes))
    width = 0.32
    vals_A = [pnl_by_position[n][0] for n in nodes]
    vals_B = [pnl_by_position[n][1] for n in nodes]
    bars_A = ax.bar(x - width/2, vals_A, width, color=BLUE, edgecolor="white", linewidth=0.6, label="Scenario A")
    bars_B = ax.bar(x + width/2, vals_B, width, color=RED, edgecolor="white", linewidth=0.6, label="Scenario B")
    ax.axhline(0, color=GREY, linewidth=0.9)
    ax.set_yscale("symlog", linthresh=200)
    for bars in (bars_A, bars_B):
        for bar in bars:
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            ax.annotate(f"${y:,.0f}", (bar.get_x() + bar.get_width()/2, y),
                        textcoords="offset points", xytext=(0, 4 if y >= 0 else -4),
                        ha="center", va=va, fontsize=8.5, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(nodes, fontsize=9.5)
    ax.set_ylabel("P&L by position ($, symmetric-log scale)", fontsize=9.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title("P&L by Position (Barbell)", fontsize=12.5, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, edgecolor="#999999")
    ax.set_ylim(-60000, 3000)

    # Pannello 3: P&L aggregato
    ax = axes[2]
    bars = ax.bar(scenario_labels, pnl_portfolio, width=0.5, color=[BLUE, RED], edgecolor="white", linewidth=0.8)
    ax.axhline(0, color=GREY, linewidth=0.9)
    for bar, val, pct in zip(bars, pnl_portfolio, pnl_portfolio_pct):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        ax.annotate(f"${val:,.0f}\n({pct:+.2f}%)", (bar.get_x() + bar.get_width()/2, y),
                    textcoords="offset points", xytext=(0, 5 if y >= 0 else -5),
                    ha="center", va=va, fontsize=10, fontweight="bold", color="#222222")
    ax.set_ylabel("Portfolio P&L ($)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title("Aggregate Portfolio P&L", fontsize=12.5, fontweight="bold")
    ymax = max(abs(v) for v in pnl_portfolio) * 1.4
    ax.set_ylim(-ymax, ymax * 0.25)

    fig.suptitle("Macroeconomic Stress Test — Barbell Portfolio (Notional $1,000,000)",
                 fontsize=15, fontweight="bold", y=1.06)
    fig.text(0.5, 1.005,
              f"Effective Mod. Duration: {mdur_pf:.2f}y   |   Effective Convexity: {conv_pf:.2f}   |   "
              f"Settlement: {SETTLEMENT.strftime('%d/%m/%Y')}   |   Source: bootstrapped spot curve (CubicSpline)",
              ha="center", fontsize=9.5, color=GREY)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(filename, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico stress test salvato: {filename}")

# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    fair_value_base, breakdown_base, risk_df, scenario_df, mdur_pf, conv_pf, dv01_pf = run_analytics()
    render_dashboard(breakdown_base, risk_df, mdur_pf, conv_pf, dv01_pf)
    render_stress_test_charts(fair_value_base, breakdown_base, mdur_pf, conv_pf)
