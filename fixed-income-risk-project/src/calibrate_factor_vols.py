"""
====================================================================
calibrate_factor_vols.py
====================================================================
Ristima ASSUMED_VOL_LEVEL / SLOPE / CURVE / IDIO di advanced_risk_
analytics.py da dati storici REALI (non piu' placeholder).

Fonte dati: U.S. Department of the Treasury, "Daily Treasury Par
Yield Curve Rates" (CMT), scaricate da home.treasury.gov il
09/09/2026, per gli anni 2025 e 2026 (2 gennaio 2025 - 8 settembre
2026, ~440 osservazioni giornaliere). File salvati localmente in
treasury_2025.csv / treasury_2026.csv.

Metodologia:
  1. Le 12 colonne CMT (1, 2, 3, 6 mesi; 1, 2, 3, 5, 7, 10, 20, 30
     anni) corrispondono esattamente, tenor per tenor, ai 12 nodi
     bootstrappati KEY_RATE_NODES del progetto (T-Bill 1M...T-Bond
     30Y). Nessuna interpolazione necessaria.
  2. Variazioni giornaliere in bp: delta_y_t = 100 * (y_t - y_{t-1}).
  3. Si mantiene la STESSA struttura fattoriale del modello (Level
     piatto, Slope log-maturity, Curvature a gobba, gia' definita in
     build_factor_loadings()) invece di sostituirla con una PCA pura:
     e' l'approccio standard "loadings fissi, fattori stimati via
     regressione" (Litterman-Scheinkman operativo). Ad ogni giorno t,
     si stima [f_level, f_slope, f_curve]_t via OLS multivariata:
         F = ΔY @ L @ (L'L)^-1          (L = [level|slope|curvature], 12x3)
     Le vol di fattore sono poi la std storica delle serie F. Il
     residuo (ΔY - F@L') fornisce la vol idiosincratica per-nodo.
  4. Finestra primaria: ultimi 252 giorni di trading (~1Y, termina
     08/09/2026) - la piu' rilevante per il regime corrente. Riportata
     anche la finestra intera (~1.7Y) per controllo di robustezza.
====================================================================
"""
import numpy as np
import pandas as pd

from advanced_risk_analytics import build_factor_loadings, KEY_RATE_NODES, KEY_RATE_LABELS, N_NODES

# --------------------------------------------------------------
# 1. Caricamento e merge delle serie storiche Treasury
# --------------------------------------------------------------
COL_MAP = {
    "1 Mo": "1M", "2 Mo": "2M", "3 Mo": "3M", "6 Mo": "6M", "1 Yr": "12M",
    "2 Yr": "2Y", "3 Yr": "3Y", "5 Yr": "5Y", "7 Yr": "7Y", "10 Yr": "10Y",
    "20 Yr": "20Y", "30 Yr": "30Y",
}
ORDERED_COLS = ["1M", "2M", "3M", "6M", "12M", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
assert ORDERED_COLS == KEY_RATE_LABELS, "Ordine tenor non allineato a KEY_RATE_LABELS del modello"

def load_treasury_history(paths=("../data/treasury_2025.csv", "../data/treasury_2026.csv")):
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.rename(columns=COL_MAP)[["Date"] + ORDERED_COLS]
    df = df.sort_values("Date").reset_index(drop=True)
    df[ORDERED_COLS] = df[ORDERED_COLS].apply(pd.to_numeric, errors="coerce")
    return df

# --------------------------------------------------------------
# 2. Regressione fattoriale (loadings fissi del modello) su ΔY
# --------------------------------------------------------------
def factor_scores(delta_y: np.ndarray, L: np.ndarray):
    """OLS multivariata riga per riga: F (T x 3) = ΔY (T x 12) @ L @ (L'L)^-1.
    Ritorna anche i residui (T x 12)."""
    LtL_inv = np.linalg.inv(L.T @ L)
    F = delta_y @ L @ LtL_inv
    resid = delta_y - F @ L.T
    return F, resid

def calibrate(window_label: str, delta_y: np.ndarray, L: np.ndarray):
    F, resid = factor_scores(delta_y, L)
    vol_level, vol_slope, vol_curve = F.std(axis=0, ddof=1)
    idio_per_node = resid.std(axis=0, ddof=1)          # 12 vol idiosincratiche (bp/g)
    vol_idio = np.sqrt(np.mean(idio_per_node ** 2))    # scalare unico, per coerenza col modello

    corr = np.corrcoef(F.T)
    total_var_explained = 1 - (resid.var(axis=0, ddof=1).sum() / delta_y.var(axis=0, ddof=1).sum())

    print(f"\n--- Finestra: {window_label} (n={len(delta_y)} variazioni giornaliere) ---")
    print(f"  Vol Level     : {vol_level:.3f} bp/g")
    print(f"  Vol Slope     : {vol_slope:.3f} bp/g")
    print(f"  Vol Curvature : {vol_curve:.3f} bp/g")
    print(f"  Vol Idio (RMS per nodo): {vol_idio:.3f} bp/g")
    print(f"  Correlazione fra fattori stimati (off-diag, dovrebbe essere piccola):")
    print(f"    Level-Slope={corr[0,1]:+.3f}  Level-Curve={corr[0,2]:+.3f}  Slope-Curve={corr[1,2]:+.3f}")
    print(f"  Quota di varianza totale (12 nodi) spiegata dai 3 fattori: {total_var_explained:.1%}")
    print(f"  Vol idiosincratica per nodo (bp/g): " +
          ", ".join(f"{lab}={v:.2f}" for lab, v in zip(KEY_RATE_LABELS, idio_per_node)))
    return {"vol_level": vol_level, "vol_slope": vol_slope, "vol_curve": vol_curve,
            "vol_idio": vol_idio, "corr": corr, "var_explained": total_var_explained,
            "n_obs": len(delta_y)}

if __name__ == "__main__":
    df = load_treasury_history()
    print(f"Serie storica caricata: {df['Date'].min().date()} -> {df['Date'].max().date()} "
          f"({len(df)} osservazioni giornaliere)")

    level, slope, curvature = build_factor_loadings()
    L = np.column_stack([level, slope, curvature])

    y = df[ORDERED_COLS].to_numpy()
    delta_y_full = 100.0 * np.diff(y, axis=0)   # percentage points -> bp

    res_full = calibrate("intera (~1.7Y, da gen-2025)", delta_y_full, L)
    res_1y = calibrate("trailing 1Y (ultimi 252 gg)", delta_y_full[-252:], L)

    print("\n" + "=" * 78)
    print("Valori scelti per advanced_risk_analytics.py (finestra trailing 1Y, "
          "piu' rilevante per il regime corrente):")
    print(f"  ASSUMED_VOL_LEVEL = {res_1y['vol_level']:.2f}")
    print(f"  ASSUMED_VOL_SLOPE = {res_1y['vol_slope']:.2f}")
    print(f"  ASSUMED_VOL_CURVE = {res_1y['vol_curve']:.2f}")
    print(f"  ASSUMED_VOL_IDIO  = {res_1y['vol_idio']:.2f}")
