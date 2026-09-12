"""
====================================================================
bootstrap_yield_curve.py
====================================================================
Costruisce la curva zero-coupon (spot curve) dei Titoli di Stato USA
a partire da T-Bills (zero-coupon, clean price) e T-Notes/T-Bonds
(cedolari, mid-price), poi la estende con una curva continua
(CubicSpline) usata dal secondo script (portfolio_risk_stress_test.py)
per il pricing esatto e lo stress testing.

Eseguito direttamente (`python bootstrap_yield_curve.py`) stampa la
tabella dei nodi bootstrappati e salva il grafico della curva.
Importato da un altro script, espone: NODES_INPUT, SETTLEMENT, FACE,
year_frac_act365, coupon_schedule, accrued_interest, build_continuous_curve().

Metodologia:
  1. T-Bills: flusso unico -> discount factor diretto dal clean price.
  2. T-Notes/T-Bonds: scadenzario cedolare semestrale (Actual/Actual
     ICMA), Dirty Price = Clean/Mid Price + rateo; le cedole intermedie
     sono scontate con la curva bootstrappata fino a quel punto,
     l'ultimo flusso risolve il nuovo discount factor.
  3. Oltre l'ultimo nodo bootstrappato, estrapolazione FLAT sull'ultimo
     tasso zero (sia in fase di bootstrap sia nella curva continua).

Convenzioni: time fraction curva Actual/365; accrued interest
Actual/Actual (ICMA) semestrale; valuation date = settlement dei T-Bill.
====================================================================
"""

import math
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np
from scipy.interpolate import CubicSpline

FACE = 100.0
SETTLEMENT = date(2026, 9, 2)  # dal foglio "T-Bills data" del dataset WSJ

# Nodi selezionati (verificati contro le celle evidenziate in giallo
# nei fogli "T-Bills data" / "T-Bonds data" del dataset)
NODES_INPUT = [
    # name,          maturity,          coupon,   price(clean/mid),  freq, type
    ("T-Bill 1M",   date(2026, 10, 1),  0.0,      99.703,   0, "bill"),
    ("T-Bill 2M",   date(2026, 11, 3),  0.0,      99.356,   0, "bill"),
    ("T-Bill 3M",   date(2026, 12, 1),  0.0,      99.065,   0, "bill"),
    ("T-Bill 6M",   date(2027, 3, 4),   0.0,      98.020,   0, "bill"),
    ("T-Bill 12M",  date(2027, 8, 5),   0.0,      96.274,   0, "bill"),
    ("T-Note 2Y",   date(2028, 9, 30),  0.04625, 100.135,   2, "bond"),
    ("T-Note 3Y",   date(2029, 6, 30),  0.04250,  99.145,   2, "bond"),
    ("T-Note 5Y",   date(2031, 7, 31),  0.04375,  99.065,   2, "bond"),
    ("T-Note 7Y",   date(2033, 8, 31),  0.04500,  98.657,   2, "bond"),
    ("T-Note 10Y",  date(2036, 8, 15),  0.04625,  98.204,   2, "bond"),
    ("T-Bond 20Y",  date(2046, 8, 15),  0.05125,  98.060,   2, "bond"),
    ("T-Bond 30Y",  date(2056, 8, 15),  0.05125,  97.270,   2, "bond"),
]

# --------------------------------------------------------------
# Utility di calendario
# --------------------------------------------------------------
def year_frac_act365(d0: date, d1: date) -> float:
    return (d1 - d0).days / 365.0

def coupon_schedule(maturity: date, settlement: date, freq: int):
    """Scadenzario cedolare a ritroso dalla maturity. Restituisce
    (data ultimo stacco <= settlement, lista date future > settlement)."""
    if freq == 0:
        return settlement, []
    step = 12 // freq
    dates = [maturity]
    d = maturity
    while d > settlement:
        d = d - relativedelta(months=step)
        dates.append(d)
    dates = sorted(set(dates))
    prev_coupon = max(d for d in dates if d <= settlement)
    future_dates = [d for d in dates if d > settlement]
    return prev_coupon, future_dates

def accrued_interest(maturity: date, settlement: date, coupon: float, freq: int) -> float:
    """Rateo Actual/Actual (ICMA) su base semestrale."""
    if freq == 0:
        return 0.0
    prev_coupon, future_dates = coupon_schedule(maturity, settlement, freq)
    if not future_dates:
        return 0.0
    next_coupon = future_dates[0]
    period_days = (next_coupon - prev_coupon).days
    accrued_days = (settlement - prev_coupon).days
    coupon_cash = FACE * coupon / freq
    return coupon_cash * (accrued_days / period_days)

# --------------------------------------------------------------
# Curva discreta di supporto al bootstrap (interpolazione lineare sui
# tassi zero, estrapolazione flat) - alimenta la ContinuousCurve finale
# --------------------------------------------------------------
class _BootstrapCurve:
    def __init__(self):
        self.t_nodes = [0.0]
        self.df_nodes = [1.0]

    def zero_rate(self, t):
        if t <= self.t_nodes[0]:
            return 0.0 if len(self.t_nodes) == 1 else self._z(self.t_nodes[1], self.df_nodes[1])
        if t >= self.t_nodes[-1]:
            return self._z(self.t_nodes[-1], self.df_nodes[-1])
        for i in range(1, len(self.t_nodes)):
            if self.t_nodes[i-1] <= t <= self.t_nodes[i]:
                t0, t1 = self.t_nodes[i-1], self.t_nodes[i]
                z0 = self._z(t0, self.df_nodes[i-1]) if t0 > 0 else self._z(t1, self.df_nodes[i])
                z1 = self._z(t1, self.df_nodes[i])
                if t1 == t0:
                    return z1
                w = (t - t0) / (t1 - t0)
                return z0 + w * (z1 - z0)
        return self._z(self.t_nodes[-1], self.df_nodes[-1])

    @staticmethod
    def _z(t, df):
        return -math.log(df) / t if t > 0 else 0.0

    def discount_factor(self, t):
        if t <= 0:
            return 1.0
        return math.exp(-self.zero_rate(t) * t)

    def add_node(self, t, df):
        self.t_nodes.append(t)
        self.df_nodes.append(df)
        order = sorted(range(len(self.t_nodes)), key=lambda i: self.t_nodes[i])
        self.t_nodes = [self.t_nodes[i] for i in order]
        self.df_nodes = [self.df_nodes[i] for i in order]

def bootstrap_curve(nodes_input=NODES_INPUT, settlement=SETTLEMENT):
    """Esegue il bootstrap e restituisce (t_nodes, z_nodes, results)
    dove results e' la lista di dict pronta per una tabella/DataFrame."""
    curve = _BootstrapCurve()
    t_nodes, z_nodes, results = [], [], []

    for name, maturity, coupon, price, freq, kind in nodes_input:
        T = year_frac_act365(settlement, maturity)

        if kind == "bill":
            df = price / FACE
            curve.add_node(T, df)
            ai, dirty_price = 0.0, price
        else:
            ai = accrued_interest(maturity, settlement, coupon, freq)
            dirty_price = price + ai
            _, future_dates = coupon_schedule(maturity, settlement, freq)
            coupon_cash = FACE * coupon / freq
            pv_interim = sum(
                coupon_cash * curve.discount_factor(year_frac_act365(settlement, d))
                for d in future_dates[:-1]
            )
            final_cf = coupon_cash + FACE
            df = (dirty_price - pv_interim) / final_cf
            curve.add_node(T, df)

        z = -math.log(df) / T
        t_nodes.append(T)
        z_nodes.append(z)
        results.append({
            "Titolo": name, "Scadenza": maturity, "T (anni, ACT/365)": round(T, 4),
            "Prezzo (clean/mid)": price, "Rateo (AI)": round(ai, 4),
            "Dirty Price": round(dirty_price, 4), "Discount Factor": round(df, 6),
            "Zero Rate cont.comp (%)": round(z * 100, 4),
            "Zero Rate annual.comp (%)": round((math.exp(z) - 1) * 100, 4),
        })

    return t_nodes, z_nodes, results

# --------------------------------------------------------------
# Curva continua: CubicSpline sui tassi zero bootstrappati, con
# supporto a shock per la scenario analysis (usata dal secondo script)
# --------------------------------------------------------------
class ContinuousCurve:
    """
    Curva zero-coupon continua (CubicSpline naturale sui nodi bootstrappati,
    estrapolazione FLAT fuori range). shock_fn: callable t -> shock in
    decimali (es. 0.0070 = +70bp), usato per shiftare la curva negli
    scenari di stress senza toccare i nodi bootstrappati originali.
    """
    def __init__(self, t_nodes, z_nodes):
        self.t_min, self.t_max = min(t_nodes), max(t_nodes)
        self._spline = CubicSpline(t_nodes, z_nodes, bc_type="natural")

    def spot_rate(self, t, shock_fn=None):
        t_clip = min(max(t, self.t_min), self.t_max)
        z = float(self._spline(t_clip))
        if shock_fn is not None:
            z += shock_fn(t)
        return z

    def discount_factor(self, t, shock_fn=None):
        if t <= 0:
            return 1.0
        return math.exp(-self.spot_rate(t, shock_fn) * t)

def build_continuous_curve(nodes_input=NODES_INPUT, settlement=SETTLEMENT):
    t_nodes, z_nodes, results = bootstrap_curve(nodes_input, settlement)
    return ContinuousCurve(t_nodes, z_nodes), t_nodes, z_nodes, results

# ====================================================================
# MAIN: eseguito solo se lo script gira direttamente (non se importato)
# ====================================================================
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.interpolate import PchipInterpolator

    curve, t_nodes, z_nodes, results = build_continuous_curve()
    df_results = pd.DataFrame(results)
    pd.set_option("display.width", 140)
    print(df_results.to_string(index=False))
    df_results.to_csv("yield_curve_bootstrap.csv", index=False)

    x = df_results["T (anni, ACT/365)"].values
    y = df_results["Zero Rate annual.comp (%)"].values
    labels = [n.replace("T-Bill ", "").replace("T-Note ", "").replace("T-Bond ", "")
              for n in df_results["Titolo"]]

    x_smooth = np.linspace(x.min(), x.max(), 400)
    y_smooth = PchipInterpolator(x, y)(x_smooth)  # solo estetico: nessun overshoot tra i nodi

    plt.rcParams["font.family"] = "DejaVu Sans"
    NAVY, GOLD, GREY = "#1B2A4A", "#C9A24B", "#8C97A8"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.fill_between(x_smooth, y_smooth, y_smooth.min() - 0.3, color=NAVY, alpha=0.06, zorder=1)
    ax.plot(x_smooth, y_smooth, color=NAVY, linewidth=2.2, zorder=2, solid_capstyle="round")
    ax.scatter(x, y, s=55, color="white", edgecolor=NAVY, linewidth=1.8, zorder=3)
    ax.scatter(x, y, s=10, color=GOLD, zorder=4)
    ax.axvline(1.0, color=GREY, linewidth=0.8, linestyle=(0, (4, 3)), alpha=0.6, zorder=1)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(0, 11),
                    fontsize=8.5, ha="center", color=NAVY, fontweight="medium")

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0.08, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    ax.set_xticklabels(["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"])
    ax.minorticks_off()
    ax.set_xlabel("Maturity", fontsize=10.5, color=NAVY, labelpad=10)
    ax.set_ylabel("Zero-coupon rate (%, annual compounding)", fontsize=10.5, color=NAVY, labelpad=10)
    fig.suptitle("Zero-Coupon Yield Curve — US Treasury Securities",
                 x=0.045, y=0.975, ha="left", fontsize=14, color=NAVY, fontweight="bold")
    fig.text(0.047, 0.912, f"Bootstrapping · settlement {SETTLEMENT.strftime('%d/%m/%Y')} · Actual/365",
              ha="left", fontsize=9.5, color=GREY)
    ax.grid(axis="y", color=GREY, alpha=0.25, linewidth=0.7)
    ax.grid(axis="x", visible=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(axis="both", length=0, labelsize=9, colors=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    pad = (y.max() - y.min()) * 0.25
    ax.set_ylim(y.min() - pad, y.max() + pad * 1.4)

    plt.tight_layout(rect=[0, 0, 1, 0.87])
    plt.savefig("yield_curve_bootstrap.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Grafico curva salvato: yield_curve_bootstrap.png")