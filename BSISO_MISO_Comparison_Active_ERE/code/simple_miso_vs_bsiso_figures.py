# Simple, readable MISO vs BSISO figures and tables (at most 4 lines per plot, one score per plot).
# Reads the tables already made by R8 / R8 follow-ups / R9e; computes nothing new.
# Output: sem_project_7/MISO_vs_BSISO_results/simple/  (S1-S6 png + MISO_vs_BSISO_simple_tables.md)
import os, matplotlib
matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

SP7 = r"C:\Users\divya\OneDrive\Desktop\IIT- Bombay Summer 26\Moaap\Regional_claude\sem_project_7"
I = os.path.join(SP7, "other", "intermediate")
OUT = os.path.join(SP7, "MISO_vs_BSISO_results", "simple")
os.makedirs(OUT, exist_ok=True)
rd = lambda p: pd.read_csv(os.path.join(I, p))
BL, OR, GR, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#8c8b86", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                     "axes.titlelocation": "left", "axes.titlesize": 15, "axes.titleweight": "bold",
                     "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "legend.fontsize": 12, "lines.linewidth": 2.5})


def end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x[-1], y[-1]), xytext=(8, dy), textcoords="offset points", va="center", fontsize=12, color=INK)


def save(fig, name, note=None):
    if note:
        fig.text(0.01, -0.01, note, fontsize=10.5, color=INK2, ha="left", va="top")
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- S1: how far ahead does each index forecast itself
IX = rd("bsiso_forecast_tests/R8/R8_index_scores.csv")
S1 = [("MISO T", OR, "-", "MISO forecast"), ("BSISO T", BL, "-", "BSISO forecast"),
      ("persistence MISO", OR, "--", "MISO persistence"), ("persistence BSISO", BL, "--", "BSISO persistence")]
fig, ax = plt.subplots(figsize=(10, 5.5))
for m, c, ls, lab in S1:
    d = IX[IX.model == m].sort_values("lead")
    ax.plot(d.lead, d.correlation, color=c, ls=ls, label=lab)
ax.axhline(0.5, color=INK2, lw=1, ls=":"); ax.text(1, 0.52, "useful (ACC 0.5)", fontsize=11, color=INK2)
ax.axhline(0, color=INK2, lw=1)
ax.set(xlabel="forecast lead (days)", ylabel="ACC (forecast vs observed index)", xlim=(1, 20.5), xticks=[1, 5, 10, 15, 20])
ax.set_title("1. MISO forecasts itself much further ahead than BSISO")
ax.legend(frameon=False, loc="lower left")
save(fig, "S1_index_forecast_ACC.png", "Same network (transformer), same days (May-Oct 1998-2020). ACC 0.5 reached at: MISO 13.4 d, BSISO 6.2 d.")

# ---------------------------------------------------------------- S2: daily ERE skill
RS = rd("bsiso_forecast_tests/R8/R8_rain_scores.csv")
S2 = [("true MISO (target day)", OR, "--", "perfect MISO on the ERE day (needs the future)"),
      ("true BSISO1 (5 d before)", BL, "--", "perfect BSISO 5 days before"),
      ("MISO T", OR, "-", "MISO forecast"), ("BSISO T", BL, "-", "BSISO forecast")]
fig, ax = plt.subplots(figsize=(10, 5.5))
for m, c, ls, lab in S2:
    d = RS[RS.model == m].sort_values("lead")
    ax.plot(d.lead, d.RPSS, color=c, ls=ls, label=lab, marker="o" if ls == "-" else None, ms=7)
ax.axhline(0, color=INK2, lw=1)
ax.set(xlabel="rain lead (days)", ylabel="RPSS (0 = normal chance)", xticks=range(1, 15), ylim=(-0.005, 0.092))
ax.set_title("2. Daily ERE skill: MISO only wins in the first ~4 days")
ax.legend(frameon=False, loc="upper right")
save(fig, "S2_daily_ERE_RPSS.png", "Central-India ERE, 4 size classes, JJAS 2000-2020 (2,562 days). At 8 d: MISO forecast 0.022 worse than BSISO "
     "[90% range -0.033, -0.012].")

# ---------------------------------------------------------------- S3: weekly ERE skill
WS = rd("R8_miso/compare/weekly_scores.csv")
fig, ax = plt.subplots(figsize=(9, 5.5))
wins = ["week 1 (days 1-7)", "week 2 (days 8-14)"]
for i, (m, c, lab) in enumerate((("BSISO T", BL, "BSISO forecast"), ("MISO T", OR, "MISO forecast"))):
    v = [WS[(WS.window == w) & (WS.model == m)]["RPSS (0/1/2+)"].iloc[0] for w in wins]
    xs = np.arange(2) + (i - 0.5) * 0.36
    ax.bar(xs, v, width=0.34, color=c, label=lab, edgecolor=SURF, linewidth=2)
    for x, y in zip(xs, v):
        ax.text(x, y + (0.003 if y >= 0 else -0.009), f"{y:.3f}", ha="center", fontsize=13)
ax.axhline(0, color=INK2, lw=1)
ax.set(xticks=range(2), xticklabels=["week 1 (days 1-7)", "week 2 (days 8-14)"], ylabel="RPSS, ERE days 0/1/2+", ylim=(-0.03, 0.13))
ax.grid(axis="x", visible=False)
ax.set_title("3. Weekly ERE skill: BSISO is better in both weeks")
ax.legend(frameon=False, loc="upper right")
save(fig, "S3_weekly_ERE_RPSS.png", "MISO minus BSISO: week 1 -0.043 [-0.069, -0.016], week 2 -0.048 [-0.081, -0.016].")

# ---------------------------------------------------------------- S4: when does each index know about the ERE
LL = rd("R8_miso/ceiling/lead_lag_minus14_to_10.csv")
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.axvspan(-14.5, 0, color=GRID, alpha=0.5, lw=0)
ax.text(-14.2, 0.088, "<- index taken AFTER the ERE (no use for forecasting)", fontsize=11.5, color=INK2, va="top")
ax.text(0.4, 0.088, "taken BEFORE (a real warning) ->", fontsize=11.5, color=INK2, va="top")
for col, c, lab in (("MISO", OR, "MISO"), ("BSISO1", BL, "BSISO")):
    ax.plot(LL.lag, LL[col], color=c, label=lab)
    k = LL[col].idxmax()
    ax.plot(LL.lag[k], LL[col][k], "o", color=c, ms=9, mec=SURF, mew=2)
    when = f"{abs(LL.lag[k])} d {'after' if LL.lag[k] < 0 else 'before'}"
    ax.annotate(f"{lab} best: {when}", (LL.lag[k], LL[col][k]), xytext=(14, 6), textcoords="offset points",
                ha="left", fontsize=12)
ax.axvline(0, color=INK2, lw=1)
ax.set(xlabel="days the observed index is taken before the ERE day (negative = after)", ylabel="RPSS (0 = normal chance)",
       xlim=(-14.5, 10.5), ylim=(0, 0.09), xticks=range(-14, 11, 2))
ax.set_title("4. MISO knows about EREs after they happen; BSISO before")
ax.legend(frameon=False, loc="center right")
save(fig, "S4_when_does_the_index_know.png", "Observed indices (no forecast), JJAS 2000-2020. MISO is a 15-day trailing summary of rain.")

# ---------------------------------------------------------------- S5: spell warning
SP = rd("R8_miso/spell_check/spell_lead_lag.csv")
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, sp in zip(axs, ("active", "break")):
    ax.axvspan(-10.5, 0, color=GRID, alpha=0.5, lw=0)
    for ix, c, lab in (("MISO", OR, "MISO"), ("BSISO1", BL, "BSISO")):
        d = SP[(SP.spell == sp) & (SP["index"] == ix)].sort_values("lag (+ = before)")
        ax.plot(d["lag (+ = before)"], d["AUC in-sample"], color=c, label=lab)
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    ax.axvline(0, color=INK2, lw=1)
    ax.set(xlabel="days before the spell day (negative = after)", xlim=(-10.5, 10.5), xticks=range(-10, 11, 5), ylim=(0.45, 0.9))
    ax.set_title(f"{sp} spells")
axs[0].set_ylabel("AUC (0.5 = coin flip)")
axs[0].text(-10, 0.47, "after", fontsize=11, color=INK2); axs[0].text(6.5, 0.47, "before", fontsize=11, color=INK2)
axs[0].legend(frameon=False, loc="upper right")
fig.suptitle("5. Spells: MISO is strongest once the spell has started; ahead of it the gap shrinks", x=0.01, ha="left",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0.03, 1, 0.93))
save(fig, "S5_spell_AUC.png", "Dey spell rule, IMERG central India 2000-2020. In-sample AUC (leave-one-year-out is lower, ~0.5 for MISO 5 d before).")

# ---------------------------------------------------------------- S6: the best BSISO product
WE = rd("R9e_regional_on_best/weekly_scores.csv")
S6 = [("BSISO only", GR, "BSISO only"), ("product: old best", "#86b6ef", "BSISO + today's rain"),
      ("product: v2 (pre-registered)", BL, "best product\n(+ BSISO2 in week 2)")]
fig, ax = plt.subplots(figsize=(10, 5.5))
for i, (m, c, lab) in enumerate(S6):
    v = [WE[(WE.window == w) & (WE.variant == m)]["RPSS"].iloc[0] for w in ("week 1", "week 2")]
    if m.startswith("product: v2"):
        v[0] = WE[(WE.window == "week 1") & (WE.variant == "product: old best")]["RPSS"].iloc[0]   # week 1 rule failed -> old best
    xs = np.arange(2) + (i - 1) * 0.27
    ax.bar(xs, v, width=0.25, color=c, label=lab.replace("\n", " "), edgecolor=SURF, linewidth=2)
    for x, y in zip(xs, v):
        ax.text(x, y + 0.004, f"{y:.3f}", ha="center", fontsize=12.5)
ax.set(xticks=range(2), xticklabels=["week 1 (days 1-7)", "week 2 (days 8-14)"], ylabel="RPSS, ERE days 0/1/2+", ylim=(0, 0.24))
ax.grid(axis="x", visible=False)
ax.set_title("6. Where the BSISO forecast stands now")
ax.legend(frameon=False, loc="upper right")
save(fig, "S6_best_BSISO_product.png", "JJAS 2000-2024, same days. Best product: week 1 = BSISO + today's IMERG rain; week 2 adds BSISO2 (+0.011 [+0.004, +0.018]).")
print("figures saved to", OUT)
