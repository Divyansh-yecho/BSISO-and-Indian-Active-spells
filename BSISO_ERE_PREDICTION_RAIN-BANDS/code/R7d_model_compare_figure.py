# Figure for R7d: normal chance vs ridge / lasso / BSISO-only transformer / best (transformer + IMERG rain inputs to 10 d),
# all scored on the same days (tables from R7d_model_compare_same_days.py).
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP7 = r"C:\Users\divya\OneDrive\Desktop\IIT- Bombay Summer 26\Moaap\Regional_claude\sem_project_7"
T = os.path.join(SP7, "other", "intermediate", "R7_rain_band", "model_compare")
D = pd.read_csv(os.path.join(T, "daily_scores_same_days.csv"))
W = pd.read_csv(os.path.join(T, "weekly_scores_same_days.csv"))
# best model = rain inputs to 10 d, BSISO only after (the R7c product rule)
best = pd.concat([D[(D.model == "transformer|rain") & (D.lead <= 10)], D[(D.model == "transformer|bsiso") & (D.lead > 10)]])
LINES = [("normal chance", D[D.model == "normal chance"], "#a3a29d", "--"), ("ridge", D[D.model == "ridge|bsiso"], "#4c78a8", "-"),
         ("lasso", D[D.model == "lasso|bsiso"], "#72b7b2", ":"), ("transformer, BSISO only", D[D.model == "transformer|bsiso"], "#f58518", "-"),
         ("best: transformer + rain band", best, "#c0392b", "-")]
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for i, (c, t) in enumerate((("RPSS", "(a) daily RPSS (0 = normal chance)"), ("corr", "(b) daily correlation of P(ERE) with ERE days"))):
    for nm, s, col, ls in LINES:
        ax[i].plot(s.lead, s[c], color=col, ls=ls, lw=2.2 if nm.startswith("best") else 1.8, marker="o", ms=3.5, label=nm)
    ax[i].set(xlabel="lead (days)", xticks=range(1, 15), title=t); ax[i].grid(alpha=.3)
ax[0].legend(fontsize=9)
wk = {"normal chance": "normal chance", "ridge": "ridge|bsiso", "lasso": "lasso|bsiso", "transformer, BSISO only": "transformer|bsiso",
      "best: transformer + rain band": "transformer|rain"}
x = np.arange(2); w = 0.16
for j, (nm, key) in enumerate(wk.items()):
    v = [W[(W.window == k) & (W.model == key)]["RPSS"].iloc[0] for k in ("week 1", "week 2")]
    b = ax[2].bar(x + (j - 2) * w, v, w, color=LINES[j][2], label=nm)
    ax[2].bar_label(b, fmt="%.3f", fontsize=7.5, padding=1)
ax[2].set(xticks=x, xticklabels=["week 1 (days 1-7)", "week 2 (days 8-14)"], ylabel="RPSS (0 / 1 / 2+ ERE days)",
          title="(c) weekly RPSS"); ax[2].grid(alpha=.3, axis="y")
fig.suptitle("Same days for every model (JJAS 2000-2024, leave-one-year-out)", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(SP7, "figs", "R7_rain_band", "R7d_model_compare.png"), dpi=150, bbox_inches="tight")
