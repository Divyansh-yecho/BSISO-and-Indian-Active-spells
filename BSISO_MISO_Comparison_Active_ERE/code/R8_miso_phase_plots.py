# R8 follow-up (local): the BSISO-Rain phase figures redone for the official MISO index (IITM, Suhas et al. 2013).
# Same spell rule as BSISO-Rain ("project": CMZ standardised IMERG rain >= +1 / <= -1 sd for >= 3 days, JJAS),
# same phase frame, same season re-dating null. Days: JJAS 2000-2020 (the MISO files stop in Oct 2020).
# MISO phase = octant of atan2(-MISO2, MISO1) (matches the phase column of the IITM files on 100% of days);
# drawn with MISO1 on x and MISO2 on y, so phases run clockwise from the +MISO1 axis.
# ERE classes = the final package's central-India catalogue events: 16-39, 40-90, 91+ cells (any = 16+).
# Figures go to sem_project_7/figs/R8_miso/.
import os, re, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = r"C:\Users\divya\OneDrive\Desktop\IIT- Bombay Summer 26\Moaap\Regional_claude"
SP7 = os.path.join(PROJ, "sem_project_7")
FIG = os.path.join(SP7, "figs", "R8_miso")
OUTT = os.path.join(SP7, "other", "intermediate", "R8_miso", "phase_plots")
os.makedirs(FIG, exist_ok=True); os.makedirs(OUTT, exist_ok=True)
os.environ["BSISO_DATA"] = os.path.join(PROJ, "data"); os.environ["FAST"] = "1"
cwd = os.getcwd(); os.chdir(os.environ.get("TEMP", "."))


def cells_of(path):
    src = open(path, encoding="utf-8").read()
    return {blk.split(" --")[0].strip(): blk for blk in re.split(r"^# %% ", src, flags=re.M)[1:] if not blk.startswith("MD")}


# 1) BSISO-Rain cells 1-2: IMERG, BSISO, spells (project rule), plot style
RP = cells_of(os.path.join(SP7, "other", "scripts", "kaggle_BSISO_Rain_plots.py"))
ns = {"__name__": "__main__", "display": lambda *a, **k: None}
exec(compile("# " + RP["CELL 1"], "rp1", "exec"), ns)
exec(compile("# " + RP["CELL 2"], "rp2", "exec"), ns)
times, years, kidx, ND, lab = ns["times"], ns["years"], ns["kidx"], ns["ND"], ns["lab"]
act_runs, brk_runs, MODES = ns["act_runs"], ns["brk_runs"], ns["MODES"]
SURF, INK, INK2, GRID, BLUE, BROWN, NULLC, GREEN = (ns[k] for k in
                                                    ("SURF", "INK", "INK2", "GRID", "BLUE", "BROWN", "NULLC", "GREEN"))
# 2) the final package: ERE size classes on the same IMERG days
PK = cells_of(os.path.join(SP7, "BSISO_daily_ERE_Prediction", "BSISO_daily_ERE_Prediction.py"))
pk = {"__name__": "__main__", "display": lambda *a, **k: None}
exec(compile("# " + PK["CELL 1"], "c1", "exec"), pk)
exec(compile("# " + PK["CELL 2"], "c2", "exec"), pk)
c7 = PK["CELL 7"].split("YC = [")[0].replace(
    "FCI = {nm: pd.DataFrame(FC[nm].reshape(len(ISSUE), -1), index=ISSUE) for nm in MODELS}", "")
exec(compile("# " + c7, "c7", "exec"), pk)
assert (pk["times"] == times).all()
CLS4 = pk["CLS4"]
os.chdir(cwd)

# 3) MISO on the same days
rows = []
for f in sorted(glob.glob(os.path.join(PROJ, "data", "official_indices", "miso_iitm", "*_pc_MJJASO.dat"))):
    a = np.loadtxt(f)
    d = pd.Timestamp(int(os.path.basename(f)[:4]), 5, 1) + pd.to_timedelta(a[:, 0] - 1, "D")
    rows.append(pd.DataFrame({"m1": a[:, 1], "m2": a[:, 2], "ph": a[:, 3]}, index=d))
MI = pd.concat(rows).asfreq("D")
X, Y = MI.m1.reindex(times).values, MI.m2.reindex(times).values
ph = np.floor(np.degrees(np.arctan2(-Y, X)) % 360 / 45) + 1
_f = np.isfinite(X)
assert (ph[_f] == MI.ph.reindex(times).values[_f]).all()
SEL = years <= 2020
NYR = int(SEL.sum() // ND)
MODES["MISO"] = dict(pc1=X, pc2=Y, amp=np.hypot(X, Y), phase=np.nan_to_num(ph).astype(int), PC1=MI.m1, PC2=MI.m2)
XY = {"MISO": lambda th, r: (r * np.cos(np.deg2rad(th)), -r * np.sin(np.deg2rad(th))),   # phase angle -> (x, y)
      "BSISO1": lambda th, r: (r * np.sin(np.deg2rad(th)), -r * np.cos(np.deg2rad(th)))}
AX = {"MISO": ("MISO1", "MISO2"), "BSISO1": ("BSISO1 PC1", "BSISO1 PC2")}
x8 = np.arange(1, 9)


def frame(a, mode, rmax=4.0, label="Ph-", fs=10, circle=True):
    to_xy = XY[mode]
    for k in range(8):
        a.plot(*zip((0, 0), to_xy(k * 45, rmax)), color=GRID, lw=1, zorder=1)
    if circle:
        a.add_patch(plt.Circle((0, 0), 1, fill=False, ec=INK, lw=1.2, zorder=2))
    for k in range(1, 9):
        a.text(*to_xy((k - 0.5) * 45, rmax * 0.8), f"{label}{k}", ha="center", va="center", fontsize=fs, color=INK2)
    a.set(xlim=(-rmax, rmax), ylim=(-rmax, rmax), aspect="equal", xlabel=AX[mode][0], ylabel=AX[mode][1])
    a.grid(False)


# ---------- Fig 1: active and break days in MISO phase space + share by phase
M = MODES["MISO"]
strong = (M["amp"] >= 1) & SEL
fig, ax = plt.subplots(2, 2, figsize=(11, 10.5))
share_all = np.array([(M["phase"][strong] == k).mean() for k in x8])
tab = []
for j, (v, col, nm) in enumerate(((1, BLUE, "active"), (-1, BROWN, "break"))):
    sel = (lab == v) & SEL
    frame(ax[0, j], "MISO")
    ax[0, j].scatter(M["pc1"][sel], M["pc2"][sel], s=16, facecolor="none", edgecolor=col, lw=0.9, zorder=3)
    ax[0, j].set_title(f"({'ab'[j]})  {nm} days ({sel.sum()})", loc="left")
    sh = np.array([(M["phase"][sel & strong] == k).mean() for k in x8])
    tab.append(pd.Series(sh, index=x8, name=f"{nm} days"))
    a = ax[1, j]
    a.bar(x8, sh, color=col, width=0.5, label=f"{nm} days (amp >= 1)")
    a.plot(x8, share_all, "D", color=INK2, ms=6, label="all days (if no link)")
    a.set(xticks=x8, xlabel="phase of MISO", ylabel=f"proportion of {nm} days")
    a.set_title(f"({'cd'[j]})  proportion of {nm} days in each MISO phase", loc="left")
    a.legend(frameon=False, fontsize=9)
tab.append(pd.Series(share_all, index=x8, name="all strong days"))
fig.suptitle("MISO (IITM): active and break days in phase space, JJAS 2000-2020", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig(os.path.join(FIG, "fig1_spell_days_phase_space_miso.png"), dpi=140, bbox_inches="tight")
plt.close()
pd.concat(tab, axis=1).round(3).to_csv(os.path.join(OUTT, "spell_share_by_miso_phase.csv"))

# ---------- Fig 2: mean index path around spell onset (MISO and BSISO1, same spells)
LAGS, KEY = np.arange(-15, 16), (-15, -10, -5, 0, 5, 10, 15)


def mean_path(runs, mode):
    P1, P2 = MODES[mode]["PC1"], MODES[mode]["PC2"]
    on = [times[i] for i, _ in runs if years[i] <= 2020]
    mp = np.array([[P1.get(t + pd.Timedelta(days=int(L)), np.nan) for t in on] for L in LAGS])
    mq = np.array([[P2.get(t + pd.Timedelta(days=int(L)), np.nan) for t in on] for L in LAGS])
    return np.nanmean(mp, 1), np.nanmean(mq, 1), len(on)


fig, ax = plt.subplots(2, 2, figsize=(13, 13))
for r, mode in enumerate(("MISO", "BSISO1")):
    for c, (runs, kind, col) in enumerate(((act_runs, "active", BLUE), (brk_runs, "break", BROWN))):
        x_, y_, n = mean_path(runs, mode)
        rmax = max(0.6, 1.35 * np.nanmax(np.hypot(x_, y_)))
        a = ax[r, c]
        frame(a, mode, rmax=rmax, label="Phase ", fs=10, circle=False)
        a.add_patch(plt.Circle((0, 0), rmax, fill=False, ec=GRID, lw=1))
        pre, post = LAGS <= 0, LAGS >= 0
        a.plot(x_[pre], y_[pre], "--", color=col, lw=2, alpha=0.55, label="before onset")
        a.plot(x_[post], y_[post], "-", color=col, lw=2.6, label="after onset")
        a.plot(x_, y_, "o", color=col, ms=2.5)
        for k in range(1, len(LAGS) - 1, 3):
            a.annotate("", (x_[k + 1], y_[k + 1]), (x_[k], y_[k]),
                       arrowprops=dict(arrowstyle="-|>", color=col, lw=0, mutation_scale=14))
        for L in KEY:
            k = int(np.where(LAGS == L)[0][0])
            a.plot(x_[k], y_[k], "o", ms=11 if L == 0 else 8, color=INK if L == 0 else col,
                   mfc=INK if L == 0 else (SURF if L < 0 else col), mew=1.8, zorder=5)
            a.annotate("onset" if L == 0 else f"{L:+d} d", (x_[k], y_[k]), xytext=(7, 7), textcoords="offset points",
                       fontsize=10, fontweight="bold" if L == 0 else None, zorder=6,
                       bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec=GRID, lw=0.6))
        a.set_title(f"({'abcd'[2 * r + c]})  {kind.upper()} spells: mean {mode} path ({n} spells)", loc="left",
                    fontsize=11)
        a.legend(frameon=False, fontsize=9, loc="lower left")
fig.suptitle("Mean index path from 15 days before to 15 days after spell onset (same spells, JJAS 2000-2020)",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig(os.path.join(FIG, "fig2_index_path_around_spell_onset.png"), dpi=140, bbox_inches="tight")
plt.close()

# ---------- Fig 3: ERE rate by phase for each size class, MISO vs BSISO1 (same days), season re-dating null
CL = {"any ERE (16+ cells)": CLS4 >= 1, "16-39 cells": CLS4 == 1, "40-90 cells": CLS4 == 2, "91+ cells": CLS4 == 3}
rng = np.random.default_rng(12)
PERMS = [rng.permutation(NYR) for _ in range(1000)]
ratio = lambda v, p: np.array([v[p == k].mean() for k in x8]) / v.mean()
rows, RES = [], {}
for mode in ("MISO", "BSISO1"):
    Mm = MODES[mode]
    PH = np.where(Mm["amp"] >= 1, Mm["phase"], 0)[SEL]
    for nm, ev in CL.items():
        v = ev[SEL].astype(float)
        obs = ratio(v, PH)
        nul = np.array([ratio(v, PH.reshape(NYR, ND)[p].ravel()) for p in PERMS])
        st = lambda r: np.abs(r - 1).sum(axis=-1)
        p = (1 + (st(nul) >= st(obs)).sum()) / 1001
        RES[(mode, nm)] = (obs, np.percentile(nul, 5, 0), np.percentile(nul, 95, 0), p, int(v.sum()))
        rows.append({"index": mode, "class": nm, "ERE days": int(v.sum()),
                     **{f"ph{k}": round(obs[k - 1], 2) for k in x8}, "pattern p": round(p, 3)})
pd.DataFrame(rows).to_csv(os.path.join(OUTT, "ere_rate_by_phase_and_class.csv"), index=False)
fig, ax = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
COLS = [INK2, "#9cc3f0", BLUE, "#123f80"]
for r, mode in enumerate(("MISO", "BSISO1")):
    for c, (nm, col) in enumerate(zip(CL, COLS)):
        obs, lo, hi, p, n = RES[(mode, nm)]
        a = ax[r, c]
        a.fill_between(x8, lo, hi, step="mid", color=NULLC, alpha=0.8, label="chance (90%)")
        a.bar(x8, obs, color=col, width=0.6, label="observed")
        a.axhline(1, color=INK, lw=0.8)
        a.set(xticks=x8, ylim=(0, 3.9 if nm == "91+ cells" else 2.6))
        a.set_title(f"{mode}: {nm} ({n} days)\npattern p = {p:.3f}", loc="left", fontsize=10.5)
        if c == 0:
            a.set_ylabel("rate vs average day")
        if r == 1:
            a.set_xlabel(f"phase (amplitude >= 1)")
ax[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Central-India EREs by phase and size class: MISO (top) vs BSISO1 (bottom), index on the ERE day, "
             "JJAS 2000-2020", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(os.path.join(FIG, "fig3_ere_rate_by_phase_and_class.png"), dpi=140, bbox_inches="tight")
plt.close()

# ---------- Fig 4: ERE days by class on the MISO phase plane
fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))
for a, (k, nm, col) in zip(ax, ((1, "16-39 cells", "#9cc3f0"), (2, "40-90 cells", BLUE), (3, "91+ cells", "#123f80"))):
    frame(a, "MISO")
    s = SEL & (CLS4 == 0)
    a.scatter(M["pc1"][s], M["pc2"][s], s=4, color=NULLC, zorder=2, label="no-ERE days")
    s = SEL & (CLS4 == k)
    a.scatter(M["pc1"][s], M["pc2"][s], s=18, facecolor=col, edgecolor=INK, lw=0.3, zorder=3, label=f"ERE {nm}")
    a.set_title(f"{nm}: {s.sum()} ERE days", loc="left")
    a.legend(frameon=False, fontsize=9, loc="lower left")
fig.suptitle("Central-India ERE days on the MISO phase plane (index on the ERE day), JJAS 2000-2020", x=0.01,
             ha="left", fontsize=12.5, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.93))
plt.savefig(os.path.join(FIG, "fig4_ere_days_miso_phase_plane.png"), dpi=140, bbox_inches="tight")
plt.close()

# ---------- Fig 5: when does each index know? (from the R8 follow-up CSVs)
LL = pd.read_csv(os.path.join(SP7, "other", "intermediate", "R8_miso", "ceiling", "lead_lag_minus14_to_10.csv"),
                 index_col=0)
SPL = pd.read_csv(os.path.join(SP7, "other", "intermediate", "R8_miso", "spell_check", "spell_lead_lag.csv"))
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for nm, col in (("MISO", "#c0392b"), ("BSISO1", BLUE)):
    ax[0].plot(LL.index, LL[nm], "o-", color=col, label=nm)
    for a, sp in zip(ax[1:], ("active", "break")):
        g = SPL[(SPL.spell == sp) & (SPL["index"] == nm)]
        a.plot(g["lag (+ = before)"], g["AUC in-sample"], "o-", color=col, label=nm)
ax[0].set(ylabel="daily 4-class RPSS (0 = normal chance)", title="(a) central-India ERE classes")
ax[1].set(ylabel="AUC (0.5 = no skill)", title="(b) active-spell days")
ax[2].set(title="(c) break-spell days")
for a in ax:
    a.axvline(0, color=INK, lw=0.8)
    a.set_xlabel("index taken k days BEFORE the day (negative = after)")
    a.legend(frameon=False)
    a.set_title(a.get_title(), loc="left")
ax[1].axhline(0.5, color=INK2, lw=0.8, ls=":"); ax[2].axhline(0.5, color=INK2, lw=0.8, ls=":")
fig.suptitle("When does each index know? MISO peaks AFTER the event, BSISO1 BEFORE it (observed indices, JJAS "
             "2000-2020; (b, c) in-sample AUC)", x=0.01, ha="left", fontsize=12, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig(os.path.join(FIG, "fig5_when_does_the_index_know.png"), dpi=140, bbox_inches="tight")
plt.close()
print(pd.DataFrame(rows).to_string(index=False))
print(pd.concat(tab, axis=1).round(3).to_string())
print("spells (<= 2020): active", sum(years[i] <= 2020 for i, _ in act_runs), "break",
      sum(years[i] <= 2020 for i, _ in brk_runs))
