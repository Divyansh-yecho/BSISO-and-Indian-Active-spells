# R7a: the monsoon rain band in the RAIN STEP (local, CPU). BSISO input = the final transformer forecast of T-5
# (BSISO_daily_ERE_Prediction, 20 seeds; observed BSISO for leads <= 5). Extra inputs, all from the ISSUE day (T - lead):
#   IMERG band  = rain averaged over 70-95E in 1-degree latitude bins 5-30N, anomaly from the date's normal, 3 patterns
#   band 3 d ago = the same 3 patterns three days before the issue day (which way and how fast the band moves)
#   CI rain     = log(1 + mean central-India rain) and "large ERE today" (0/1)
#   OLR band    = NOAA OLR anomaly over 70-95E in 2.5-degree bins 10S-30N (available all year), 3 patterns
# Strictly out of sample: for each held-out summer the band normals, scaling and patterns are fitted on the other 24.
# All variants are scored on the same days (issue day and 3 days before it inside JJAS). Daily leads 1-14 + weekly.
import os, re, matplotlib
matplotlib.use("Agg")
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

PROJ = r"C:\Users\divya\OneDrive\Desktop\IIT- Bombay Summer 26\Moaap\Regional_claude"
SP7 = os.path.join(PROJ, "sem_project_7")
TAG = os.environ.get("R7_TAG", "R7a")                             # R7c = same test on the R7b BSISO forecast
OUTT = os.path.join(SP7, "other", "intermediate", "R7_rain_band", TAG)
OUTF = os.path.join(SP7, "figs", "R7_rain_band")
os.makedirs(OUTT, exist_ok=True); os.makedirs(OUTF, exist_ok=True)
os.environ["BSISO_DATA"] = os.path.join(PROJ, "data"); os.environ["FAST"] = "1"
src = open(os.path.join(SP7, "BSISO_daily_ERE_Prediction", "BSISO_daily_ERE_Prediction.py"), encoding="utf-8").read()
cells = {blk.split(" --")[0].strip(): blk for blk in re.split(r"^# %% ", src, flags=re.M)[1:] if not blk.startswith("MD")}
ns = {"__name__": "__main__", "display": lambda *a: None}
os.chdir(os.environ.get("TEMP", "."))
for c in ("CELL 1", "CELL 2", "CELL 3"):
    exec(compile("# " + cells[c], c, "exec"), ns)
ISSUE, H, times, years, find = ns["ISSUE"], ns["H"], ns["times"], ns["years"], ns["find"]
F = pd.read_parquet(os.environ.get("R7_FC", os.path.join(SP7, "BSISO_daily_ERE_Prediction", "results", "bsiso_forecasts.parquet")))
F = F[F.model == "transformer"] if "model" in F else F             # test-round parquets hold one model each
P = np.full((len(ISSUE), H, 2), np.nan, dtype="float32")
for L in range(1, H + 1):
    P[:, L - 1] = F[F.lead == L].set_index("issue")[["f1", "f2"]].reindex(ISSUE).values
ns.update(FC={"transformer": P}, MODELS=["transformer"])
exec(compile("# " + cells["CELL 7"].split("YC = [")[0], "cell7a", "exec"), ns)
CLS4, OBS1, kidx, ND, NYR = ns["CLS4"], ns["OBS1"], ns["kidx"], ns["ND"], ns["NYR"]
fit_predict, clim_logit, bsiso_T5 = ns["fit_predict"], ns["clim_logit"], ns["bsiso_T5"]
YC = [(CLS4 >= t).astype(float) for t in (1, 2, 3)]
UY = np.unique(years)

# ---------------------------------------------------------------- raw daily inputs (target-day calendar = JJAS days)
ds = xr.open_dataset(find("imerg_india_025_jjas_2000_2024.nc"))
lat, lon = ds.lat.values, ds.lon.values
BAND = ds.tp.sel(lon=slice(70, 95)).mean("lon").sel(lat=slice(30, 5)).coarsen(lat=4, boundary="trim").mean().values
tp = ds.tp.values
ds.close()
LA2, LO2 = np.meshgrid(lat, lon, indexing="ij")
mask = ns["mask"]
ci = mask & (LA2 >= 17) & (LA2 < 26.5) & (LO2 >= 74) & (LO2 <= 88.5) & ~((LO2 <= 77.5) & (LA2 <= 21))
CIR = np.log1p(tp[:, ci].mean(1))
del tp
o = xr.open_dataset(find("olr_cbo_daily_1991_2026_30S45N_30E130E.nc"))["olr"].sel(lon=slice(70, 95)).mean("lon").sel(lat=slice(30, -10))
OLR = pd.DataFrame(o.values, index=pd.DatetimeIndex(o.time.values).normalize())
odoy = OLR.index.dayofyear
ocl = OLR[(OLR.index.year >= 1991) & (OLR.index.year <= 2020)].groupby(odoy[(OLR.index.year >= 1991) & (OLR.index.year <= 2020)]).mean()
ocl = pd.concat([ocl, ocl, ocl]).rolling(31, center=True).mean().iloc[len(ocl):2 * len(ocl)]
OLRA = OLR.values - ocl.loc[odoy].values                           # anomaly from the fixed 1991-2020 normal
OLRA = pd.DataFrame(OLRA, index=OLR.index)
print(f"IMERG band {BAND.shape[1]} bins; OLR band {OLRA.shape[1]} bins, {OLRA.index[0]:%Y}-{OLRA.index[-1]:%Y}")


def shift_idx(L):
    """index (in times) of the day L days before each target day, -1 if it is outside that summer."""
    j = np.arange(len(times)) - L
    ok = (j >= 0) & (j < len(times))
    j = np.where(ok, j, 0)
    ok &= (times[j] == times - pd.Timedelta(days=L))
    return np.where(ok, j, -1)


def band_patterns(yr):
    """IMERG band: normal, scaling and 3 patterns from the summers other than yr; returns (len(times), 3)."""
    tr = years != yr
    cl = pd.DataFrame(BAND[tr]).groupby(kidx[tr]).mean().reindex(range(ND)).interpolate(limit_direction="both")
    cl = cl.rolling(15, center=True, min_periods=1).mean().values
    an = BAND - cl[kidx]
    mu, sd = an[tr].mean(0), an[tr].std(0)
    an = (an - mu) / sd
    Vt = np.linalg.svd(an[tr], full_matrices=False)[2]
    return an @ Vt[:3].T


def olr_patterns(yr, issue_dates):
    """OLR band on the issue dates: scaling and 3 patterns from JJAS days of the summers other than yr."""
    A = OLRA.reindex(issue_dates).values
    ref = OLRA[(OLRA.index.month.isin([6, 7, 8, 9])) & (OLRA.index.year >= 2000) & (OLRA.index.year <= 2024) &
               (OLRA.index.year != yr)].values
    mu, sd = np.nanmean(ref, 0), np.nanstd(ref, 0)
    Vt = np.linalg.svd(np.nan_to_num((ref - mu) / sd), full_matrices=False)[2]
    return ((A - mu) / sd) @ Vt[:3].T


def take(arr, idx):
    out = np.full((len(idx),) + arr.shape[1:], np.nan)
    out[idx >= 0] = arr[idx[idx >= 0]]
    return out


VARIANTS = ["BSISO only (final model)", "+ IMERG band today", "+ band today & 3 days ago", "+ CI rain & ERE today",
            "+ OLR band today", "+ all IMERG inputs", "+ all IMERG + OLR band"]


def features(L, yr, Bp):
    """feature matrix for every variant, fold yr (Bp = IMERG band patterns for that fold)."""
    i0, i3 = shift_idx(L), shift_idx(L + 3)
    base = bsiso_T5("transformer", L)
    b0, b3 = take(Bp, i0), take(Bp, i3)
    cr = np.column_stack([take(CIR, i0), take(OBS1, i0)])
    ob = olr_patterns(yr, times - pd.Timedelta(days=L))
    allf = np.column_stack([base, b0, b3, cr, ob])
    ok = np.isfinite(allf).all(1)                                  # the same days for every variant
    V = {VARIANTS[0]: base, VARIANTS[1]: np.column_stack([base, b0]), VARIANTS[2]: np.column_stack([base, b0, b3]),
         VARIANTS[3]: np.column_stack([base, cr]), VARIANTS[4]: np.column_stack([base, ob]),
         VARIANTS[5]: np.column_stack([base, b0, b3, cr]), VARIANTS[6]: allf}
    return {k: np.where(ok[:, None], v, np.nan) for k, v in V.items()}


BP = {yr: band_patterns(yr) for yr in UY}
RES = {}
for L in range(1, 15):
    PM = {v: np.full((3, len(times)), np.nan) for v in VARIANTS}
    PR = np.full((3, len(times)), np.nan)
    G = np.zeros(len(times), bool)
    for yr in UY:
        Fv = features(L, yr, BP[yr])
        te = years == yr
        good = np.isfinite(Fv[VARIANTS[0]]).all(1)
        G |= te & good
        for c, y in enumerate(YC):
            cl = clim_logit(y, years, kidx, yr, ND)
            tr = (years != yr) & good
            PR[c, te & good] = fit_predict(cl[tr], y[tr], cl[te & good])
            for v in VARIANTS:
                Xv = np.column_stack([cl, Fv[v]])
                PM[v][c, te & good] = fit_predict(Xv[tr], y[tr], Xv[te & good])
    PRm = np.minimum.accumulate(PR, 0)
    Yy = np.array(YC)
    rr = np.where(G, ((np.nan_to_num(PRm) - Yy) ** 2).sum(0), 0)
    for v in VARIANTS:
        Pm = np.minimum.accumulate(PM[v], 0)
        rm = np.where(G, ((np.nan_to_num(Pm) - Yy) ** 2).sum(0), 0)
        RES[(v, L)] = {"rps_m": np.array([rm[years == y].sum() for y in UY]), "rps_r": np.array([rr[years == y].sum() for y in UY]),
                       "corr": np.corrcoef(Pm[0][G], OBS1[G])[0, 1], "auc": roc_auc_score(OBS1[G], Pm[0][G]),
                       "P": Pm, "G": G}
    print(f"lead {L}: {G.sum()} days; RPSS " + ", ".join(f"{1 - RES[(v, L)]['rps_m'].sum() / RES[(v, L)]['rps_r'].sum():.3f}" for v in VARIANTS), flush=True)

rpss = lambda r: 1 - r["rps_m"].sum() / r["rps_r"].sum()
BI = np.random.default_rng(0).integers(0, NYR, (2000, NYR))
sk = lambda r: 1 - r["rps_m"][BI].sum(1) / r["rps_r"][BI].sum(1)
D = pd.DataFrame([{"variant": v, "lead": L, "RPSS": rpss(RES[(v, L)]), "corr": RES[(v, L)]["corr"], "AUC": RES[(v, L)]["auc"],
                   "days": int(RES[(v, L)]["G"].sum())} for v in VARIANTS for L in range(1, 15)])
rows = []
for v in VARIANTS[1:]:
    for L in range(1, 15):
        d = sk(RES[(v, L)]) - sk(RES[(VARIANTS[0], L)])
        rows.append({"variant": v, "lead": L, "RPSS gain": d.mean(), "90% low": np.percentile(d, 5),
                     "90% high": np.percentile(d, 95), "share of resamples better": (d > 0).mean()})
CM = pd.DataFrame(rows)

# ---------------------------------------------------------------- weekly: expected ERE days -> P(0/1/2+)
J = np.arange(len(times) - 6); J = J[(times[J + 6] - times[J]).days == 6]
NW = np.array([OBS1[j:j + 7].sum() for j in J]); yW, kW = years[J], kidx[J]
YCW = [(NW >= 1).astype(float), (NW >= 2).astype(float)]
ordinal_eval = ns["ordinal_eval"]


def weekly(get_p, a, b_):
    S = np.sum([get_p(L)[J + (L - a)] for L in range(a, b_ + 1)], axis=0)
    r = ordinal_eval(YCW, np.log(S + 1e-3)[:, None], yW, kW, ND, (NW >= 1).astype(float))
    g = r["G"]
    r["corr_count"] = np.corrcoef(S[g], NW[g])[0, 1]
    return r


dp = lambda v: (lambda L: np.where(RES[(v, L)]["G"], RES[(v, L)]["P"][0], np.nan))
# a-priori rule from the first quick test: rain inputs only help up to ~10 days, so the "combined" product uses the
# best rain-informed variant for leads 1-10 and the BSISO-only model for 11-14
COMBO = VARIANTS[-1]
combo = lambda L: dp(COMBO if L <= 10 else VARIANTS[0])(L)
# chosen after seeing the daily table: all IMERG inputs (no OLR band) to 10 d - the best daily variant at 1-10 d
combo2 = lambda L: dp(VARIANTS[5] if L <= 10 else VARIANTS[0])(L)
wrows, WE = [], {}
for wk, (a, b_) in {"week 1 (days 1-7)": (1, 7), "week 2 (days 8-14)": (8, 14)}.items():
    for v in VARIANTS + ["combined (rain inputs to 10 d)", "combined, all IMERG to 10 d"]:
        g = combo2 if v == "combined, all IMERG to 10 d" else (combo if v.startswith("combined") else dp(v))
        r = weekly(g, a, b_)
        WE[(wk, v)] = r
        wrows.append({"window": wk, "variant": v, "RPSS (0/1/2+)": rpss(r), "corr count": r["corr_count"], "AUC any": r["auc"]})
    for v in VARIANTS[1:] + ["combined (rain inputs to 10 d)", "combined, all IMERG to 10 d"]:
        d = sk(WE[(wk, v)]) - sk(WE[(wk, VARIANTS[0])])
        wrows.append({"window": wk, "variant": f"gain: {v}", "RPSS (0/1/2+)": d.mean(), "corr count": np.percentile(d, 5),
                      "AUC any": np.percentile(d, 95)})
W = pd.DataFrame(wrows)
CN = ["no ERE", "16-39 cells", "40-90 cells", "91+ cells"]
rows = []                                                          # the product: daily 4-class chances, leads 1-14
for L in range(1, 15):
    r = RES[(VARIANTS[5] if L <= 10 else VARIANTS[0], L)]
    C = r["P"]; Pc = np.array([1 - C[0], C[0] - C[1], C[1] - C[2], C[2]])
    rows.append(pd.DataFrame({"issue": times - pd.Timedelta(days=L), "target": times, "lead": L,
                              "inputs": "BSISO + IMERG rain" if L <= 10 else "BSISO only",
                              **{f"P({CN[c]})": Pc[c] for c in range(4)}, "observed": np.array(CN)[CLS4]})[r["G"]])
pd.concat(rows).sort_values(["issue", "lead"]).to_csv(os.path.join(OUTT, "daily_ere_class_probabilities_with_rain.csv"), index=False)
wk_rows = []
for wk, a in (("week 1 (days 1-7)", 1), ("week 2 (days 8-14)", 8)):
    r = WE[(wk, "combined, all IMERG to 10 d")]
    wk_rows.append(pd.DataFrame({"window": wk, "issue": times[J] - pd.Timedelta(days=a), "first day": times[J],
                                 "P(0 ERE days)": 1 - r["P"][0], "P(1)": r["P"][0] - r["P"][1], "P(2+)": r["P"][1],
                                 "observed ERE days": NW})[r["G"]])
pd.concat(wk_rows).to_csv(os.path.join(OUTT, "weekly_ere_count_probabilities_with_rain.csv"), index=False)
D.to_csv(os.path.join(OUTT, "daily_scores.csv"), index=False)
CM.to_csv(os.path.join(OUTT, "daily_gain_vs_bsiso_only.csv"), index=False)
W.to_csv(os.path.join(OUTT, "weekly_scores.csv"), index=False)   # 'gain:' rows = mean gain, 90% low, 90% high
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(D.pivot(index="variant", columns="lead", values="RPSS").loc[VARIANTS].round(3).to_string())
print(D.pivot(index="variant", columns="lead", values="corr").loc[VARIANTS].round(3).to_string())
print(CM[CM.lead.isin([1, 2, 3, 5, 7, 8, 10, 12, 14])].round(4).to_string(index=False))
print(W.round(4).to_string(index=False))

# ---------------------------------------------------------------- figure
COL = {VARIANTS[1]: "#2a78d6", VARIANTS[2]: "#6250d6", VARIANTS[3]: "#eb6834", VARIANTS[4]: "#1baf7a",
       VARIANTS[6]: "#0b0b0b"}
SURF, INK2, GRID = "#fcfcfb", "#52514e", "#e6e5e1"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
                     "axes.titlesize": 10.5, "text.color": "#0b0b0b", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for v, c in COL.items():
    s = D[D.variant == v]; g = CM[CM.variant == v]
    ax[0].plot(s.lead, s["corr"], color=c, marker="o", ms=4, lw=2, label=v)
    ax[1].plot(g.lead, g["RPSS gain"], color=c, marker="o", ms=4, lw=2, label=v)
s0 = D[D.variant == VARIANTS[0]]
ax[0].plot(s0.lead, s0["corr"], color="#a3a29d", ls="--", lw=2, label=VARIANTS[0])
g6 = CM[CM.variant == VARIANTS[6]]
ax[1].fill_between(g6.lead, g6["90% low"], g6["90% high"], color="#0b0b0b", alpha=0.08, lw=0)
ax[1].axhline(0, color=INK2, lw=1); ax[1].set_yscale("symlog", linthresh=0.01)
ax[0].set(xlabel="rain lead (days)", ylabel="correlation of P(ERE) with ERE days", xticks=range(1, 15),
          title="(a) daily correlation, same days for every variant")
ax[1].set(xlabel="rain lead (days)", ylabel="RPSS gain over BSISO only (symlog)", xticks=range(1, 15),
          title="(b) daily skill gain (shading: 90% range, all inputs)")
ax[0].legend(frameon=False, fontsize=8)
wk = W[~W.variant.str.startswith("gain")]
labs = [VARIANTS[0], VARIANTS[1], VARIANTS[3], VARIANTS[6], "combined (rain inputs to 10 d)"]
cols = ["#a3a29d", "#2a78d6", "#eb6834", "#0b0b0b", "#1baf7a"]
for i, (v, c) in enumerate(zip(labs, cols)):
    vals = [wk[(wk.window == w) & (wk.variant == v)]["corr count"].iloc[0] for w in ("week 1 (days 1-7)", "week 2 (days 8-14)")]
    ax[2].bar(np.arange(2) + (i - 2) * 0.16, vals, width=0.15, color=c, label=v, edgecolor=SURF)
ax[2].set(xticks=[0, 1], xticklabels=["week 1 (days 1-7)", "week 2 (days 8-14)"], ylim=(0, 0.65),
          ylabel="corr, expected vs observed ERE days", title="(c) weekly: expected vs observed ERE days")
ax[2].legend(frameon=False, fontsize=7.5, loc="upper right"); ax[2].grid(axis="x", visible=False)
fig.suptitle(f"{TAG}: adding the monsoon rain band (issue day) to the rain step", x=0.01, ha="left", fontsize=12.5,
             fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUTF, f"{TAG}_rainband_rain_step.png"), dpi=150, bbox_inches="tight")
print("saved figure")
