# BSISO_Forecast_Tests R8: MISO in place of BSISO. The same patch transformer forecasts either the BSISO1 (APCC, Lee et
# al. 2013) or the official MISO index (IITM, Suhas et al. 2013: EEOF of IMD rain averaged over India, 15-day lags), and
# the same rain step turns each forecast into daily P(no ERE / 16-39 / 40-90 / 91+ cells) in central India.
# Fair comparison: the official MISO exists only for May-October 1998-2020, so EVERY variant here uses the same issue
# days (30-day history and the next 20 days inside May-October 1998-2020), the same 5 year blocks, seeds and loss.
# Scored on JJAS target days 2000-2020, the same days for every model. Inputs (Kaggle datasets):
# BSISO.INDEX.NORM.LY.data, *_pc_MJJASO.dat (MISO), imerg_india_025_jjas_2000_2024.nc, india_mask_025.npy.

# %% CELL 1 -- settings: which variants this round tests
ROUND = "R8"
# each variant: model type and its settings; "win" = days of history used; "amp_w" = weight of the amplitude term;
# "target" = the index the network forecasts ("bsiso" = BSISO1 PC1/PC2, "miso" = MISO1/MISO2); "inputs" = channels
# the network sees, the target's own two channels first (0,1 BSISO1; 2,3 BSISO2; 4,5 season; 6,7 MISO1, MISO2)
# R8: 30-day history (60 days would lose June: MISO starts on 1 May); 20-day output so the MISO rain step (lag 0)
# reaches lead 14 and the 0.3 correlation line can be seen.
BASE = dict(kind="patch", d=64, heads=4, layers=3, ff=128, win=30, amp_w=0.5, lin=True)
VARIANTS = {
    "BSISO T":                       {**BASE, "target": "bsiso", "inputs": [0, 1, 2, 3, 4, 5]},
    "BSISO1-only T":                 {**BASE, "target": "bsiso", "inputs": [0, 1, 4, 5]},
    "MISO T":                        {**BASE, "target": "miso", "inputs": [6, 7, 4, 5]},
    "MISO T + BSISO input":          {**BASE, "target": "miso", "inputs": [6, 7, 0, 1, 2, 3, 4, 5]},
    "BSISO T + MISO input":          {**BASE, "target": "bsiso", "inputs": [0, 1, 2, 3, 4, 5, 6, 7]},
}
LAG = {"bsiso": 5, "miso": 0}      # rain step uses the index this many days before the target day (pre-registered from
                                   # the observed-index test R8_local_index_ceiling.py: BSISO 5 as before, MISO 0)
N_SEEDS = 10                       # a variant may override: seeds=..., batch=..., maxep=..., lr=...

# %% CELL 2 -- setup and data
import os, glob, time, math, warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from IPython.display import display

warnings.filterwarnings("ignore")
ROOTS = [r for r in os.environ.get("BSISO_DATA", "/kaggle/input").split(os.pathsep) if r]
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs"
os.makedirs(OUT, exist_ok=True)
FAST = os.environ.get("FAST") == "1"                              # quick local check: 1 seed, 2 epochs
if FAST:
    N_SEEDS = 1
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find(pattern, required=True, every=False):
    for r in ROOTS:
        hits = sorted(glob.glob(os.path.join(r, "**", pattern), recursive=True))
        if hits:
            return hits if every else hits[0]
    if required:
        raise FileNotFoundError(f"{pattern} not found in the attached datasets")
    return None


SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.dpi": 100, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.axisbelow": True})

b = pd.read_csv(find("BSISO.INDEX.NORM.LY.data"), sep=r"\s+", skiprows=1, header=None,
                names=["year", "doy", "p1", "p2", "q1", "q2", "a1", "a2"]).replace(-999.9, np.nan)
b["date"] = pd.to_datetime(b["year"] * 1000 + b["doy"], format="%Y%j")
b = b.set_index("date").asfreq("D")
b[["p1", "p2", "q1", "q2"]] = b[["p1", "p2", "q1", "q2"]].interpolate(limit=2, limit_area="inside")
DATES = b.index
doy = DATES.dayofyear.values
mrows = []
for f in find("*_pc_MJJASO.dat", every=True):                     # official MISO: day 1 = 1 May, days 1-184
    a = np.loadtxt(f)
    mrows.append(pd.DataFrame({"m1": a[:, 1], "m2": a[:, 2]},
                              index=pd.Timestamp(int(os.path.basename(f)[:4]), 5, 1) + pd.to_timedelta(a[:, 0] - 1, "D")))
MI = pd.concat(mrows).sort_index().reindex(DATES)
print(f"MISO: {MI.m1.notna().sum():,} days, {MI.first_valid_index():%Y-%m-%d} to {MI.last_valid_index():%Y-%m-%d}")
FEAT = np.column_stack([b[["p1", "p2", "q1", "q2"]].values, np.sin(2 * np.pi * doy / 365.25),
                        np.cos(2 * np.pi * doy / 365.25), MI[["m1", "m2"]].values]).astype("float32")
N_FEAT = FEAT.shape[1]

ds = xr.open_dataset(find("imerg_india_025_jjas_2000_2024.nc"))
tp = ds["tp"].values.astype("float32")
lat, lon = ds["lat"].values, ds["lon"].values
times = pd.DatetimeIndex(ds["time"].values).normalize()
ds.close()
mask = np.load(find("india_mask_025.npy"))
years = times.year.values
NYR, ND = len(np.unique(years)), len(set(times.strftime("%m%d")))
kidx = np.tile(np.arange(ND), NYR)
THR = np.maximum(np.nanpercentile(np.where(tp > 1, tp, np.nan), 99, axis=0), 50).astype("float32")
LA2, LO2 = np.meshgrid(lat, lon, indexing="ij")
MXS = np.zeros(len(times))
for t in range(len(times)):
    L, n = ndimage.label(mask & (tp[t] > THR), structure=np.ones((3, 3)))
    if n:
        idx = np.arange(1, n + 1)
        sz = ndimage.sum(np.ones_like(THR), L, idx)
        la, lo = np.array(ndimage.mean(LA2, L, idx)), np.array(ndimage.mean(LO2, L, idx))
        ci = (la >= 17) & (la < 26.5) & (lo >= 74) & (lo <= 88.5) & ~((lo <= 77.5) & (la <= 21))
        if (ci & (sz >= 16)).any():
            MXS[t] = sz[ci & (sz >= 16)].max()
del tp
CLS4 = np.digitize(MXS, [16, 40, 91])                             # 0 none, 1 16-39, 2 40-90, 3 91+ cells
print(f"round {ROUND} | device {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV.type == "cuda" else "") +
      f" | variants: {', '.join(VARIANTS)} | seeds: {N_SEEDS}")

# %% CELL 3 -- samples (the same issue days for every variant) and year blocks
WMAX, H = max(c.get("win", 30) for c in VARIANTS.values()), 20
tt = np.arange(WMAX - 1, len(DATES) - H)
tt = tt[np.array([np.isfinite(FEAT[t - WMAX + 1:t + H + 1]).all() for t in tt])]   # needs MISO: May-Oct 1998-2020
X = np.stack([FEAT[t - WMAX + 1:t + 1] for t in tt])
YB = np.stack([FEAT[t + 1:t + H + 1, 0:2] for t in tt])            # BSISO1 next 1-20 days
YM = np.stack([FEAT[t + 1:t + H + 1, 6:8] for t in tt])            # MISO next 1-20 days
YOF = {"bsiso": YB, "miso": YM}
ISSUE = DATES[tt]
uyr = np.unique(ISSUE.year)
BLK = np.array([{y: i * 5 // len(uyr) for i, y in enumerate(uyr)}[y] for y in ISSUE.year])
Xt = torch.tensor(X).to(DEV)                                      # whole data set lives on the GPU
YT = {k: torch.tensor(v).to(DEV) for k, v in YOF.items()}
print(f"{len(tt):,} samples, issue days {ISSUE.min():%Y-%m-%d} to {ISSUE.max():%Y-%m-%d}; blocks: " +
      ", ".join(f"{ISSUE.year[BLK == k].min()}-{ISSUE.year[BLK == k].max()}" for k in range(5)))

# %% CELL 4 -- the model
class PatchModel(nn.Module):
    """Encoder-only patch transformer (after PatchTST, Nie et al. 2023); all inputs share a token; direct 1-H d head;
    optional linear path from the target's own history (after DLinear, Zeng et al. 2023)."""
    def __init__(self, win, d, heads, layers, ff, lin=True, patch=5, drop=0.1, inputs=(0, 1, 2, 3, 4, 5), **_):
        super().__init__()
        self.win, self.np_, self.inp = win, win // patch, list(inputs)
        self.no = 2
        self.embed = nn.Linear(patch * len(self.inp), d)
        self.pos = nn.Parameter(torch.zeros(1, self.np_, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True,
                                                                    norm_first=True), layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(self.np_ * d, H * self.no)
        self.lin = nn.Linear(win * self.no, H * self.no) if lin else None
        if lin:
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, y=None):
        x = x[:, -self.win:]
        B = x.shape[0]
        z = self.norm(self.enc(self.embed(x[:, :, self.inp].reshape(B, self.np_, -1)) + self.pos)).reshape(B, -1)
        out = self.head(z)
        if self.lin is not None:
            out = out + self.lin(x[:, :, self.inp[:self.no]].reshape(B, -1))
        return out.reshape(B, H, self.no)

    predict = forward


ZOO = {"patch": PatchModel}
for nm, cfg in VARIANTS.items():
    m = ZOO[cfg["kind"]](**cfg)
    print(f"{nm:22s} {sum(p.numel() for p in m.parameters()):>10,} parameters, forecasts {cfg['target']}")

# %% CELL 5 -- training: every variant x 5 year blocks x seeds; forecasts saved after each variant
MAXEP, PATIENCE, BATCH = (2, 2, 256) if FAST else (200, 15, 256)


def loss_fn(p, y, amp_w):
    """(1 - amp_w) x MSE on the two components + amp_w x MSE on amplitude."""
    amp = lambda v: torch.sqrt((v ** 2).sum(-1) + 1e-6)
    return (1 - amp_w) * ((p - y) ** 2).mean() + amp_w * ((amp(p) - amp(y)) ** 2).mean()


def train_one(cfg, tr, va, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Yt = YT[cfg["target"]]
    maxep, batch = (MAXEP if FAST else cfg.get("maxep", MAXEP)), cfg.get("batch", BATCH)
    m = ZOO[cfg["kind"]](**cfg).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    trt, vat = torch.tensor(tr, device=DEV), torch.tensor(va, device=DEV)
    best, bad, state = np.inf, 0, None
    for ep in range(maxep):
        m.train()
        perm = trt[torch.randperm(len(trt), device=DEV)]
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(m(Xt[j]), Yt[j], cfg["amp_w"])
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            v = loss_fn(m(Xt[vat]), Yt[vat], cfg["amp_w"]).item()
        sched.step(v)
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: w.detach().clone() for k, w in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(state)
    m.eval()
    return m, ep + 1


FC, TGT_OF = {}, {}
for nm, (tg, ch) in {"persistence BSISO": ("bsiso", [0, 1]), "persistence MISO": ("miso", [6, 7])}.items():
    FC[nm], TGT_OF[nm] = np.repeat(X[:, -1:, ch], H, axis=1), tg
RIDGES = [("ridge BSISO, all inputs", "bsiso", [0, 1, 2, 3, 4, 5]), ("ridge MISO + season", "miso", [6, 7, 4, 5]),
          ("ridge MISO + BSISO", "miso", list(range(N_FEAT)))]
for nm, tg, ch in RIDGES:
    Xf, Yf, P = X[:, :, ch].reshape(len(X), -1), YOF[tg], np.zeros_like(YOF[tg])
    for k in range(5):
        te, tr = BLK == k, BLK != k
        P[te] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xf[tr], Yf[tr].reshape(tr.sum(), -1)).predict(Xf[te]).reshape(-1, H, 2)
    FC[nm], TGT_OF[nm] = P, tg
BASELINES = list(FC)
rng = np.random.default_rng(0)
VA_YEARS = {k: rng.choice(np.unique(ISSUE.year[BLK != k]), max(2, len(np.unique(ISSUE.year[BLK != k])) // 7),
                          replace=False) for k in range(5)}      # ~15% of training years for early stopping
# Checkpoints: every trained model's test-block forecast is saved at once to ckpt/ (with its epoch count). A run that is
# cut off keeps them in its output; attaching that output to a new run (or re-running locally) skips finished models.
CKPT = os.path.join(OUT, "ckpt")
os.makedirs(CKPT, exist_ok=True)
CK_FOUND = {os.path.basename(f): f for r in ROOTS for f in glob.glob(os.path.join(r, "**", "ckpt", "*.npz"), recursive=True)}
CK_FOUND.update({os.path.basename(f): f for f in glob.glob(os.path.join(CKPT, "*.npz"))})
print(f"checkpoints found from earlier runs: {len(CK_FOUND)}")
LOG = []
for nm, cfg in VARIANTS.items():
    P = np.zeros_like(YOF[cfg["target"]])
    t0 = time.time()
    for k in range(5):
        te = np.where(BLK == k)[0]
        va = np.where((BLK != k) & np.isin(ISSUE.year, VA_YEARS[k]))[0]
        tr = np.where((BLK != k) & ~np.isin(ISSUE.year, VA_YEARS[k]))[0]
        preds = []
        for s in range(1 if FAST else cfg.get("seeds", N_SEEDS)):
            key = f"{ROUND}_{cfg.get('ck', nm).replace(' ', '_')}_b{k}_s{s}{'_fast' if FAST else ''}.npz"
            if key in CK_FOUND:                                   # finished before: reuse, don't retrain
                z = np.load(CK_FOUND[key])
                pred, ne = z["pred"], int(z["epochs"])
                if not os.path.exists(os.path.join(CKPT, key)):
                    np.savez(os.path.join(CKPT, key), pred=pred, epochs=ne)
            else:
                m, ne = train_one(cfg, tr, va, s)
                with torch.no_grad():
                    pred = np.concatenate([m.predict(Xt[te[i:i + 2048]]).cpu().numpy()
                                           for i in range(0, len(te), 2048)])
                np.savez(os.path.join(CKPT, key), pred=pred, epochs=ne)
            preds.append(pred)
            LOG.append({"variant": nm, "block": k, "seed": s, "epochs": ne})
        P[te] = np.mean(preds, axis=0)
        print(f"  {nm}: block {k + 1}/5 done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    FC[nm], TGT_OF[nm] = P, cfg["target"]
    sec = time.time() - t0
    LOG.append({"variant": nm, "block": -1, "seed": -1, "epochs": np.nan, "seconds": sec})
    print(f"{nm}: {5 * (1 if FAST else cfg.get('seeds', N_SEEDS))} models in {sec / 60:.1f} min", flush=True)
for nm, P in FC.items():                                          # every forecast, transformer and baselines
    pd.DataFrame({"model": nm, "target": TGT_OF[nm], "issue": np.repeat(ISSUE, H),
                  "lead": np.tile(np.arange(1, H + 1), len(ISSUE)), "f1": P[:, :, 0].ravel(),
                  "f2": P[:, :, 1].ravel()}).to_parquet(
        os.path.join(OUT, f"{ROUND}_forecasts_{nm.replace(' ', '_').replace(',', '')}.parquet"), index=False)
TL = pd.DataFrame(LOG)
TL.to_csv(os.path.join(OUT, f"{ROUND}_training_log.csv"), index=False)
display(TL[TL["block"] >= 0].groupby("variant")["epochs"].agg(["mean", "min", "max"]).round(1))

# %% CELL 6 -- index forecast scores: correlation with its OWN true index, amplitude (JJAS 2000-2020 target days)
TGT = ISSUE.values[:, None] + np.arange(1, H + 1)[None, :] * np.timedelta64(1, "D")
TGTI = pd.DatetimeIndex(TGT.ravel())
SEL = np.asarray(TGTI.month.isin([6, 7, 8, 9]) & (TGTI.year >= 2000) & (TGTI.year <= 2020)).reshape(TGT.shape)
YR_T = TGTI.year.values.reshape(TGT.shape)


def index_scores(nm):
    P, Y = FC[nm], YOF[TGT_OF[nm]]
    rows = []
    for L in range(H):
        s = SEL[:, L] & np.isfinite(P[:, L]).all(1)
        f, o = P[s, L], Y[s, L]
        rows.append({"model": nm, "target": TGT_OF[nm], "lead": L + 1,
                     "correlation": (f * o).sum() / np.sqrt((f ** 2).sum() * (o ** 2).sum()),
                     "RMSE": np.sqrt(((f - o) ** 2).sum(1).mean()),
                     "amplitude kept": np.hypot(*f.T).mean() / np.hypot(*o.T).mean()})
    return pd.DataFrame(rows)


def crossing(c, thr):
    """first lead (days, interpolated) where the correlation falls below thr; > H if it never does."""
    below = np.where(c < thr)[0]
    if not len(below):
        return np.inf
    i = below[0]
    return 1.0 if i == 0 else i + (c[i - 1] - thr) / (c[i - 1] - c[i])


BS = pd.concat([index_scores(nm) for nm in FC])
BS.to_csv(os.path.join(OUT, f"{ROUND}_index_scores.csv"), index=False)
for col in ("correlation", "amplitude kept"):
    print(f"\n{col} with its own true index")
    display(BS.pivot(index="model", columns="lead", values=col).round(3))
CROSS = pd.DataFrame([{"model": nm, "target": TGT_OF[nm],
                       **{f"days to corr {t}": crossing(BS[BS.model == nm]["correlation"].values, t) for t in (0.5, 0.3)}}
                      for nm in FC])
CROSS.to_csv(os.path.join(OUT, f"{ROUND}_index_crossings.csv"), index=False)
display(CROSS.round(1))
yrsJ = np.arange(2000, 2021)
BI = rng.integers(0, len(yrsJ), (2000, len(yrsJ)))


def year_sums(nm):
    P, Y = FC[nm], YOF[TGT_OF[nm]]
    S = np.zeros((len(yrsJ), H, 3))
    for i, y in enumerate(yrsJ):
        for L in range(H):
            s = SEL[:, L] & (YR_T[:, L] == y) & np.isfinite(P[:, L]).all(1)
            f, o = P[s, L], Y[s, L]
            S[i, L] = (f * o).sum(), (f ** 2).sum(), (o ** 2).sum()
    return S


cor_of = lambda S: S[..., 0] / np.sqrt(S[..., 1] * S[..., 2])
YS = {nm: year_sums(nm) for nm in FC}
PAIRS = [("BSISO T", "ridge BSISO, all inputs"), ("BSISO1-only T", "BSISO T"), ("BSISO T + MISO input", "BSISO T"),
         ("MISO T", "ridge MISO + season"), ("MISO T + BSISO input", "MISO T"), ("MISO T", "BSISO T")]
rows = []
for a_, c_ in PAIRS:
    D = cor_of(YS[a_][BI].sum(1)) - cor_of(YS[c_][BI].sum(1))
    for L in (3, 5, 7, 10, 14, 20):
        rows.append({"model": a_, "vs": c_, "lead": L, "corr difference": D[:, L - 1].mean(),
                     "90% low": np.percentile(D[:, L - 1], 5), "90% high": np.percentile(D[:, L - 1], 95)})
CMP = pd.DataFrame(rows)
CMP.to_csv(os.path.join(OUT, f"{ROUND}_index_comparisons.csv"), index=False)
display(CMP.round(3))

# %% CELL 7 -- rain: daily ERE size-class forecast from each index forecast (RPSS, correlation, AUC), same days
FCI = {nm: pd.DataFrame(P.reshape(len(ISSUE), -1), index=ISSUE) for nm, P in FC.items()}
OBSV = {"bsiso": b[["p1", "p2"]], "miso": MI[["m1", "m2"]]}
REFS = {"true BSISO1 (5 d before)": "bsiso", "true MISO (target day)": "miso"}


def index_at(nm, L):
    """rain-step input for target days at lead L: the index LAG days before the target day, observed when that day
    is on or before the issue day (and for the 'true' references), otherwise forecast from the issue day."""
    tg = REFS.get(nm) or TGT_OF[nm]
    lag = LAG[tg]
    d = times - pd.Timedelta(days=lag)
    if nm in REFS or L <= lag:
        v = OBSV[tg].reindex(d).values
    else:
        lb = L - lag
        v = FCI[nm].reindex(d - pd.Timedelta(days=lb)).values.reshape(-1, H, 2)[:, lb - 1]
    return np.column_stack([v, np.hypot(v[:, 0], v[:, 1])])


def fit_predict(Xtr, ytr, Xte, l2=1e-3):
    Xtr = np.column_stack([np.ones(len(Xtr)), Xtr]); Xte = np.column_stack([np.ones(len(Xte)), Xte])
    w = np.zeros(Xtr.shape[1]); R_ = l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
    for _ in range(30):
        p = 1 / (1 + np.exp(-np.clip(Xtr @ w, -30, 30)))
        step = np.linalg.solve((Xtr * (p * (1 - p))[:, None]).T @ Xtr + R_, Xtr.T @ (p - ytr) + R_ @ w)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return 1 / (1 + np.exp(-np.clip(Xte @ w, -30, 30)))


def loyo(y, Xb):
    good = np.isfinite(Xb).all(1)
    pm, pr = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for yr in np.unique(years[good]):
        te, tr = good & (years == yr), good & (years != yr)
        c = np.nanmean(np.where((years != yr) & good, y, np.nan).reshape(NYR, ND), axis=0)
        c = np.clip(pd.Series(c).rolling(15, center=True, min_periods=1).mean().values, 0.002, 0.998)
        cl = np.log(c / (1 - c))[kidx][:, None]
        pr[te] = fit_predict(cl[tr], y[tr], cl[te])
        pm[te] = fit_predict(np.column_stack([cl, Xb])[tr], y[tr], np.column_stack([cl, Xb])[te])
    return pm, pr, good


YC = [(CLS4 >= t).astype(float) for t in (1, 2, 3)]
OBS1 = YC[0]
UYR = np.unique(years)


def rain_eval(Xb):
    PMs, PRs, G = [], [], np.ones(len(times), bool)
    for y in YC:
        pm, pr, g = loyo(y, Xb)
        PMs.append(pm); PRs.append(pr); G &= g
    PMs, PRs = np.minimum.accumulate(np.array(PMs), 0), np.minimum.accumulate(np.array(PRs), 0)
    Yy = np.array(YC)
    rm = np.where(G, ((np.nan_to_num(PMs) - Yy) ** 2).sum(0), 0)
    rr = np.where(G, ((np.nan_to_num(PRs) - Yy) ** 2).sum(0), 0)
    p = PMs[0][G]
    return {"rps_m": np.array([rm[years == y].sum() for y in yrsJ]),
            "rps_r": np.array([rr[years == y].sum() for y in yrsJ]),
            "corr": np.corrcoef(p, OBS1[G])[0, 1], "corr_normal": np.corrcoef(PRs[0][G], OBS1[G])[0, 1],
            "auc": roc_auc_score(OBS1[G], p), "P": PMs, "Pr": PRs, "G": G}


LEADS = np.arange(1, 15)
RMODELS = list(REFS) + list(FC)
RE = {}
for L in LEADS:
    XL = {nm: index_at(nm, L) for nm in RMODELS}
    ok = (years <= 2020) & np.all([np.isfinite(v).all(1) for v in XL.values()], axis=0)   # the same days for all
    for nm in RMODELS:
        RE[(nm, L)] = rain_eval(np.where(ok[:, None], XL[nm], np.nan))
    print(f"lead {L}: {ok.sum()} target days", flush=True)
rpss = lambda r: 1 - r["rps_m"].sum() / r["rps_r"].sum()
RAIN = pd.DataFrame([{"model": nm, "lead": L, "RPSS": rpss(RE[(nm, L)]), "corr": RE[(nm, L)]["corr"],
                      "corr, normal chance": RE[(nm, L)]["corr_normal"], "AUC": RE[(nm, L)]["auc"],
                      "days": int(RE[(nm, L)]["G"].sum())} for nm in RMODELS for L in LEADS])
RAIN.to_csv(os.path.join(OUT, f"{ROUND}_rain_scores.csv"), index=False)
for col in ("RPSS", "corr"):
    print(f"\nrain {col}")
    display(RAIN.pivot(index="model", columns="lead", values=col).loc[RMODELS].round(3))
BI2 = rng.integers(0, len(yrsJ), (2000, len(yrsJ)))
sk = lambda r: 1 - r["rps_m"][BI2].sum(1) / r["rps_r"][BI2].sum(1)
RPAIRS = PAIRS + [("true MISO (target day)", "true BSISO1 (5 d before)"), ("BSISO T", "true BSISO1 (5 d before)"),
                  ("MISO T", "true MISO (target day)")]
rows = []
for a_, c_ in RPAIRS:
    for L in (1, 3, 5, 8, 10, 12, 14):
        d = sk(RE[(a_, L)]) - sk(RE[(c_, L)])
        rows.append({"model": a_, "vs": c_, "lead": L, "RPSS difference": d.mean(), "90% low": np.percentile(d, 5),
                     "90% high": np.percentile(d, 95), "share of resamples better": (d > 0).mean()})
RCMP = pd.DataFrame(rows)
RCMP.to_csv(os.path.join(OUT, f"{ROUND}_rain_comparisons.csv"), index=False)
display(RCMP.round(4))
prow = []                                                         # daily chances, for the weekly view (done locally)
for (nm, L), r in RE.items():
    g = r["G"]
    prow.append(pd.DataFrame({"model": nm, "lead": L, "target": times[g], "P1": r["P"][0][g], "P2": r["P"][1][g],
                              "P3": r["P"][2][g], "normal P1": r["Pr"][0][g], "class": CLS4[g]}))
pd.concat(prow).to_parquet(os.path.join(OUT, f"{ROUND}_daily_cumulative_probabilities.parquet"), index=False)

# %% CELL 8 -- figures
PAL = ["#2a78d6", "#1baf7a", "#d4572a", "#7b5ea7", "#c9a227", "#e05d8f", "#3aa0c8", "#123e7a", "#8a5a19", "#52514e"]
COLS = {"persistence BSISO": "#a3a29d", "persistence MISO": "#d9b98a", "ridge BSISO, all inputs": "#52514e",
        "ridge MISO + season": "#8a5a19", "ridge MISO + BSISO": "#c9a227", "true BSISO1 (5 d before)": INK,
        "true MISO (target day)": "#e05d8f"}
for i, nm in enumerate(VARIANTS):
    COLS[nm] = PAL[i % len(PAL)]
STY = lambda nm: dict(color=COLS[nm], lw=2.2 if nm in VARIANTS else 1.4, ls="-" if nm in VARIANTS else "--",
                      marker="o" if nm in VARIANTS else None, ms=3.5)
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
for nm in FC:
    v = BS[BS["model"] == nm]
    ax[0].plot(v["lead"], v["correlation"], label=nm, **STY(nm))
    ax[1].plot(v["lead"], v["amplitude kept"], label=nm, **STY(nm))
for y_ in (0.5, 0.3):
    ax[0].axhline(y_, color=INK2, lw=0.8, ls=":")
ax[0].set(xlabel="lead (days)", ylabel="correlation with its own true index", xticks=range(1, H + 1, 2))
ax[1].set(xlabel="lead (days)", ylabel="amplitude kept (1 = none lost)", xticks=range(1, H + 1, 2))
ax[0].set_title("(a) forecast correlation (each model vs the index it forecasts)", loc="left", fontsize=10.5)
ax[1].set_title("(b) amplitude kept", loc="left", fontsize=10.5)
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"{ROUND}: forecasting BSISO1 vs MISO, same issue days (May-Oct 1998-2020), JJAS 2000-2020 targets",
             x=0.01, ha="left", fontsize=13, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig(os.path.join(OUT, f"{ROUND}_fig1_index.png"), dpi=130, bbox_inches="tight")
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
for j, col in enumerate(("RPSS", "corr")):
    for nm in RMODELS:
        v = RAIN[RAIN["model"] == nm]
        ax[j].plot(v["lead"], v[col], label=nm, **STY(nm))
    if col == "corr":
        v = RAIN[RAIN["model"] == RMODELS[0]]
        ax[j].plot(v["lead"], v["corr, normal chance"], color=INK2, ls=":", lw=1.2, label="normal chance")
    ax[j].set(xlabel="rain lead (days)", xticks=LEADS, ylabel=col)
ax[0].axhline(0, color=INK2, lw=0.8)
ax[0].set_title("(a) skill of the 4-class daily ERE forecast (RPSS; 0 = normal chance)", loc="left", fontsize=10.5)
ax[1].set_title("(b) correlation of P(ERE) with the observed ERE days", loc="left", fontsize=10.5)
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"{ROUND}: daily ERE forecast in central India from BSISO1 vs MISO forecasts (same days)", x=0.01,
             ha="left", fontsize=13, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig(os.path.join(OUT, f"{ROUND}_fig2_rain.png"), dpi=130, bbox_inches="tight")
plt.show()
