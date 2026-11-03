# BSISO_Forecast_Tests: compare forecast architectures for BSISO1, and the daily ERE size-class forecast they lead to
# Every variant is trained the same way: 5 blocks of years (each forecast by models trained on the other 4), several
# seeds averaged, 60 days of BSISO history in, BSISO1 (PC1, PC2) 1-10 days out. Scored on JJAS 2000-2024 target days.
# Rain: the forecast BSISO for 5 days before the target day -> daily P(no ERE / 16-39 / 40-90 / 91+ cells) in central
# India, each summer predicted from the other 24. Inputs (Kaggle datasets): BSISO.INDEX.NORM.LY.data,
# imerg_india_025_jjas_2000_2024.nc, india_mask_025.npy; optional lasso_bsiso_forecasts.parquet.

# %% CELL 1 -- settings: which variants this round tests
ROUND = "R7"
# each variant: model type and its settings; "win" = days of history used; "amp_w" = weight of the amplitude term;
# "lead_w" = weight per lead day (1-10); "inputs" = channels the network sees (0,1 BSISO1; 2,3 BSISO2; 4,5 season);
# "aux" = the network also forecasts BSISO2 (PC3, PC4), with this weight on its error (a helper task; only the BSISO1
# forecast is used); "ck" = reuse the saved models of another variant for the seeds they share; "save_probs" = write
# the daily size-class probability table from this variant
# "inputs" may also be "all" = every channel, including the OLR rain band (channels 6.. : 17 latitude bins + a
# "band available" flag; built in cell 2)
# R7: the monsoon rain band as an extra transformer input. Band = NOAA CBO OLR anomaly averaged over 70-95E in
# 2.5-degree latitude bins from 10S to 30N (low OLR = deep cloud/rain), daily 1991-2026; zero + flag 0 before 1991.
BEST = dict(kind="patch", d=64, heads=4, layers=3, ff=128, win=60, amp_w=0.5, lin=True)
VARIANTS = {
    "T best":                        BEST,
    "T + OLR band":                  {**BEST, "inputs": "all"},
    "T + OLR band, wider (d 96)":    {**BEST, "inputs": "all", "d": 96, "ff": 192},
    "T + OLR band, no BSISO2":       {**BEST, "inputs": "nobsiso2"},
}
N_SEEDS = 10                       # a variant may override: seeds=..., batch=..., maxep=..., lr=..., val_tf=True

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


def find(pattern, required=True):
    for r in ROOTS:
        hits = sorted(glob.glob(os.path.join(r, "**", pattern), recursive=True))
        if hits:
            return hits[0]
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
FEAT = np.column_stack([b[["p1", "p2", "q1", "q2"]].values, np.sin(2 * np.pi * doy / 365.25),
                        np.cos(2 * np.pi * doy / 365.25)]).astype("float32")   # BSISO1, BSISO2, season
PCS = b[["p1", "p2"]].values.astype("float32")
olr_f = find("olr_cbo_daily_1991_2026_30S45N_30E130E.nc", required=False)
if olr_f:                                                         # the OLR rain band (see cell 1)
    o = xr.open_dataset(olr_f)["olr"].sel(lon=slice(70, 95)).mean("lon").sel(lat=slice(30, -10))
    od = pd.DataFrame(o.values, index=pd.DatetimeIndex(o.time.values).normalize()).reindex(DATES)
    od = od.interpolate(limit=3, limit_area="inside")
    base = od[(od.index.year >= 1991) & (od.index.year <= 2020)]
    cl = base.groupby(base.index.dayofyear).mean().reindex(range(1, 367)).interpolate()
    cl = pd.concat([cl, cl, cl]).rolling(31, center=True).mean().iloc[366:732]   # smoothed, wraps the year
    cl.index = range(1, 367)
    an = od.values - cl.loc[doy].values
    an = an / np.nanstd(an[(DATES.year >= 1991) & (DATES.year <= 2020)], axis=0)
    FLAG = np.isfinite(an).all(1).astype("float32")
    BAND = np.nan_to_num(an).astype("float32") * FLAG[:, None]
    FEAT = np.column_stack([FEAT, BAND, FLAG]).astype("float32")
    print(f"OLR band: {BAND.shape[1]} latitude bins, available on {FLAG.mean():.0%} of days since 1981")
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

# %% CELL 3 -- samples (60 days of history; shorter windows use the last part) and year blocks
WMAX, H = max(60, max(c.get('win', 60) for c in VARIANTS.values())), 10
tt = np.arange(WMAX - 1, len(DATES) - H)
tt = tt[np.array([np.isfinite(FEAT[t - WMAX + 1:t + H + 1]).all() for t in tt])]
X = np.stack([FEAT[t - WMAX + 1:t + 1] for t in tt])
Y = np.stack([PCS[t + 1:t + H + 1] for t in tt])
Y4 = np.stack([FEAT[t + 1:t + H + 1, :4] for t in tt])                # BSISO1 and BSISO2 (for the helper task)
ISSUE = DATES[tt]
uyr = np.unique(ISSUE.year)
BLK = np.array([{y: i * 5 // len(uyr) for i, y in enumerate(uyr)}[y] for y in ISSUE.year])
Xt, Yt, Y4t = torch.tensor(X).to(DEV), torch.tensor(Y).to(DEV), torch.tensor(Y4).to(DEV)        # whole data set lives on the GPU
print(f"{len(tt):,} samples")

# %% CELL 4 -- the model zoo
class PatchModel(nn.Module):
    """Encoder-only patch transformer (after PatchTST, Nie et al. 2023); all inputs share a token; direct 1-10 d head;
    optional linear path from the PC1/PC2 history (after DLinear, Zeng et al. 2023)."""
    def __init__(self, win, d, heads, layers, ff, lin=True, patch=5, drop=0.1, prob=False, inputs=(0, 1, 2, 3, 4, 5), aux=0, **_):
        super().__init__()
        if inputs == "all":
            inputs = range(N_FEAT)
        elif inputs == "nobsiso2":                                # BSISO1 + season + band
            inputs = [0, 1, 4, 5] + list(range(6, N_FEAT))
        self.win, self.np_, self.prob, self.inp = win, win // patch, prob, list(inputs)
        self.no = 4 if aux else 2                                 # outputs: BSISO1 (+ BSISO2 for the helper task)
        self.embed = nn.Linear(patch * len(self.inp), d)
        self.pos = nn.Parameter(torch.zeros(1, self.np_, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True,
                                                                    norm_first=True), layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(self.np_ * d, H * self.no)
        self.lin = nn.Linear(win * self.no, H * self.no) if lin else None
        if lin:
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.var = nn.Linear(self.np_ * d, H * 2) if prob else None  # log-variance head (Delaunay & Christensen 2022)

    def forward(self, x, y=None):
        x = x[:, -self.win:]
        B = x.shape[0]
        z = self.norm(self.enc(self.embed(x[:, :, self.inp].reshape(B, self.np_, -1)) + self.pos)).reshape(B, -1)
        out = self.head(z)
        if self.lin is not None:
            out = out + self.lin(x[:, :, :self.no].reshape(B, -1))
        out = out.reshape(B, H, self.no)
        if self.prob:                                             # (B, H, 4): mean PC1, PC2, log-variance PC1, PC2
            out = torch.cat([out, self.var(z).reshape(B, H, 2).clamp(-6, 4)], dim=-1)
        return out

    def predict(self, x, y=None):                                 # the BSISO1 forecast (drops the helper output)
        out = self.forward(x)
        return out if self.no == 2 else out[..., :2]


class GRUModel(nn.Module):
    def __init__(self, win, hidden, layers, lin=True, n_in=6, drop=0.1, **_):
        super().__init__()
        self.win = win
        self.gru = nn.GRU(n_in, hidden, layers, batch_first=True, dropout=drop)
        self.head = nn.Linear(hidden, H * 2)
        self.lin = nn.Linear(win * 2, H * 2) if lin else None
        if lin:
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, y=None):
        x = x[:, -self.win:]
        B = x.shape[0]
        out = self.head(self.gru(x)[0][:, -1])
        if self.lin is not None:
            out = out + self.lin(x[:, :, :2].reshape(B, -1))
        return out.reshape(B, H, 2)

    predict = forward


def sinusoid(n, d):
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class EncDecModel(nn.Module):
    """Encoder-decoder transformer as in Anirudh et al. (2025): 4+4 layers, width 120, post-norm, teacher forcing in
    training, day-by-day (autoregressive) forecasting at inference."""
    def __init__(self, win, d, heads, layers, ff, n_in=6, drop=0.1, **_):
        super().__init__()
        self.win = win
        self.src = nn.Linear(n_in, d)
        self.tgt = nn.Linear(2, d)
        self.register_buffer("pe", sinusoid(win + H, d))
        self.tf = nn.Transformer(d, heads, layers, layers, ff, drop, batch_first=True)
        self.out = nn.Linear(d, 2)
        self.register_buffer("causal", nn.Transformer.generate_square_subsequent_mask(H))

    def _run(self, x, yin):
        s = self.src(x) + self.pe[:x.shape[1]]
        t = self.tgt(yin) + self.pe[:yin.shape[1]]
        n = yin.shape[1]
        return self.out(self.tf(s, t, tgt_mask=self.causal[:n, :n]))

    def forward(self, x, y):                                      # teacher forcing: the true previous day goes in
        x = x[:, -self.win:]
        yin = torch.cat([x[:, -1:, :2], y[:, :-1]], dim=1)
        return self._run(x, yin)

    @torch.no_grad()
    def predict(self, x, y=None):                                 # day by day, feeding back its own forecast
        x = x[:, -self.win:]
        yin = x[:, -1:, :2]
        for _ in range(H):
            nxt = self._run(x, yin)[:, -1:]
            yin = torch.cat([yin, nxt], dim=1)
        return yin[:, 1:]


ZOO = {"patch": PatchModel, "gru": GRUModel, "encdec": EncDecModel}
for nm, cfg in VARIANTS.items():
    m = ZOO[cfg["kind"]](**cfg)
    print(f"{nm:16s} {sum(p.numel() for p in m.parameters()):>10,} parameters")

# %% CELL 5 -- training: every variant x 5 year blocks x seeds; forecasts saved after each variant
MAXEP, PATIENCE, BATCH = (2, 2, 256) if FAST else (200, 15, 256)


ISSUE_JJAS = torch.tensor(np.isin(ISSUE.month, [5, 6, 7, 8, 9]), device=DEV)   # forecast targets fall in summer


def loss_fn(p, y, amp_w, w=None):
    """(1 - amp_w) x MSE on (PC1, PC2) + amp_w x MSE on amplitude; w = optional weight per (sample, lead)."""
    amp = lambda v: torch.sqrt((v ** 2).sum(-1) + 1e-6)
    if p.shape[-1] == 4:                                          # probabilistic: NLL on the mean and variance
        mu, lv = p[..., :2], p[..., 2:]
        e1 = (0.5 * (lv + (y - mu) ** 2 / torch.exp(lv))).mean(-1)
        p = mu
    else:
        e1 = ((p - y) ** 2).mean(-1)
    e2 = (amp(p) - amp(y)) ** 2
    if w is None:
        return (1 - amp_w) * e1.mean() + amp_w * e2.mean()
    return ((1 - amp_w) * (e1 * w).sum() + amp_w * (e2 * w).sum()) / w.sum()


def total_loss(cfg, p, j, w):
    """loss_fn on BSISO1, plus aux x MSE on BSISO2 when the variant has the helper task."""
    if not cfg.get("aux"):
        return loss_fn(p, Yt[j], cfg["amp_w"], w)
    return loss_fn(p[..., :2], Yt[j], cfg["amp_w"], w) + cfg["aux"] * ((p[..., 2:] - Y4t[j][..., 2:]) ** 2).mean()


def weights(cfg, j):
    """weight per (sample, lead) from the variant's jjas_w and lead_w settings (None = plain average)."""
    if "jjas_w" not in cfg and "lead_w" not in cfg:
        return None
    sw = torch.where(ISSUE_JJAS[j], cfg.get("jjas_w", 1.0), 1.0)
    lw = torch.tensor(cfg.get("lead_w", [1.0] * H), device=DEV, dtype=torch.float32)
    return sw[:, None] * lw[None, :]


def train_one(cfg, tr, va, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if FAST:                                                      # quick local check: small subset
        tr, va = tr[:3000], va[:800]
    maxep, batch = (MAXEP if FAST else cfg.get("maxep", MAXEP)), cfg.get("batch", BATCH)
    m = ZOO[cfg["kind"]](**cfg).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.get("lr", 1e-3 if cfg["kind"] != "encdec" else 1e-4),
                            weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    trt = torch.tensor(tr, device=DEV)
    best, bad, state = np.inf, 0, None
    for ep in range(maxep):
        m.train()
        perm = trt[torch.randperm(len(trt), device=DEV)]
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            opt.zero_grad()
            loss = total_loss(cfg, m(Xt[j], Yt[j]), j, weights(cfg, j))
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            fwd = (lambda xx, yy: m(xx, yy)) if cfg.get("val_tf") or cfg.get("aux") else (lambda xx, yy: m.predict(xx))
            vat = torch.tensor(va, device=DEV)
            v = np.mean([total_loss(cfg, fwd(Xt[vat[i:i + 2048]], Yt[vat[i:i + 2048]]), vat[i:i + 2048],
                                    weights(cfg, vat[i:i + 2048])).item() for i in range(0, len(va), 2048)])
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


FC = {"persistence": np.repeat(X[:, -1:, :2], H, axis=1)}
RIDGES = [("ridge", X[:, :, :2].reshape(len(X), -1)), ("ridge, all inputs", X[:, :, :6].reshape(len(X), -1))]
if N_FEAT > 6:
    RIDGES.append(("ridge + OLR band", X.reshape(len(X), -1)))
for nm, Xf in RIDGES:
    P = np.zeros_like(Y)
    for k in range(5):
        te, tr = BLK == k, BLK != k
        P[te] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xf[tr], Y[tr].reshape(tr.sum(), -1)).predict(Xf[te]).reshape(-1, H, 2)
    FC[nm] = P
lz = find("lasso_bsiso_forecasts.parquet", required=False)
if lz:
    LZ = pd.read_parquet(lz).set_index(["lead", "issue"])[["f1", "f2"]]
    P = np.full_like(Y, np.nan)
    for L in range(1, H + 1):
        P[:, L - 1] = LZ.reindex(pd.MultiIndex.from_arrays([np.full(len(ISSUE), L), ISSUE])).values
    FC["lasso"] = P
BASELINES = list(FC)
REF_T = next(iter(VARIANTS))                                      # comparisons: each variant vs the first one
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
LOG, SPR = [], {}                                                 # SPR: forecast spread for probabilistic variants
for nm, cfg in VARIANTS.items():
    P = np.zeros_like(Y)
    t0 = time.time()
    for k in range(5):
        te = np.where(BLK == k)[0]
        va = np.where((BLK != k) & np.isin(ISSUE.year, VA_YEARS[k]))[0]
        tr = np.where((BLK != k) & ~np.isin(ISSUE.year, VA_YEARS[k]))[0]
        preds = []
        for s in range(1 if FAST else cfg.get("seeds", N_SEEDS)):
            key = f"{ROUND}_{cfg.get('ck', nm).replace(' ', '_')}_b{k}_s{s}{'_fast' if FAST else ''}.npz"
            if key not in CK_FOUND and os.path.exists(os.path.join(CKPT, key)):
                CK_FOUND[key] = os.path.join(CKPT, key)           # saved earlier in this run by a variant it shares
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
        if preds[0].shape[-1] == 4:                               # seeds' mixture: mean, and spread incl. seed spread
            pr_ = np.stack(preds)
            mu = pr_[..., :2].mean(0)
            var = (np.exp(pr_[..., 2:]) + pr_[..., :2] ** 2).mean(0) - mu ** 2
            P[te] = mu
            SPR.setdefault(nm, np.full(Y.shape[:2], np.nan))[te] = np.sqrt(var.sum(-1))
        else:
            P[te] = np.mean(preds, axis=0)
        print(f"  {nm}: block {k + 1}/5 done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    FC[nm] = P
    sec = time.time() - t0
    LOG.append({"variant": nm, "block": -1, "seed": -1, "epochs": np.nan, "seconds": sec})
    pd.DataFrame({"variant": nm, "issue": np.repeat(ISSUE, H), "lead": np.tile(np.arange(1, H + 1), len(ISSUE)),
                  "f1": P[:, :, 0].ravel(), "f2": P[:, :, 1].ravel(),
                  "spread": SPR[nm].ravel() if nm in SPR else np.nan}).to_parquet(
        os.path.join(OUT, f"{ROUND}_forecasts_{nm.replace(' ', '_')}.parquet"), index=False)
    print(f"{nm}: {5 * (1 if FAST else cfg.get('seeds', N_SEEDS))} models in {sec / 60:.1f} min", flush=True)
TL = pd.DataFrame(LOG)
TL.to_csv(os.path.join(OUT, f"{ROUND}_training_log.csv"), index=False)
display(TL[TL["block"] >= 0].groupby("variant")["epochs"].agg(["mean", "min", "max"]).round(1))

# %% CELL 6 -- BSISO forecast scores: correlation, error, amplitude, phase (JJAS 2000-2024 target days)
TGT = ISSUE.values[:, None] + np.arange(1, H + 1)[None, :] * np.timedelta64(1, "D")
TGTI = pd.DatetimeIndex(TGT.ravel())
SEL = np.asarray(TGTI.month.isin([6, 7, 8, 9]) & (TGTI.year >= 2000) & (TGTI.year <= 2024)).reshape(TGT.shape)
YR_T = TGTI.year.values.reshape(TGT.shape)
ang = lambda v: np.degrees(np.arctan2(v[..., 0], -v[..., 1])) % 360


def bsiso_scores(P):
    rows = []
    for L in range(H):
        s = SEL[:, L] & np.isfinite(P[:, L]).all(1)
        f, o = P[s, L], Y[s, L]
        strong = np.hypot(*o.T) >= 1
        perr = np.abs((ang(f) - ang(o) + 180) % 360 - 180)[strong]
        rows.append({"lead": L + 1, "correlation": (f * o).sum() / np.sqrt((f ** 2).sum() * (o ** 2).sum()),
                     "RMSE": np.sqrt(((f - o) ** 2).sum(1).mean()),
                     "amplitude kept": np.hypot(*f.T).mean() / np.hypot(*o.T).mean(),
                     "phase error (deg)": np.median(perr), "phase right or 1 off": (perr <= 67.5).mean()})
    return pd.DataFrame(rows)


BS = pd.concat([bsiso_scores(P).assign(model=nm) for nm, P in FC.items()])
BS.to_csv(os.path.join(OUT, f"{ROUND}_bsiso_scores.csv"), index=False)
for col in ("correlation", "RMSE", "amplitude kept", "phase error (deg)"):
    print(f"\n{col}")
    display(BS.pivot(index="model", columns="lead", values=col).round(3))
yrsJ = np.arange(2000, 2025)
BI = rng.integers(0, len(yrsJ), (2000, len(yrsJ)))


def year_sums(P):
    S = np.zeros((len(yrsJ), H, 3))
    for i, y in enumerate(yrsJ):
        for L in range(H):
            s = SEL[:, L] & (YR_T[:, L] == y) & np.isfinite(P[:, L]).all(1)
            f, o = P[s, L], Y[s, L]
            S[i, L] = (f * o).sum(), (f ** 2).sum(), (o ** 2).sum()
    return S


cor_of = lambda S: S[..., 0] / np.sqrt(S[..., 1] * S[..., 2])
YS = {nm: year_sums(P) for nm, P in FC.items()}
rows = []
for nm in VARIANTS:
    for base in ("ridge, all inputs", REF_T):
        if nm == base or base not in FC:
            continue
        D = cor_of(YS[nm][BI].sum(1)) - cor_of(YS[base][BI].sum(1))
        for L in (3, 5, 7, 9):
            rows.append({"variant": nm, "vs": base, "lead": L, "corr difference": D[:, L - 1].mean(),
                         "90% low": np.percentile(D[:, L - 1], 5), "90% high": np.percentile(D[:, L - 1], 95)})
CMP = pd.DataFrame(rows)
CMP.to_csv(os.path.join(OUT, f"{ROUND}_bsiso_comparisons.csv"), index=False)
display(CMP.round(3))

# %% CELL 7 -- rain: daily ERE size-class forecast from each BSISO forecast (RPSS, correlation, AUC)
FCI = {nm: pd.DataFrame(P.reshape(len(ISSUE), -1), index=ISSUE) for nm, P in FC.items()}


def bsiso_T5(nm, L):
    d5 = times - pd.Timedelta(days=5)
    if L <= 5 or nm == "observed":
        v = b[["p1", "p2"]].reindex(d5).values
    else:
        lb = L - 5
        v = FCI[nm].reindex(d5 - pd.Timedelta(days=lb)).values.reshape(-1, H, 2)[:, lb - 1]
        if nm in SPR:                                             # forecast spread as an extra input to the rain step
            sp = pd.DataFrame(SPR[nm], index=ISSUE).reindex(d5 - pd.Timedelta(days=lb)).values[:, lb - 1]
            return np.column_stack([v, np.hypot(v[:, 0], v[:, 1]), sp])
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
    for yr in np.unique(years):
        te, tr = good & (years == yr), good & (years != yr)
        c = np.nanmean(np.where(years != yr, y, np.nan).reshape(NYR, ND), axis=0)
        c = np.clip(pd.Series(c).rolling(15, center=True, min_periods=1).mean().values, 0.002, 0.998)
        cl = np.log(c / (1 - c))[kidx][:, None]
        pr[te] = fit_predict(cl[tr], y[tr], cl[te])
        pm[te] = fit_predict(np.column_stack([cl, Xb])[tr], y[tr], np.column_stack([cl, Xb])[te])
    return pm, pr, good


YC = [(CLS4 >= t).astype(float) for t in (1, 2, 3)]
OBS1 = YC[0]


def rain_eval(Xb):
    PMs, PRs, G = [], [], np.ones(len(times), bool)
    for y in YC:
        pm, pr, g = loyo(y, Xb)
        PMs.append(pm); PRs.append(pr); G &= g
    PMs, PRs = np.minimum.accumulate(np.array(PMs), 0), np.minimum.accumulate(np.array(PRs), 0)
    Yy = np.array(YC)
    rm = np.where(G, ((PMs - Yy) ** 2).sum(0), 0); rr = np.where(G, ((PRs - Yy) ** 2).sum(0), 0)
    p = PMs[0][G]
    return {"rps_m": np.array([rm[years == y].sum() for y in np.unique(years)]),
            "rps_r": np.array([rr[years == y].sum() for y in np.unique(years)]),
            "corr": np.corrcoef(p, OBS1[G])[0, 1], "auc": roc_auc_score(OBS1[G], p), "P": PMs, "Pr": PRs, "G": G}


LEADS = np.arange(6, 15)
RE = {("observed", 5): rain_eval(bsiso_T5("observed", 5))}
for nm in FC:
    for L in LEADS:
        RE[(nm, L)] = rain_eval(bsiso_T5(nm, L))
rpss = lambda r: 1 - r["rps_m"].sum() / r["rps_r"].sum()
REF = RE[("observed", 5)]
RAIN = pd.DataFrame([{"model": nm, "lead": L, "RPSS": rpss(RE[(nm, L)]), "corr": RE[(nm, L)]["corr"],
                      "AUC": RE[(nm, L)]["auc"]} for nm in FC for L in LEADS])
RAIN.to_csv(os.path.join(OUT, f"{ROUND}_rain_scores.csv"), index=False)
CNAMES = ["no ERE", "16-39 cells", "40-90 cells", "91+ cells"]
cls_p = lambda C: np.array([1 - C[0], C[0] - C[1], C[1] - C[2], C[2]])   # cumulative chances -> 4 class chances
for nm in [n for n, c in VARIANTS.items() if c.get("save_probs")]:     # the daily probability table (as Arm A's)
    rows = []
    for L in range(1, 15):
        r = RE[("observed", 5)] if L <= 5 else RE[(nm, L)]
        Pc, Pr = cls_p(r["P"]), cls_p(r["Pr"])
        rows.append(pd.DataFrame({"issue": times - pd.Timedelta(days=L), "target": times, "lead": L,
                                  **{f"P({CNAMES[c]})": Pc[c] for c in range(4)},
                                  **{f"normal P({CNAMES[c]})": Pr[c] for c in range(4)},
                                  "observed": np.array(CNAMES)[CLS4]})[r["G"]])
    TAB = pd.concat(rows).sort_values(["issue", "lead"]).reset_index(drop=True)
    TAB.to_csv(os.path.join(OUT, f"{ROUND}_daily_ere_class_probabilities.csv"), index=False)
    print(f"{nm}: saved {len(TAB):,} forecasts (issue day x lead 1-14; leads 1-5 use the observed BSISO)")
print(f"reference (true BSISO 5 days before): RPSS {rpss(REF):.3f}, corr {REF['corr']:.3f}, AUC {REF['auc']:.3f}")
for col in ("RPSS", "corr"):
    print(f"\nrain {col}")
    display(RAIN.pivot(index="model", columns="lead", values=col).round(3))
BI2 = rng.integers(0, NYR, (2000, NYR))
rows = []
for nm in VARIANTS:
    for base in ("ridge, all inputs", REF_T):
        if nm == base or base not in FC:
            continue
        for L in (8, 10, 12, 14):
            a_, c_ = RE[(nm, L)], RE[(base, L)]
            d = (1 - a_["rps_m"][BI2].sum(1) / a_["rps_r"][BI2].sum(1)) - (1 - c_["rps_m"][BI2].sum(1) / c_["rps_r"][BI2].sum(1))
            rows.append({"variant": nm, "vs": base, "lead": L, "RPSS difference": d.mean(),
                         "90% low": np.percentile(d, 5), "90% high": np.percentile(d, 95)})
RCMP = pd.DataFrame(rows)
RCMP.to_csv(os.path.join(OUT, f"{ROUND}_rain_comparisons.csv"), index=False)
display(RCMP.round(4))

# %% CELL 8 -- figures
PAL = ["#2a78d6", "#1baf7a", "#d4572a", "#7b5ea7", "#c9a227", "#e05d8f", "#3aa0c8", "#123e7a", "#8a5a19", "#52514e"]
COLS = {"persistence": "#a3a29d", "ridge": "#8a5a19", "ridge, all inputs": "#52514e", "lasso": "#d9b98a", "ridge + OLR band": "#0b0b0b"}
for i, nm in enumerate(VARIANTS):
    COLS[nm] = PAL[i % len(PAL)]
STY = lambda nm: dict(color=COLS[nm], lw=2.2 if nm in VARIANTS else 1.4, ls="-" if nm in VARIANTS else "--",
                      marker="o" if nm in VARIANTS else None, ms=3.5)
fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
for j, (col, t) in enumerate((("correlation", "(a) correlation with the true BSISO"), ("RMSE", "(b) error"),
                              ("amplitude kept", "(c) amplitude kept (1 = none lost)"),
                              ("phase error (deg)", "(d) median phase error, strong days"))):
    for nm in FC:
        v = BS[BS["model"] == nm]
        ax[j].plot(v["lead"], v[col], label=nm, **STY(nm))
    ax[j].set(xlabel="BSISO lead (days)", xticks=range(1, H + 1))
    ax[j].set_title(t, loc="left", fontsize=10.5)
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"{ROUND}: forecasting BSISO1 (JJAS 2000-2024, out of sample, {N_SEEDS} seeds per variant)", x=0.01,
             ha="left", fontsize=13, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig(os.path.join(OUT, f"{ROUND}_fig1_bsiso.png"), dpi=130, bbox_inches="tight")
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
for j, (col, ref) in enumerate((("RPSS", rpss(REF)), ("corr", REF["corr"]))):
    ax[j].axhline(ref, color=INK, ls=":", lw=1.2, label="true BSISO 5 days before (reference)")
    for nm in FC:
        v = RAIN[RAIN["model"] == nm]
        ax[j].plot(v["lead"], v[col], label=nm, **STY(nm))
    ax[j].set(xlabel="rain lead (days)", xticks=LEADS, ylabel=col)
ax[0].set_title("(a) skill of the 4-class daily ERE forecast (RPSS)", loc="left", fontsize=10.5)
ax[1].set_title("(b) correlation of P(large ERE) with the observed ERE days", loc="left", fontsize=10.5)
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"{ROUND}: daily ERE size-class forecast in central India from each BSISO forecast", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig(os.path.join(OUT, f"{ROUND}_fig2_rain.png"), dpi=130, bbox_inches="tight")
plt.show()
