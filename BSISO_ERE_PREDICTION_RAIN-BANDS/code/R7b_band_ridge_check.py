import os, re, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
PROJ = r"C:\Users\divya\OneDrive\Desktop\IIT- Bombay Summer 26\Moaap\Regional_claude"
os.environ["BSISO_DATA"] = os.path.join(PROJ, "data"); os.environ["FAST"] = "1"
src = open(os.path.join(PROJ, "sem_project_7", "other", "scripts", "kaggle_BSISO_Tests.py"), encoding="utf-8").read()
cells = re.split(r"^# %% CELL ", src, flags=re.M)[1:]
ns = {"__name__": "__main__", "display": lambda *a: None}
import matplotlib; matplotlib.use("Agg")
for c in cells[:3]:
    exec(compile("# " + c, "cell", "exec"), ns)
X, Y, BLK, ISSUE, FEAT, DATES, H = (ns[k] for k in ("X", "Y", "BLK", "ISSUE", "FEAT", "DATES", "H"))
TGT = ISSUE.values[:, None] + np.arange(1, H + 1)[None, :] * np.timedelta64(1, "D")
TG = pd.DatetimeIndex(TGT.ravel())
SEL = np.asarray(TG.month.isin([6, 7, 8, 9]) & (TG.year >= 2000) & (TG.year <= 2024)).reshape(TGT.shape)
def corr(P, L):
    s = SEL[:, L]; f, o = P[s, L], Y[s, L]
    return (f * o).sum() / np.sqrt((f ** 2).sum() * (o ** 2).sum())
band = X[:, :, 6:23]; flag = X[:, -1, 23]
print("band available on", flag.mean().round(2), "of samples; band std (avail)", band[flag > 0].std().round(2))
# sanity: does the issue-day band relate to the BSISO? correlation of band bins with PC1/PC2 on the issue day
b0 = band[:, -1]; ok = flag > 0
print("max |corr| band bin vs PC1:", np.abs([np.corrcoef(b0[ok, i], X[ok, -1, 0])[0, 1] for i in range(17)]).max().round(2),
      " vs PC2:", np.abs([np.corrcoef(b0[ok, i], X[ok, -1, 1])[0, 1] for i in range(17)]).max().round(2))
U, S, Vt = np.linalg.svd(b0[ok] - b0[ok].mean(0), full_matrices=False)
pcs = lambda a: (a - b0[ok].mean(0)) @ Vt[:3].T
feats = {
  "ridge, all inputs (60 d x 6)": X[:, :, :6].reshape(len(X), -1),
  "+ band, 60 d x 17 bins (as R7)": X.reshape(len(X), -1),
  "+ band 3 patterns, last 5 days": np.column_stack([X[:, :, :6].reshape(len(X), -1), pcs(band[:, -5:]).reshape(len(X), -1) * flag[:, None], flag]),
  "+ band 3 patterns, issue day only": np.column_stack([X[:, :, :6].reshape(len(X), -1), pcs(band[:, -1]) * flag[:, None], flag]),
}
for nm, Xf in feats.items():
    for how in ("RidgeCV (GCV alpha)", "block-CV alpha"):
        P = np.zeros_like(Y)
        for k in range(5):
            te, tr = BLK == k, BLK != k
            if how.startswith("RidgeCV"):
                m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xf[tr], Y[tr].reshape(tr.sum(), -1))
            else:   # choose alpha by leaving out one training block (respects autocorrelation)
                best = None
                for a in np.logspace(-1, 5, 13):
                    err = 0
                    for kk in set(range(5)) - {k}:
                        t2, v2 = tr & (BLK != kk), tr & (BLK == kk)
                        p = Ridge(alpha=a).fit(Xf[t2], Y[t2].reshape(t2.sum(), -1)).predict(Xf[v2])
                        err += ((p - Y[v2].reshape(v2.sum(), -1)) ** 2).sum()
                    best = (err, a) if best is None or err < best[0] else best
                m = Ridge(alpha=best[1]).fit(Xf[tr], Y[tr].reshape(tr.sum(), -1))
            P[te] = m.predict(Xf[te]).reshape(-1, H, 2)
        print(f"{nm:38s} {how:20s} corr 3/5/7/9 d: " + " ".join(f"{corr(P, L - 1):.3f}" for L in (3, 5, 7, 9)), flush=True)
