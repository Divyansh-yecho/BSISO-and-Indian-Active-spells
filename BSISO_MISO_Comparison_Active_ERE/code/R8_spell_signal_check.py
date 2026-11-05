# R8 follow-up (local): check the Dey et al. (2022) result that MISO phase tracks active/break spells, and ask
# WHEN the index knows. Spells as in Dey: core-monsoon-zone standardised rain > 0.9 (< -0.9) for >= 4 days with a
# spell-mean > 1.2 (< -1.2), 10 Jun - 15 Sep. Rain: IMERG CMZ (spell_drivers_daily.csv), 2000-2020 (MISO stops 2020).
# 1) Dey's same-day statistic: share of spell days (amplitude > 1) in MISO phases 2-5 / 6-8 (BSISO1: phases 4-6 / 8,1,2).
# 2) Lead-lag: index taken k days before (+) or after (-) each day -> leave-one-year-out logistic P(active day),
#    P(break day); AUC and Brier skill vs normal chance. Same days for both indices.
import os, glob
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJ = r"C:\Users\divya\OneDrive\Desktop\IIT- Bombay Summer 26\Moaap\Regional_claude"
OUTT = os.path.join(PROJ, "sem_project_7", "other", "intermediate", "R8_miso", "spell_check")
os.makedirs(OUTT, exist_ok=True)
D = pd.read_csv(os.path.join(PROJ, "Summer_26", "Other", "derived", "spell_drivers_daily.csv"))
D.index = pd.to_datetime(D.date.astype(str))
D = D[D.index.year <= 2020]


def spells(z, sign):
    """Dey et al. 2022 rule on one season's daily series z (sign +1 active, -1 break)"""
    x = sign * z.values
    out = np.zeros(len(x), bool)
    i = 0
    while i < len(x):
        if x[i] > 0.9:
            j = i
            while j < len(x) and x[j] > 0.9:
                j += 1
            if j - i >= 4 and x[i:j].mean() > 1.2:
                out[i:j] = True
            i = j
        else:
            i += 1
    return pd.Series(out, index=z.index)


win = lambda s: s[((s.index.month == 6) & (s.index.day >= 10)) | s.index.month.isin([7, 8]) |
                  ((s.index.month == 9) & (s.index.day <= 15))]
ACT = pd.concat([spells(win(g.cmz_z), 1) for _, g in D.groupby(D.index.year)])
BRK = pd.concat([spells(win(g.cmz_z), -1) for _, g in D.groupby(D.index.year)])
days = ACT.index

rows = []
for f in sorted(glob.glob(os.path.join(PROJ, "data", "official_indices", "miso_iitm", "*_pc_MJJASO.dat"))):
    a = np.loadtxt(f)
    d = pd.Timestamp(int(os.path.basename(f)[:4]), 5, 1) + pd.to_timedelta(a[:, 0] - 1, "D")
    rows.append(pd.DataFrame({"i1": a[:, 1], "i2": a[:, 2], "ph": a[:, 3]}, index=d))
MI = pd.concat(rows).asfreq("D")
import re                                                          # BSISO1: the package loader (-999.9 removed)
SP7 = os.path.join(PROJ, "sem_project_7")
os.environ["BSISO_DATA"] = os.path.join(PROJ, "data"); os.environ["FAST"] = "1"
src = open(os.path.join(SP7, "BSISO_daily_ERE_Prediction", "BSISO_daily_ERE_Prediction.py"), encoding="utf-8").read()
cells = {blk.split(" --")[0].strip(): blk for blk in re.split(r"^# %% ", src, flags=re.M)[1:] if not blk.startswith("MD")}
ns = {"__name__": "__main__", "display": lambda *a: None}
cwd = os.getcwd(); os.chdir(os.environ.get("TEMP", "."))
exec(compile("# " + cells["CELL 1"], "c1", "exec"), ns)
exec(compile("# " + cells["CELL 2"], "c2", "exec"), ns)
os.chdir(cwd)
b = ns["b"]
BSI = pd.DataFrame({"i1": b["p1"], "i2": b["p2"]})
ang = np.arctan2(BSI.i1, -BSI.i2)
BSI["ph"] = np.floor(np.degrees(ang) % 360 / 45) + 1
agree = (BSI.ph.reindex(D.index) == D.bsiso1_phase).mean()             # must match the validated phases (India 4-6)
print(f"BSISO1 phase agreement with spell_drivers_daily.csv: {agree:.3f}")
IDX = {"MISO": MI, "BSISO1": BSI}
PH = {"MISO": ([2, 3, 4, 5], [6, 7, 8]), "BSISO1": ([4, 5, 6], [8, 1, 2])}


def at(nm, k):
    v = IDX[nm][["i1", "i2"]].reindex(days - pd.Timedelta(days=k)).values
    return np.column_stack([v, np.hypot(v[:, 0], v[:, 1])])


LAGS = list(range(-10, 16))
ok = np.all([np.isfinite(at(nm, k)).all(1) for nm in IDX for k in LAGS], axis=0)
yrs = days.year.values

# 1) Dey's same-day statistic
dey = []
for nm in IDX:
    V = IDX[nm].reindex(days)
    amp = np.hypot(V.i1, V.i2).values
    for lab, S, phs in (("active", ACT, PH[nm][0]), ("break", BRK, PH[nm][1])):
        for k in (0, 5, 10):
            Vk = IDX[nm].reindex(days - pd.Timedelta(days=k))
            ampk = np.hypot(Vk.i1, Vk.i2).values
            m = S.values & ok & (ampk > 1)
            dey.append({"index": nm, "spell": lab, "index taken (days before)": k, "spell days (amp>1)": int(m.sum()),
                        "share in expected phases": np.isin(Vk.ph.values[m], phs).mean(),
                        "share of all strong days in those phases": np.isin(Vk.ph.values[ok & (ampk > 1)], phs).mean()})
DEY = pd.DataFrame(dey)


def loyo(y, X):
    p, c = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for yr in np.unique(yrs[ok]):
        te, tr = ok & (yrs == yr), ok & (yrs != yr)
        m = LogisticRegression(C=1.0).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
        c[te] = y[tr].mean()
    return p, c


rows = []
for lab, S in (("active", ACT), ("break", BRK)):
    y = S.values.astype(int)
    for nm in IDX:
        for k in LAGS:
            X = at(nm, k)
            p, c = loyo(y, X)
            g = ok
            pin = LogisticRegression(C=1.0).fit(X[g], y[g]).predict_proba(X[g])[:, 1]   # upper bound (fit on all years)
            rows.append({"spell": lab, "index": nm, "lag (+ = before)": k, "AUC": roc_auc_score(y[g], p[g]),
                         "AUC in-sample": roc_auc_score(y[g], pin),
                         "Brier skill": 1 - np.mean((p[g] - y[g]) ** 2) / np.mean((c[g] - y[g]) ** 2)})
T = pd.DataFrame(rows)
DEY.to_csv(os.path.join(OUTT, "dey_same_day_phase_share.csv"), index=False)
T.to_csv(os.path.join(OUTT, "spell_lead_lag.csv"), index=False)
pd.set_option("display.width", 250)
print(f"days {int(ok.sum())}, active days {int(ACT[ok].sum())}, break days {int(BRK[ok].sum())}, "
      f"active spells {int((ACT.astype(int).diff() == 1).sum())}, break spells {int((BRK.astype(int).diff() == 1).sum())}")
print(DEY.round(3).to_string(index=False))
for col in ("AUC", "AUC in-sample", "Brier skill"):
    print("\n", col)
    print(T.pivot_table(index="lag (+ = before)", columns=["spell", "index"], values=col).round(3).to_string())
