# R8 follow-up: observed index taken from 14 days AFTER to 10 days BEFORE the target day (negative lag = after).
# Built from R8_local_index_ceiling.py (same days, same rain step). Output: R8_miso/ceiling/lead_lag_minus14_to_10.csv
# R8 (local, CPU): the rain-step CEILING of each index. Observed (perfectly known) BSISO1 vs observed MISO (IITM,
# Suhas et al. 2013) at lags 0-10 days before the target day -> daily 4-class ERE chances in central India.
# Same rain step as the final package (cumulative logistic + smoothed normal chance, leave-one-year-out).
# Same days for every input: JJAS target days 2000-2020 (the official MISO files stop in Oct 2020).
# Pre-registered rule: each index's lag for the forecast round = the lag with the highest observed RPSS here.
import os, re, glob
import numpy as np, pandas as pd

PROJ = r"C:\Users\divya\OneDrive\Desktop\IIT- Bombay Summer 26\Moaap\Regional_claude"
SP7 = os.path.join(PROJ, "sem_project_7")
OUTT = os.path.join(SP7, "other", "intermediate", "R8_miso", "ceiling")
os.makedirs(OUTT, exist_ok=True)
os.environ["BSISO_DATA"] = os.path.join(PROJ, "data"); os.environ["FAST"] = "1"
src = open(os.path.join(SP7, "BSISO_daily_ERE_Prediction", "BSISO_daily_ERE_Prediction.py"), encoding="utf-8").read()
cells = {blk.split(" --")[0].strip(): blk for blk in re.split(r"^# %% ", src, flags=re.M)[1:] if not blk.startswith("MD")}
ns = {"__name__": "__main__", "display": lambda *a: None}
os.chdir(os.environ.get("TEMP", "."))
exec(compile("# " + cells["CELL 1"], "c1", "exec"), ns)
exec(compile("# " + cells["CELL 2"], "c2", "exec"), ns)         # data, ERE days (needs torch only for the device)
c7 =cells["CELL 7"].split("YC = [")[0]
c7 = c7.replace("FCI = {nm: pd.DataFrame(FC[nm].reshape(len(ISSUE), -1), index=ISSUE) for nm in MODELS}", "")
exec(compile("# " + c7, "c7", "exec"), ns)
b, times, years, kidx, ND, CLS4, OBS1 = (ns[k] for k in ("b", "times", "years", "kidx", "ND", "CLS4", "OBS1"))
ordinal_eval = ns["ordinal_eval"]


def load_miso():
    rows = []
    for f in sorted(glob.glob(os.path.join(PROJ, "data", "official_indices", "miso_iitm", "*_pc_MJJASO.dat"))):
        a = np.loadtxt(f)
        d = pd.Timestamp(int(os.path.basename(f)[:4]), 5, 1) + pd.to_timedelta(a[:, 0] - 1, "D")
        rows.append(pd.DataFrame({"m1": a[:, 1], "m2": a[:, 2]}, index=d))
    return pd.concat(rows).asfreq("D")


M = load_miso()
sel = (years <= 2020)
YC = [(CLS4 >= t).astype(float)[sel] for t in (1, 2, 3)]
yrs, pos, obs = years[sel], kidx[sel], OBS1[sel]


def at(df, cols, lag):
    v = df[cols].reindex(times - pd.Timedelta(days=lag)).values[sel]
    return np.column_stack([v, np.hypot(v[:, 0], v[:, 1])])



rpss = lambda r: 1 - r["rps_m"].sum() / r["rps_r"].sum()
rows=[]
LAGS=list(range(-14,11))
X={}
for lag in LAGS:
    X[lag]=(at(b,["p1","p2"],lag), at(M,["m1","m2"],lag))
ok=np.all([np.isfinite(np.column_stack(X[l])).all(1) for l in LAGS],axis=0)
for lag in LAGS:
    XB,XM=X[lag]
    for nm,Xb in (("BSISO1",XB),("MISO",XM)):
        r=ordinal_eval(YC,np.where(ok[:,None],Xb,np.nan),yrs,pos,ND,obs)
        rows.append({"index":nm,"lag":lag,"RPSS":rpss(r)})
T=pd.DataFrame(rows).pivot(index="lag",columns="index",values="RPSS")
print("days",int(ok.sum())); print(T.round(3).to_string())
T.to_csv(os.path.join(OUTT,"lead_lag_minus14_to_10.csv"))

