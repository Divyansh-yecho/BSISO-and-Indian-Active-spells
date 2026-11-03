# Same-days comparison: normal chance vs ridge / lasso / transformer, each BSISO-only and + all IMERG rain inputs.
# Reuses the R7a machinery (strict per-fold band patterns, LOYO cumulative logistic, same day set as R7c).
import os, re
import numpy as np, pandas as pd

SP7 = r"C:\Users\divya\OneDrive\Desktop\IIT- Bombay Summer 26\Moaap\Regional_claude\sem_project_7"
src = open(os.path.join(SP7, "other", "scripts", "R7a_rainband_rain_step.py"), encoding="utf-8").read()
head = src.split("def features(")[0]
old = head[head.index("F = pd.read_parquet("):head.index("exec(compile(\"# \" + cells[\"CELL 7\"]")]
new = '''FF = pd.concat([pd.read_parquet(os.path.join(SP7, "other", "intermediate", "bsiso_forecast_tests", "R7", "R7_forecasts_T_best.parquet")).assign(model="transformer"),
                pd.read_parquet(os.path.join(SP7, "BSISO_daily_ERE_Prediction", "results", "bsiso_forecasts.parquet")).query("model in ['ridge', 'lasso']")])
FCd = {}
for m in ("transformer", "ridge", "lasso"):
    F = FF[FF.model == m]
    P = np.full((len(ISSUE), H, 2), np.nan, dtype="float32")
    for L in range(1, H + 1):
        P[:, L - 1] = F[F.lead == L].set_index("issue")[["f1", "f2"]].reindex(ISSUE).values
    FCd[m] = P
ns.update(FC=FCd, MODELS=list(FCd))
'''
head = head.replace(old, new).replace('TAG = os.environ.get("R7_TAG", "R7a")', 'TAG = "model_compare"')
exec(compile(head, "R7a_head", "exec"))

MODELS = ["ridge", "lasso", "transformer"]
VAR = [f"{m}|{k}" for m in MODELS for k in ("bsiso", "rain")]


def features(L, yr, Bp):
    i0, i3 = shift_idx(L), shift_idx(L + 3)
    b0, b3 = take(Bp, i0), take(Bp, i3)
    cr = np.column_stack([take(CIR, i0), take(OBS1, i0)])
    ob = olr_patterns(yr, times - pd.Timedelta(days=L))
    base = {m: bsiso_T5(m, L) for m in MODELS}
    ok = np.isfinite(np.column_stack([*base.values(), b0, b3, cr, ob])).all(1)   # same days for every model
    V = {}
    for m in MODELS:
        V[f"{m}|bsiso"] = base[m]
        V[f"{m}|rain"] = np.column_stack([base[m], b0, b3, cr])
    return {k: np.where(ok[:, None], v, np.nan) for k, v in V.items()}


BP = {yr: band_patterns(yr) for yr in UY}
RES, CLIM = {}, {}
for L in range(1, 15):
    PM = {v: np.full((3, len(times)), np.nan) for v in VAR}
    PR = np.full((3, len(times)), np.nan)
    G = np.zeros(len(times), bool)
    for yr in UY:
        Fv = features(L, yr, BP[yr])
        te = years == yr
        good = np.isfinite(Fv[VAR[0]]).all(1)
        G |= te & good
        for c, y in enumerate(YC):
            cl = clim_logit(y, years, kidx, yr, ND)
            tr = (years != yr) & good
            PR[c, te & good] = fit_predict(cl[tr], y[tr], cl[te & good])
            for v in VAR:
                Xv = np.column_stack([cl, Fv[v]])
                PM[v][c, te & good] = fit_predict(Xv[tr], y[tr], Xv[te & good])
    PRm = np.minimum.accumulate(PR, 0)
    Yy = np.array(YC)
    rr = np.where(G, ((np.nan_to_num(PRm) - Yy) ** 2).sum(0), 0)
    CLIM[L] = {"corr": np.corrcoef(PRm[0][G], OBS1[G])[0, 1], "auc": roc_auc_score(OBS1[G], PRm[0][G]), "P": PRm, "G": G}
    for v in VAR:
        Pm = np.minimum.accumulate(PM[v], 0)
        rm = np.where(G, ((np.nan_to_num(Pm) - Yy) ** 2).sum(0), 0)
        RES[(v, L)] = {"rps_m": np.array([rm[years == y].sum() for y in UY]), "rps_r": np.array([rr[years == y].sum() for y in UY]),
                       "corr": np.corrcoef(Pm[0][G], OBS1[G])[0, 1], "auc": roc_auc_score(OBS1[G], Pm[0][G]), "P": Pm, "G": G}
    print(f"lead {L}: {G.sum()} days", flush=True)

rpss = lambda r: 1 - r["rps_m"].sum() / r["rps_r"].sum()
rows = [{"model": "normal chance", "lead": L, "RPSS": 0.0, "corr": CLIM[L]["corr"], "AUC": CLIM[L]["auc"]} for L in range(1, 15)]
rows += [{"model": v, "lead": L, "RPSS": rpss(RES[(v, L)]), "corr": RES[(v, L)]["corr"], "AUC": RES[(v, L)]["auc"]}
         for v in VAR for L in range(1, 15)]
D = pd.DataFrame(rows)

# weekly: rain inputs to 10 d, BSISO-only after (same rule as the R7c combined product)
J = np.arange(len(times) - 6); J = J[(times[J + 6] - times[J]).days == 6]
NW = np.array([OBS1[j:j + 7].sum() for j in J]); yW, kW = years[J], kidx[J]
YCW = [(NW >= 1).astype(float), (NW >= 2).astype(float)]
ordinal_eval = ns["ordinal_eval"]


def weekly(get_p, a, b_):
    S = np.sum([get_p(L)[J + (L - a)] for L in range(a, b_ + 1)], axis=0)
    r = ordinal_eval(YCW, np.log(S + 1e-3)[:, None], yW, kW, ND, (NW >= 1).astype(float))
    g = r["G"]
    r["corr_count"] = np.corrcoef(S[g], NW[g])[0, 1]
    r["clim_corr"] = np.corrcoef(r["Pr"][0][g], NW[g])[0, 1]
    r["clim_auc"] = roc_auc_score((NW[g] >= 1), r["Pr"][0][g])
    return r


dp = lambda v: (lambda L: np.where(RES[(v, L)]["G"], RES[(v, L)]["P"][0], np.nan))
wr = []
for wk, (a, b_) in {"week 1": (1, 7), "week 2": (8, 14)}.items():
    for m in MODELS:
        for k in ("bsiso", "rain"):
            g = dp(f"{m}|bsiso") if k == "bsiso" else (lambda L, m=m: dp(f"{m}|rain" if L <= 10 else f"{m}|bsiso")(L))
            r = weekly(g, a, b_)
            wr.append({"window": wk, "model": f"{m}|{k}", "RPSS": rpss(r), "corr": r["corr_count"], "AUC": r["auc"]})
    wr.append({"window": wk, "model": "normal chance", "RPSS": 0.0, "corr": r["clim_corr"], "AUC": r["clim_auc"]})
W = pd.DataFrame(wr)
OUT = os.path.join(SP7, "other", "intermediate", "R7_rain_band", "model_compare")
os.makedirs(OUT, exist_ok=True)
D.to_csv(os.path.join(OUT, "daily_scores_same_days.csv"), index=False)
W.to_csv(os.path.join(OUT, "weekly_scores_same_days.csv"), index=False)
pd.set_option("display.width", 250)
for c in ("RPSS", "corr", "AUC"):
    print(c); print(D.pivot(index="model", columns="lead", values=c).round(3).to_string())
print(W.round(3).to_string())
