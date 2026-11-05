# Simple MISO vs BSISO stat tables (two indices, same network, same days; a few key leads). Reads existing tables only.
# Output: sem_project_7/MISO_vs_BSISO_results/simple/MISO_vs_BSISO_simple_tables.md
import os
import pandas as pd

SP7 = r"C:\Users\divya\OneDrive\Desktop\IIT- Bombay Summer 26\Moaap\Regional_claude\sem_project_7"
I = os.path.join(SP7, "other", "intermediate")
OUT = os.path.join(SP7, "MISO_vs_BSISO_results", "simple")
rd = lambda p: pd.read_csv(os.path.join(I, p))
f3 = lambda v: f"{v:.3f}"
rg = lambda r, c="RPSS difference": f"{r[c]:+.3f} [{r['90% low']:+.3f}, {r['90% high']:+.3f}]"


def table(head, rows):
    return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in rows)


S = ["""# MISO vs BSISO — simple stat tables

- **BSISO forecast** and **MISO forecast** use the same transformer, the same inputs window and the same issue days.
- MISO = official IITM index; it only exists for May–Oct 1998–2020, so these tests cover JJAS 2000–2020 (2,562 days).
- **RPSS:** 0 = normal chance; higher is better.
- **AUC:** 0.5 = coin flip.
- **corr:** correlation of the forecast ERE chance with the ERE days.
- **ACC:** correlation of the forecast index with the observed index.
- **[ ]** = 90% range from year resamples.
- Figures S1–S6 are in this folder.
"""]

# 1 index forecast
IX, CR = rd("bsiso_forecast_tests/R8/R8_index_scores.csv"), rd("bsiso_forecast_tests/R8/R8_index_crossings.csv")
rows = []
for m, lab in (("BSISO T", "BSISO forecast"), ("MISO T", "MISO forecast"), ("persistence BSISO", "BSISO persistence"),
               ("persistence MISO", "MISO persistence")):
    d = IX[IX.model == m].set_index("lead").correlation
    rows.append([lab] + [f3(d[L]) for L in (5, 10, 15, 20)] + [f"{CR[CR.model == m]['days to corr 0.5'].iloc[0]:.1f}"])
S.append("\n## 1. How well each index forecasts itself (ACC)   — figure S1\n\n" +
         table(["", "5 d", "10 d", "15 d", "20 d", "days to ACC 0.5"], rows))
A = rd("bsiso_forecast_tests/R9a/R9a_crossings.csv").set_index("model")
S.append(f"\nAnirudh-style encoder-decoder on our OLR-MISO: ACC 0.3 at **{A.loc['enc-dec AR (Anirudh), OLR-MISO', 'bivariate corr: days to 0.3']:.1f} d**. "
         f"This matches the paper's ~18 d. A simple ridge model on the same index reaches 0.3 at "
         f"{A.loc['ridge, OLR', 'bivariate corr: days to 0.3']:.1f} d.\n")

# 2 daily ERE
RS = rd("bsiso_forecast_tests/R8/R8_rain_scores.csv")
LD = (1, 3, 5, 8, 10, 14)
g = lambda m, c: RS[RS.model == m].set_index("lead")[c]
rows = []
for c, nm in (("RPSS", "RPSS"), ("AUC", "AUC"), ("corr", "corr")):
    for m, lab in (("BSISO T", "BSISO forecast"), ("MISO T", "MISO forecast")):
        rows.append([nm, lab] + [f3(g(m, c)[L]) for L in LD])
rows.append(["corr", "normal chance"] + [f3(g("BSISO T", "corr, normal chance")[L]) for L in LD])
S.append("\n## 2. Daily ERE forecast   — figure S2\n\n" + table(["score", "", *[f"{L} d" for L in LD]], rows))
CM = rd("bsiso_forecast_tests/R8/R8_rain_comparisons.csv")
cm = CM[(CM.model == "MISO T") & (CM.vs == "BSISO T")].set_index("lead")
S.append("\n**MISO minus BSISO, daily RPSS**\n\n" + table([f"{L} d" for L in LD if L in cm.index],
                                                        [[rg(cm.loc[L]) for L in LD if L in cm.index]]))
S.append("\nMISO is a little better at 1–3 d, but not significantly. BSISO is clearly better at 8–10 d (range below 0). At 5 d and 14 d the difference is not significant.\n")

# 3 weekly
WS, WC = rd("R8_miso/compare/weekly_scores.csv"), rd("R8_miso/compare/weekly_comparisons.csv")
rows = []
for w, wl in (("week 1 (days 1-7)", "week 1"), ("week 2 (days 8-14)", "week 2")):
    for m, lab in (("BSISO T", "BSISO forecast"), ("MISO T", "MISO forecast")):
        r = WS[(WS.window == w) & (WS.model == m)].iloc[0]
        rows.append([wl, lab, f3(r["RPSS (0/1/2+)"]), f3(r["corr count"]), f3(r["AUC any"])])
    c = WC[(WC.window == w) & (WC.model == "MISO T") & (WC.vs == "BSISO T")].iloc[0]
    rows.append([wl, "**MISO minus BSISO**", rg(c), "", ""])
S.append("\n## 3. Weekly ERE forecast (ERE days 0 / 1 / 2+)   — figure S3\n\n" +
         table(["window", "", "RPSS", "corr", "AUC"], rows))

# 4 timing
LL = rd("R8_miso/ceiling/lead_lag_minus14_to_10.csv").set_index("lag")
rows = []
for col, lab in (("BSISO1", "BSISO"), ("MISO", "MISO")):
    k = LL[col].idxmax()
    rows.append([lab, f3(LL[col][5]), f3(LL[col][0]), f3(LL[col][-5]),
                 f"{abs(k)} d {'before' if k > 0 else 'after'} ({LL[col][k]:.3f})"])
S.append("\n## 4. When does the index 'know' about an ERE? (observed index, no forecast)   — figure S4\n\n" +
         table(["", "RPSS, 5 d before", "RPSS, on the day", "RPSS, 5 d after", "best timing"], rows))
S.append("\nMISO is built from the past 15 days of rain, so it matches EREs best after they have happened.\n")

# 5 spells
SH = rd("R8_miso/spell_check/dey_same_day_phase_share.csv")
SL = rd("R8_miso/spell_check/spell_lead_lag.csv")
rows = []
for sp in ("active", "break"):
    for ix, lab in (("BSISO1", "BSISO"), ("MISO", "MISO")):
        h = SH[(SH["index"] == ix) & (SH.spell == sp) & (SH["index taken (days before)"] == 0)].iloc[0]
        a = SL[(SL["index"] == ix) & (SL.spell == sp)].set_index("lag (+ = before)")
        rows.append([sp, lab, f"{h['share in expected phases']:.0%} (no-link {h['share of all strong days in those phases']:.0%})",
                     f"{a['AUC'][0]:.2f} / {a['AUC in-sample'][0]:.2f}", f"{a['AUC'][5]:.2f} / {a['AUC in-sample'][5]:.2f}"])
S.append("\n## 5. Active / break spells (Dey et al. rule)   — figure S5\n\n" +
         table(["spell", "", "same day: share in expected phases", "AUC on the day", "AUC 5 d before"], rows))
S.append("\n- **AUC** values: out-of-sample / in-sample. Out-of-sample can fall below 0.5 when there is almost no signal.\n"
         "- Expected phases: MISO active 2–5, break 6–8; BSISO active 4–6, break 8, 1, 2.\n"
         "- MISO reproduces Dey's same-day result.\n"
         "- 5 days ahead, MISO's advantage for active spells is gone. For break spells MISO stays slightly ahead (breaks follow\n"
         "  an active spell that MISO has already recorded).\n")

# 6 best product
WE = rd("R9e_regional_on_best/weekly_scores.csv")
gw = lambda v, w: WE[(WE.variant == v) & (WE.window == w)].iloc[0]
b1, b2 = gw("product: old best", "week 1"), gw("product: v2 (pre-registered)", "week 2")
rows = [["BSISO only", f3(gw("BSISO only", "week 1").RPSS), f3(gw("BSISO only", "week 2").RPSS)],
        ["BSISO + today's IMERG rain", f3(b1.RPSS), f3(gw("product: old best", "week 2").RPSS)],
        ["**best product** (+ BSISO2 in week 2)", f3(b1.RPSS), f"{b2.RPSS:.3f} ({rg(b2, 'gain vs old best product')})"]]
S.append("\n## 6. The current best BSISO forecast (JJAS 2000–2024)   — figure S6\n\n" +
         table(["", "week 1 RPSS", "week 2 RPSS"], rows))
S.append("""
## Bottom line

| | MISO | BSISO |
|---|---|---|
| Forecasting the index itself | **better** (13 d vs 6 d to ACC 0.5) | |
| Describing spells and EREs on the same day | **better** | |
| Warning of EREs ahead (5–14 d) | | **better** |
| Week 1 / week 2 ERE forecast | worse | **better** |

MISO is a description of the monsoon; BSISO is a warning.
""")
open(os.path.join(OUT, "MISO_vs_BSISO_simple_tables.md"), "w", encoding="utf-8").write("".join(S))
print("".join(S))
