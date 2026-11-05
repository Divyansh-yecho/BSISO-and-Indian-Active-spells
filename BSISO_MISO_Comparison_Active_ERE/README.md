# BSISO vs MISO: active/break spells and extreme rain over central India

Two indices of the monsoon's intraseasonal swings, compared head to head:

- **BSISO1** (Lee et al. 2013, APCC) is built from cloud (OLR) and low-level wind over a wide area: 10°S-40°N, 40-160°E.
- **MISO** (Suhas et al. 2013, the official IITM index) is built from the last 15 days of rain over the Indian region.

MISO is the index used by Dey et al. (2022) and Anirudh et al. (2025), and it looks very strong in both papers. Here we
ask whether it is also better for **forecasting** extreme rain.

**Extreme-rain event (ERE)**, at 0.25° as in the rest of this repo:
- at least one connected cluster of 16 or more IMERG cells, each above its own 99th percentile;
- the cluster is centred in central India;
- size classes: 16-39, 40-90 and 91+ cells.

**Spells:** central-India rain at least +1 sd (active) or at most -1 sd (break) for 3 or more days.

**Days:** June-September 2000-2020, because the MISO files stop in 2020.

## Scores used

| score | what it measures | no skill | perfect |
|---|---|---|---|
| **ACC** | correlation between the forecast index and the observed index | 0 | 1 |
| **RPSS** | how much better the ERE chances are than the normal seasonal chance | 0 | 1 |
| **AUC** | how often an ERE day gets a higher chance than a no-ERE day | 0.5 | 1 |
| **corr** | correlation between the forecast ERE chance and the 0/1 ERE days | normal chance: about 0.1 | 1 |

[ ] = 90% range from year resamples.

## 1. How far ahead each index forecasts itself (ACC)

Same transformer, same inputs window, same days.

| | 5 d | 10 d | 15 d | 20 d | days to ACC 0.5 |
|---|---|---|---|---|---|
| BSISO forecast | 0.558 | 0.400 | 0.335 | 0.292 | 6.2 |
| **MISO forecast** | **0.870** | **0.666** | **0.433** | **0.324** | **13.4** |
| BSISO persistence (today's value) | 0.504 | 0.096 | -0.119 | -0.222 | 5.0 |
| MISO persistence | 0.560 | -0.127 | -0.401 | -0.348 | 5.4 |

MISO forecasts itself much further ahead. An Anirudh-style encoder-decoder on an OLR version of MISO reaches ACC 0.3
at 19.0 days, matching the paper's ~18 days.

## 2. Forecasting daily EREs from each index forecast

| score | | 1 d | 3 d | 5 d | 8 d | 10 d | 14 d |
|---|---|---|---|---|---|---|---|
| RPSS | BSISO | 0.034 | 0.034 | 0.034 | **0.035** | **0.026** | **0.015** |
| RPSS | MISO | **0.046** | **0.040** | 0.030 | 0.012 | 0.006 | 0.003 |
| AUC | BSISO | 0.681 | 0.678 | **0.673** | **0.659** | **0.630** | **0.590** |
| AUC | MISO | **0.693** | 0.676 | 0.647 | 0.604 | 0.584 | 0.570 |
| corr | BSISO | 0.228 | 0.225 | **0.220** | **0.219** | **0.186** | **0.139** |
| corr | MISO | **0.258** | **0.239** | 0.204 | 0.143 | 0.113 | 0.090 |
| corr | normal chance | 0.120 | 0.115 | 0.106 | 0.092 | 0.080 | 0.061 |

**MISO minus BSISO, daily RPSS:**

| 1 d | 3 d | 5 d | 8 d | 10 d | 14 d |
|---|---|---|---|---|---|
| +0.012 [-0.005, +0.028] | +0.006 [-0.010, +0.022] | -0.004 [-0.021, +0.012] | **-0.022 [-0.033, -0.012]** | **-0.021 [-0.031, -0.010]** | -0.012 [-0.026, +0.001] |

**Weekly (number of ERE days: 0 / 1 / 2+):**

| | RPSS | corr | AUC |
|---|---|---|---|
| week 1, BSISO | **0.105** | **0.407** | **0.736** |
| week 1, MISO | 0.062 | 0.380 | 0.700 |
| week 2, BSISO | **0.036** | **0.319** | **0.667** |
| week 2, MISO | -0.012 | 0.176 | 0.604 |

MISO minus BSISO: week 1 **-0.043 [-0.069, -0.016]**, week 2 **-0.048 [-0.081, -0.016]**.

## 3. When does each index "know" about an ERE? (observed indices, no forecast)

| | RPSS, 5 d before | RPSS, on the day | RPSS, 5 d after | best timing |
|---|---|---|---|---|
| BSISO | **0.034** | 0.024 | 0.013 | **4 d before** (0.036) |
| MISO | 0.028 | **0.057** | **0.074** | **6 d after** (0.074) |

MISO is a running summary of the past 15 days of rain, so it matches EREs best **after** they happen. BSISO is a
precursor: its signal peaks 4-5 days **before** the ERE.

## 4. How spells cluster in the phase diagrams

**Dey et al. same-day test:** share of strong-index spell days that fall in the expected phases, against the share
expected with no link.

| spell | | share in expected phases | with no link | AUC on the day (out of sample / in sample) | AUC 5 d before |
|---|---|---|---|---|---|
| active | BSISO (phases 4-6) | 64% | 40% | 0.61 / 0.70 | 0.61 / 0.67 |
| active | **MISO (phases 2-5)** | **87%** | 53% | 0.65 / 0.74 | 0.51 / 0.65 |
| break | BSISO (phases 8, 1, 2) | 44% | 34% | 0.62 / 0.67 | 0.61 / 0.67 |
| break | **MISO (phases 6-8)** | **72%** | 33% | 0.73 / 0.77 | 0.63 / 0.70 |

**Where spell days sit in the MISO phases** (share of all spell days):

| MISO phase | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| active days | 6% | 18% | **34%** | 20% | 16% | 5% | 2% | 1% |
| break days | 9% | 1% | 1% | 5% | **23%** | **28%** | **24%** | 10% |
| all strong-MISO days | 14% | 14% | 14% | 13% | 13% | 10% | 11% | 11% |

On the same day, MISO separates active and break spells very cleanly; this reproduces Dey et al. Five days ahead, its
edge for active spells is gone.

## 5. How EREs cluster by phase (same day; ratio to the normal rate)

| index | ERE class | days | ph1 | ph2 | ph3 | ph4 | ph5 | ph6 | ph7 | ph8 |
|---|---|---|---|---|---|---|---|---|---|---|
| MISO | any ERE | 397 | 0.25 | 0.82 | 1.56 | **1.93** | 1.75 | 0.59 | 0.55 | 0.38 |
| MISO | 16-39 cells | 229 | 0.39 | 0.82 | 1.55 | 1.62 | 1.24 | 0.70 | 0.47 | 0.48 |
| MISO | 40-90 cells | 133 | 0.08 | 0.79 | 1.33 | **2.27** | 2.23 | 0.44 | 0.82 | 0.31 |
| MISO | 91+ cells | 35 | 0.00 | 0.90 | 2.54 | 2.65 | **3.25** | 0.42 | 0.00 | 0.00 |
| BSISO | any ERE | 397 | 0.46 | 0.54 | 0.42 | 0.72 | **1.76** | 1.72 | 1.32 | 0.96 |
| BSISO | 16-39 cells | 229 | 0.40 | 0.52 | 0.50 | 0.44 | **1.61** | 1.41 | 1.22 | 0.92 |
| BSISO | 40-90 cells | 133 | 0.69 | 0.73 | 0.29 | 1.18 | 1.52 | **2.03** | 1.64 | 0.99 |
| BSISO | 91+ cells | 35 | 0.00 | 0.00 | 0.36 | 0.82 | **3.60** | 2.57 | 0.69 | 1.13 |

All patterns are significant (season re-dating test, p ≤ 0.02). The largest EREs are the most phase-locked in both
indices: about 3 times the normal rate in MISO phases 3-5 and in BSISO phase 5.

## Bottom line

| | MISO | BSISO |
|---|---|---|
| forecasting the index itself | **better** (13 d vs 6 d to ACC 0.5) | |
| sorting spells and EREs on the same day | **better** | |
| warning of EREs 5-14 days ahead | | **better** |
| week 1 / week 2 ERE forecast | worse | **better** |

**MISO describes the monsoon; BSISO warns about it.** Both papers' results are real and are reproduced here:
- Anirudh: MISO forecasts itself far ahead;
- Dey: MISO sorts spells on the same day.

But a summary of rain that has already fallen cannot warn about rain to come, so BSISO is the index to use for
forecasting EREs.

## Files

- `bsiso_miso_comparison.ipynb`: the analysis, run locally with outputs.
  - sections 1-3 recompute the timing, spell and phase-diagram results from the data;
  - section 4 plots the forecast comparison from `results/`.
- `figures/`:
  - `fig1`-`fig5`: phase diagrams of spell and ERE days, index paths around spell onset, ERE rates by phase and
    class, and timing;
  - `S1`-`S5`: simple comparison figures for ACC, daily and weekly RPSS, timing and spell AUC.
- `results/`: the score tables behind every number above.
- `code/`:
  - the scripts used, including `kaggle_BSISO_Tests_R8.py`, which trained both forecasters on a Kaggle GPU;
  - the BSISO-Rain and BSISO-daily-ERE-Prediction scripts that the notebook reuses.

Data are not stored here: the APCC BSISO index, the IITM MISO index, and IMERG daily rain.
