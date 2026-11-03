# BSISO ERE prediction with the monsoon rain band

This folder continues `BSISO-daily-ERE-Prediction/`.

That folder forecast the daily chance of a large extreme-rain event (ERE) over central India from the BSISO alone. This one tests whether the **monsoon rain band** observed on the forecast day makes those forecasts better.

**The forecast target:**
- For every summer day (JJAS), the chance of no ERE, or of a small (16-39 cells), medium (40-90) or large (91+ cells) ERE over central India.
- The forecast is made 1-14 days ahead.
- Weekly view: the chance of 0, 1 or 2+ ERE days in days 1-7 (week 1) and days 8-14 (week 2).

## What was tried

| Test | What | Where it ran |
|---|---|---|
| R7a | Rain band in the **rain step**: rain over 70-95E in 1-degree latitude bins (IMERG), central-India rain today, "ERE today", and an OLR version of the band | local CPU |
| R7b | Rain band (OLR, 60 days) as an extra input to the **BSISO transformer** | Kaggle GPU |
| R7c | Best of both as one product: transformer BSISO forecast + IMERG rain inputs for leads 1-10 d, BSISO only for 11-14 d | local CPU |
| R7d | Fair comparison of normal chance, ridge, lasso, the BSISO-only transformer and the best model, **all on the same days** | local CPU |

Everything is out of sample:
- **Rain step:** each summer is forecast by a model fitted on the other 24 summers (2000-2024). The band patterns are refitted every time.
- **BSISO forecaster:** trained in 5 year blocks.
- **90% ranges:** come from resampling whole years 2,000 times.

## Main table (R7d, same days for every model)

How to read the scores:
- **RPSS:** 0 = no better than the normal chance for that date; 1 = perfect.
- **Correlation:** between the forecast chance and the ERE days that happened.
- **AUC:** the chance that an ERE day got a higher forecast than a non-ERE day. 0.5 = guessing.
- **Normal chance:** the "null" forecast, which only knows the time of year.

**Daily:**

| Score | Lead | Normal chance | Ridge | Lasso | Transformer, BSISO only | **Best: transformer + rain band** |
|---|---|---|---|---|---|---|
| RPSS | 1 d | 0 | 0.035 | 0.035 | 0.035 | **0.171** |
| | 5 d | 0 | 0.035 | 0.035 | 0.035 | **0.043** |
| | 8 d | 0 | 0.032 | 0.032 | 0.044 | **0.047** |
| | 10 d | 0 | 0.026 | 0.025 | 0.038 | **0.042** |
| | 14 d | 0 | 0.009 | 0.010 | 0.023 | **0.023** |
| Correlation | 1 d | 0.12 | 0.24 | 0.24 | 0.24 | **0.44** |
| | 5 d | 0.11 | 0.23 | 0.23 | 0.23 | **0.26** |
| | 8 d | 0.09 | 0.22 | 0.22 | 0.25 | **0.26** |
| | 10 d | 0.09 | 0.20 | 0.20 | 0.23 | **0.24** |
| | 14 d | 0.08 | 0.14 | 0.14 | 0.18 | **0.18** |
| AUC | 1 d | 0.60 | 0.69 | 0.69 | 0.69 | **0.81** |
| | 5 d | 0.58 | 0.68 | 0.68 | 0.68 | **0.70** |
| | 8 d | 0.57 | 0.66 | 0.66 | 0.69 | **0.70** |
| | 10 d | 0.57 | 0.65 | 0.65 | 0.67 | **0.68** |
| | 14 d | 0.56 | 0.61 | 0.61 | 0.63 | **0.63** |

**Weekly:**

| Score | Window | Normal chance | Ridge | Lasso | Transformer, BSISO only | **Best** |
|---|---|---|---|---|---|---|
| RPSS | week 1 (days 1-7) | 0 | 0.126 | 0.126 | 0.136 | **0.198** |
| | week 2 (days 8-14) | 0 | 0.055 | 0.056 | 0.091 | **0.100** |
| Correlation | week 1 | 0.20 | 0.42 | 0.42 | 0.43 | **0.50** |
| | week 2 | 0.15 | 0.32 | 0.32 | 0.39 | **0.40** |
| AUC | week 1 | 0.62 | 0.75 | 0.75 | 0.75 | **0.78** |
| | week 2 | 0.60 | 0.68 | 0.68 | 0.70 | **0.71** |

**BSISO forecast itself (ACC, bivariate correlation with the true BSISO; 0.5 = the usual "useful" line):**

| Lead | Normal chance | Ridge | Lasso | Transformer |
|---|---|---|---|---|
| 1 d | 0 | 0.97 | 0.97 | 0.94 |
| 5 d | 0 | 0.60 | 0.60 | 0.62 |
| 7 d | 0 | 0.47 | 0.47 | 0.51 |
| 10 d | 0 | 0.37 | 0.37 | 0.44 |
| ACC ≥ 0.5 until | – | ~6.5 d | ~6.5 d | ~7.4 d |

Notes:
- **Leads 1-5:** all BSISO models give the same rain scores. At those leads the rain step uses the *observed* BSISO from 5 days before the target, so no forecast is involved.
- **Ridge and lasso** are practically the same model.
- **Why the transformer wins at 6-14 d:** it keeps the BSISO's strength (amplitude kept at 9 d: 0.74 vs 0.38 for ridge), and the rain step needs that.
- **The best model switches to BSISO only after 10 d**, because the rain inputs slightly overfit there.
- **These scores are on a slightly smaller set of days than `BSISO-daily-ERE-Prediction/`:** the issue day and the day 3 days before it must have band data. That is why, for example, the 8-d correlation is 0.25 here instead of 0.27.

## R7a: the rain band in the rain step

**Daily RPSS, BSISO-only transformer vs + all IMERG inputs:**

| Lead | 1 d | 2 d | 3 d | 5 d | 8 d | 10 d | 12 d | 14 d |
|---|---|---|---|---|---|---|---|---|
| BSISO only | 0.035 | 0.035 | 0.035 | 0.035 | 0.044 | 0.038 | 0.029 | 0.024 |
| + all IMERG inputs | **0.171** | **0.082** | **0.057** | 0.043 | 0.046 | 0.042 | 0.025 | 0.019 |
| gain [90% range] | +0.137 [+0.108, +0.166] | +0.047 [+0.028, +0.066] | +0.022 [+0.005, +0.039] | +0.008 [-0.003, +0.019] | +0.002 | +0.004 [-0.005, +0.012] | -0.005 | -0.005 [-0.010, -0.001] |

- **Most of the day-1 skill comes from today's local rain and "ERE today"** (weather memory). Where the band sits helps to about 5 days.
- **The band's movement (band 3 days earlier) adds nothing.**
- **The OLR band is a weaker copy of the IMERG band.**

## R7b: the rain band inside the BSISO transformer

| Model | BSISO ACC 5 d | ACC 9 d | Amplitude kept 9 d | Rain RPSS 8 d | Rain RPSS 14 d |
|---|---|---|---|---|---|
| Ridge, all inputs | 0.633 | 0.458 | 0.38 | 0.0356 | 0.0167 |
| Ridge + OLR band | 0.581 | 0.400 | 0.44 | 0.0359 | 0.0115 |
| **Transformer (best)** | **0.620** | **0.464** | **0.74** | **0.0445** | **0.0222** |
| Transformer + OLR band | 0.588 | 0.434 | 0.71 | 0.0387 | 0.0136 |

**Adding the band made the BSISO forecast significantly worse:**
- ACC fell by 0.03 at every lead.
- Rain RPSS fell by 0.006-0.009.

A ridge check shows why. The band compressed to 3 patterns adds nothing (0.631 vs 0.633), and the full band only adds noise. The BSISO index is itself built from OLR, so an OLR band carries no new BSISO information.

## R7c: the final product (week 1 and week 2)

| Window | RPSS, BSISO only | RPSS, with rain band | Gain [90% range] |
|---|---|---|---|
| week 1 | 0.137 | **0.198** | +0.061 [+0.036, +0.087] |
| week 2 | 0.092 | **0.101** | +0.009 [+0.004, +0.015] |

## What it means
1. **The rain band belongs in the rain step, not in the BSISO forecaster.** It gives a large gain for days 1-5 and week 1, and a small but real gain in week 2.
2. **Beyond about a week, the only skill still comes from the BSISO forecast.** That forecast drops below ACC 0.5 after ~7.4 days, but it is not useless after that: it still has ACC 0.44 at 10 d, and it is what lifts week 2 from 0.055 (ridge) to 0.091-0.100.
3. **To go further, the forecast needs new information, not a bigger network:**
   - moisture, sea temperature and upper-level winds as maps (Maeda & Satoh 2025 reach BSISO correlation 0.6 at ~29 days this way);
   - a dynamical model forecast (ECMWF S2S: ~20 days);
   - a longer rain record (IMD gridded rain).

## Files
- `bsiso_forecast_tests_r7.ipynb`: the executed Kaggle notebook for R7b (transformer, with and without the band).
- `code/`:
  - `R7a_rainband_rain_step.py`: rain step with the band (also R7c, via `R7_TAG=R7c` and `R7_FC=<forecast parquet>`).
  - `R7b_band_ridge_check.py`: the ridge check.
  - `R7d_model_compare_same_days.py` and `R7d_model_compare_figure.py`: the main table and fig1.
  - `kaggle_BSISO_Tests_R7.py`: source of the Kaggle notebook.
  - `BSISO_daily_ERE_Prediction.py`: the base pipeline the R7 scripts reuse (data, ERE target, rain step).
- `results/`:
  - score tables (`*_scores*.csv`, `*comparisons.csv`);
  - the forecast product (`daily_ere_class_probabilities_with_rain.csv`, `weekly_ere_count_probabilities_with_rain.csv`).
- `figures/`:
  - fig1: model comparison;
  - fig2: R7a;
  - fig3: R7c;
  - fig4-5: R7b;
  - fig6: the rain band in 2019.

The scripts use local paths and the project's data folder (APCC BSISO index, IMERG 0.25° JJAS 2000-2024, NOAA CPC OLR). The data is not stored in the repo.
