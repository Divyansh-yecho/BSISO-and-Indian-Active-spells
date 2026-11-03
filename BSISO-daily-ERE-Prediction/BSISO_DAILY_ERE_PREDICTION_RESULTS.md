# Forecasting large extreme-rain days from the BSISO: what the numbers say

Companion to [`bsiso_daily_ere_prediction.ipynb`](bsiso_daily_ere_prediction.ipynb), a Kaggle notebook (GPU T4).
Training takes about 13 minutes and the full run about 25.
Season scored: June-September (JJAS), 2000-2024. Rain: IMERG at 0.25 degree.

The previous folder (`BSISO-ERE-Link/`) showed that the BSISO shifts the chance of a large extreme-rain event (ERE)
over central India, and that the signal is strongest about 5 days before the rain. This folder turns that into a
**forecast**:
1. forecast the BSISO with a small transformer;
2. turn the forecast into the daily chance of no ERE, or a small, medium or large ERE, 1-14 days ahead;
3. add up the daily chances into the number of ERE days in the coming week.

This is **one** of the approaches tried: the BSISO alone, with no weather maps and no rain inputs. The model is the
best of seven test rounds (about 30 variants). The notebook retrains it from scratch and re-scores every change
that was tested.

Every number below is printed by the notebook. The copy here was run with the 100 models trained on Kaggle (their saved forecasts), so it shows the same
numbers as the Kaggle run without retraining.

---

## 1. The result in one paragraph

From the BSISO alone, the daily chance of a large ERE over central India can be forecast with real but modest skill
for about two weeks. On the 10% of days the forecast rates most likely, a large ERE happens 34-35% of the time, more
than twice the normal 15%. On the 10% rated least likely it happens only 3-5% of the time. The probabilities are
reliable. The day-by-day correlation is only about 0.25, but that is close to the ceiling for this event: even a
perfectly known BSISO moves the daily chance only between about 4% and 28%. Counted per week, the same forecast
correlates 0.42 with the number of ERE days 8-14 days ahead.

The transformer is **not** a better BSISO forecaster than a simple ridge regression: its correlation is the same.
It does keep the oscillation's strength, which ridge loses, and that gives a small, significant gain in rain skill.

### What is supported, and what is not

| claim | key number | verdict |
|---|---|---|
| The forecast beats normal chance for two weeks | rain RPSS 0.044 (8 d), 0.037 (10 d), 0.023 (14 d); corr 0.27-0.215 vs 0.14 | **supported** |
| It picks out ERE-prone days | top 10% of days: 34-35% ERE vs 15% normal; bottom 10%: 3-5% | **strong** |
| The probabilities are reliable | forecast and observed frequencies on the diagonal (fig 3a) | **supported** |
| The transformer beats the old lasso model | +0.012 to +0.015 RPSS at 8-14 d, all significant | **strong** |
| The transformer beats ridge with the same inputs | +0.008 [+0.003, +0.013] at 8 d, +0.008 [+0.001, +0.015] at 12 d; 10 d and 14 d just touch zero | **supported, small** |
| The transformer forecasts the BSISO better than ridge | correlation 0.62 vs 0.63 at 5 d, 0.46 vs 0.46 at 9 d | **not supported** |
| It keeps the oscillation's strength | 74% of the true amplitude at 9 d vs 38% for ridge | **strong** |
| A weekly count is a clearer product than single days | week 2 corr 0.42 (normal 0.23); RPSS vs ridge +0.022 [+0.008, +0.036] | **strong** |
| The forecast beats "true BSISO 5 days before" | +0.008 at 8 d, but the range includes zero | **not significant** |
| It forecasts the biggest events (91+ cells) well | AUC 0.73 with the true BSISO, 0.64 at 8 d, 0.53 at 14 d | **only short-range** |
| A bigger or more complex network helps | larger, encoder-decoder, GRU, uncertainty output: none better | **not supported** |

---

## 2. Data and terms

| dataset (Kaggle) | file | contents |
|---|---|---|
| BSISO index | `BSISO.INDEX.NORM.LY.data` | daily BSISO1 (PC1, PC2) and BSISO2 (PC3, PC4), 1981-2026, APCC index (Lee et al. 2013) |
| IMERG rainfall | `imerg_india_025_jjas_2000_2024.nc` | daily rain, 0.25 degree, 5-40N, 60-100E, 3,050 JJAS days |
| India mask | `india_mask_025.npy` | India land cells on the IMERG grid |
| lasso forecasts | `lasso_bsiso_forecasts.parquet` | the older statistical BSISO forecasts (baseline) |
| test rounds | outputs of 7 test notebooks | only for section 6 (what mattered) |

The data files are not stored in this repository.

| term | meaning |
|---|---|
| **ERE day** | at least one group of >= 16 connected IMERG cells, each above its own wet-day 99th percentile, centred in central India (17-26.5N, 74-88.5E). 15.3% of JJAS days |
| **size class** | the largest ERE that day: none, 16-39, 40-90 or 91+ cells |
| **lead** | days from the day the forecast is made (issue day) to the day it is for (target day) |
| **RPSS** | skill of the 4-class forecast compared with the normal chance for that date (0 = no better, 1 = perfect) |
| **corr** | correlation of the forecast P(large ERE) with the yes/no ERE days |
| **AUC** | chance that a random ERE day gets a higher forecast probability than a random non-ERE day (0.5 = guessing) |
| **90% range** | from resampling the 25 years 2,000 times; a range that excludes zero is called significant |

---

## 3. The method

1. **BSISO forecast.**
   - Input: the last 60 days of BSISO1, BSISO2 and the time of year.
   - Model: a small transformer that cuts the 60 days into twelve 5-day pieces (after PatchTST), with a linear
     shortcut (after DLinear). About 121,000 parameters.
   - Output: BSISO1 for each of the next 10 days.
   - Training loss: half ordinary error, half error in the oscillation's strength (after Anirudh et al. 2025).
   - Testing: the years are split into 5 blocks (1981-90, 91-99, 2000-08, 09-17, 18-26). Each block is forecast
     by models trained on the other four, and 20 trained copies are averaged.
2. **Rain step.**
   - Input: the BSISO forecast for 5 days before the target day, plus the normal chance for that date.
   - Model: a cumulative logistic regression. Each summer is predicted by a regression fitted on the other 24.
   - Leads 1-5 use the BSISO already observed, so only leads 6-14 need a forecast.
3. **Weekly view.**
   - The daily chances are summed over days 1-7 or 8-14, giving the expected number of ERE days.
   - A second leave-one-year-out regression turns that into P(0, 1, 2+ ERE days).
4. **Baselines:**
   - ridge regression on the same inputs (the fair comparison);
   - ridge on BSISO1 only;
   - the older lasso forecasts;
   - "no forecast" (today's BSISO kept constant);
   - the true BSISO 5 days before.

---

## 4. How good is the BSISO forecast itself? (fig 1)

**Correlation with the true BSISO1** (0.5 is the usual "useful" level):

| lead (days) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| transformer | 0.941 | 0.863 | 0.775 | 0.692 | 0.620 | 0.558 | 0.514 | 0.482 | 0.460 | 0.442 |
| ridge, same inputs | 0.968 | 0.889 | 0.796 | 0.709 | 0.633 | 0.568 | 0.519 | 0.484 | 0.458 | 0.438 |
| lasso (older) | 0.968 | 0.882 | 0.779 | 0.684 | 0.602 | 0.531 | 0.473 | 0.431 | 0.396 | 0.367 |
| no forecast | 0.950 | 0.847 | 0.731 | 0.620 | 0.517 | 0.422 | 0.337 | 0.260 | 0.191 | 0.128 |

**Other scores:**

| lead (days) | 1 | 3 | 5 | 7 | 9 | 10 |
|---|---|---|---|---|---|---|
| amplitude kept, transformer | 0.95 | 0.86 | 0.80 | 0.76 | 0.74 | 0.74 |
| amplitude kept, ridge | 0.92 | 0.70 | 0.55 | 0.45 | 0.38 | 0.36 |
| phase right or one phase off, transformer | 1.00 | 0.93 | 0.82 | 0.73 | 0.70 | 0.68 |
| phase right or one phase off, ridge | 1.00 | 0.94 | 0.82 | 0.74 | 0.70 | 0.69 |

**Days until the correlation falls to 0.5:** transformer 7.4, ridge 7.5, lasso 6.5, no forecast 5.2.

**Transformer minus ridge, correlation:**
- 1 d: -0.028 [-0.030, -0.025]
- 5 d: -0.013 [-0.025, -0.002]
- 9-10 d: +0.002 to +0.004, within noise

So the transformer is worse at short leads and ties at long leads. Its only clear advantage is keeping about twice
the oscillation's strength. Ridge's forecasts shrink towards zero, the transformer's do not.

Dynamical models reach the 0.5 level at about 20 days (ECMWF; Lee et al. 2017, Jie et al. 2017). A CNN fed with
ERA5 weather maps reaches about 29 days (Maeda & Satoh 2025). The 4 numbers of the index are the limit here, not
the network.

---

## 5. Daily rain probabilities (figs 2-4)

| rain lead (days) | 1-5 (true BSISO) | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|
| RPSS, transformer | 0.035 | 0.040 | **0.044** | 0.037 | 0.028 | 0.023 |
| RPSS, ridge (same inputs) | 0.035 | 0.035 | 0.036 | 0.029 | 0.020 | 0.017 |
| RPSS, lasso (older) | 0.035 | 0.034 | 0.032 | 0.025 | 0.013 | 0.009 |
| RPSS, no forecast | 0.035 | 0.033 | 0.029 | 0.016 | 0.005 | 0.001 |
| corr of P(ERE) with ERE days | 0.245 | 0.259 | 0.268 | 0.251 | 0.229 | 0.215 |
| AUC | 0.699 | 0.707 | 0.710 | 0.692 | 0.676 | 0.664 |
| ERE rate, 10% of days rated most likely | 29% | 33% | 34% | 35% | 35% | 34% |
| ERE rate, 10% rated least likely | 3.0% | 2.6% | 2.6% | 3.6% | 4.9% | 4.9% |
| Brier skill | 4.2% | 4.9% | 5.5% | 4.6% | 3.5% | 2.8% |
| AUC, 91+ cells only | 0.734 | 0.713 | 0.643 | 0.607 | 0.574 | 0.531 |

Normal chance alone gives corr 0.140 and AUC 0.610.

**Transformer minus each baseline (RPSS, 90% range):**

| vs | 8 d | 10 d | 12 d | 14 d |
|---|---|---|---|---|
| ridge, same inputs | +0.008 [+0.003, +0.013] | +0.007 [-0.000, +0.014] | +0.008 [+0.001, +0.015] | +0.006 [-0.001, +0.014] |
| lasso (older) | +0.012 [+0.004, +0.019] | +0.012 [+0.002, +0.021] | +0.015 [+0.003, +0.026] | +0.014 [+0.002, +0.025] |
| no forecast | +0.015 [+0.005, +0.024] | +0.020 [+0.006, +0.033] | +0.023 [+0.009, +0.036] | +0.022 [+0.008, +0.035] |
| true BSISO 5 days before | +0.008 [-0.002, +0.020] | +0.001 | -0.007 | -0.012 [-0.027, +0.003] |

**Why the daily correlation stays near 0.25.**
- The BSISO only shifts the odds of a large ERE, between about 4% and 28%. Whether an ERE happens on a given day
  depends on the weather: monsoon low-pressure systems and how they interact (Nikumbh et al. 2020).
- Correct probabilities that never go above about 30% cannot correlate much above 0.3 with yes/no days.
- The forecast is already at about 90% of that limit at 8 days.

---

## 6. Weekly view (fig 5)

| window, model | RPSS (0/1/2+ days) | corr, expected vs observed count | AUC any ERE day | AUC 2+ days |
|---|---|---|---|---|
| days 1-7, transformer | 0.133 | 0.44 | 0.76 | 0.75 |
| days 8-14, **transformer** | **0.081** | **0.42** | 0.72 | 0.72 |
| days 8-14, ridge | 0.058 | 0.38 | 0.71 | 0.69 |
| days 8-14, lasso (older) | 0.047 | 0.35 | 0.70 | 0.67 |
| days 8-14, true BSISO (perfect) | 0.129 | 0.44 | 0.76 | 0.75 |

Normal chance alone gives corr 0.23.

For days 8-14 the transformer beats ridge by +0.022 [+0.008, +0.036] and lasso by +0.033 [+0.015, +0.051]. Summing
over a week removes much of the day-to-day noise, and no single ERE day is lost: each one counts.

---

## 7. What mattered (fig 6)

Each change is compared with the model it was tested against. Score: rain RPSS averaged over leads 8, 10, 12 and
14, with 90% ranges.

| change | effect |
|---|---|
| BSISO1 only (no BSISO2, no time of year) | **-0.012** [-0.022, -0.003] |
| no BSISO2 | **-0.0085** [-0.016, -0.000] |
| no time of year | -0.0035 [-0.007, +0.000] |
| 90-day history instead of 60 | **-0.0045** [-0.008, -0.001] |
| 30 or 45-day history | -0.0016 / -0.0006 (noise) |
| encoder-decoder (Anirudh et al. 2025 replica) | **-0.0057** [-0.012, -0.000] |
| larger transformer / GRU / tiny transformer / no linear shortcut | -0.0021 / -0.0027 / -0.0001 / -0.0008 (noise) |
| summer days weighted x3 | **-0.0015** [-0.003, -0.000] |
| later leads weighted x2 / dropout / 10-day pieces | -0.0004 / -0.0008 / -0.0005 (noise) |
| forecast its own uncertainty | -0.0007 (noise) |
| also forecast BSISO2 | **-0.0021** [-0.004, -0.000] |
| 20 trained copies instead of 10 | +0.0002 (noise) |
| strength term weight 0 / 0.4 / **0.5** / 0.6 / 0.7 / 0.8 (vs 0.2) | -0.0013 / +0.0011 / **+0.0016** / +0.0013 / +0.0002 / -0.0017 |

**What helped:**
- the BSISO2 input;
- the time of year;
- a strength term of weight 0.5;
- a 60-day history.

The architecture search converged: the best variants differ by about 0.001, which is below the noise.

---

## 8. Limits, and what comes next

**Limits:**
- **The index ceiling.** With only the 4 BSISO numbers as input, no architecture forecast the BSISO usefully beyond
  about 7.5 days.
- **The daily ceiling.** The daily correlation cannot go much above about 0.3 with the BSISO alone.
- **The biggest events.** The 91+ cell events lose their signal fastest.

**Next steps (separate folders):**
- **Add observed rain on the issue day.**
  - Inputs: the latitude profile of the monsoon rain band, central-India rain, and "ERE today".
  - In a first test this raised the 1-day correlation from 0.24 to 0.44, and the week-1 RPSS from 0.134 to 0.195.
- **Forecast the BSISO from ERA5 weather maps with a CNN.** This should extend the forecast to weeks 2-3.

## 9. Figures

| file | shows |
|---|---|
| `figures/fig1_bsiso_forecast_accuracy.png` | BSISO correlation, strength kept and phase accuracy, leads 1-10 |
| `figures/fig2_rain_skill.png` | daily rain RPSS and correlation, leads 6-14; transformer minus ridge with 90% ranges |
| `figures/fig3_probability_accuracy.png` | reliability, ERE rate on the most/least likely days, AUC by ERE size |
| `figures/fig4_example_forecasts.png` | the 2019 season (the median-skill year, picked automatically) and one 14-day forecast |
| `figures/fig5_weekly_view.png` | weekly count correlations, week-2 reliability, the 2019 weekly series |
| `figures/fig6_what_mattered.png` | the strength-term curve and every other change tested |

## References

- Anirudh et al. 2025, *JGR: Machine Learning and Computation*: transformer forecasts of the monsoon intraseasonal
  oscillation
- Jie et al. 2017, *QJRMS*: the Asian summer monsoon and the BSISO in S2S models
- Lee et al. 2013, *Climate Dynamics*: the BSISO index
- Lee et al. 2017, *J. Climate*: subseasonal prediction of extreme precipitation over Asia from the BSISO
- Maeda & Satoh 2025, *GRL*: deep learning prediction of the BSISO
- Nie et al. 2023 (PatchTST) and Zeng et al. 2023 (DLinear): the model design
- Nikumbh et al. 2020, *GRL*: large-scale extreme-rain-producing synoptic systems
