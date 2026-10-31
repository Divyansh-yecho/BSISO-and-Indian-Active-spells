# BSISO and large extreme-rain events: what the numbers say

Companion to [`bsiso_ere_link.ipynb`](bsiso_ere_link.ipynb) (a Kaggle notebook).
Season analysed: June-September (JJAS), 2000-2024, IMERG rainfall at 0.25 degree.

The other folders linked the BSISO to storms and to active/break spells. This one asks the same questions for
**large extreme-rain events (EREs)**, and turns the answers into the forecast target for the second half of the
project. It takes ideas from Lee et al. (2017, *J. Climate*), who mapped the chance of extreme rain in each BSISO
phase, and Nikumbh et al. (2021, *J. Climate*), who found that big EREs over central India need an active monsoon.

Every number below is printed by the notebook, except a few read off a figure; those are marked
"(from the figure)".

---

## 1. The result in one paragraph

The BSISO decides **when and where** large, organised extreme rain falls over India, and the bigger the event,
the tighter that control. When the BSISO is over India (phases 4-6), a day with a large ERE in central India is
about 1.6-1.7 times as likely as normal; when it is away (phases 1-3), only a third as likely. The Northeast
gets its large EREs in the opposite phases. The signal is strongest about **5 days before** the rain, which is
what makes a forecast possible. It is a real intraseasonal signal, not rain persistence and not "wet years are
BSISO years". But it is modest in absolute terms: even a perfectly known BSISO removes only about 5% of the
day-to-day uncertainty about a large ERE. A stronger BSISO does not add anything once its phase is known.

### What is supported, and what is not

| claim | key number | verdict |
|---|---|---|
| Large organised EREs carry more BSISO information than other rain events | 0.030 bits/day vs 0.016 (active spells), 0.008 (single-cell extremes) | **supported** |
| The BSISO moves the chance of a large central-India ERE day | 3.7% (phase 1) to 28.4% (phase 5), against 15.3% on average | **strong** |
| The BSISO says most about the rain **5 days ahead** | 0.030 bits/day at 4-5 days before vs 0.019 on the same day | **clear** |
| It is not just rain persistence | rain alone 0.017 bits/day at 5 days; BSISO adds on top (0.041 together) | **supported** |
| It is timing within the season, not year-to-year | shifting the BSISO within its own season removes it (p = 0.005) | **supported** |
| Large EREs cluster in the BSISO cycle as strongly as active spells | clustering 0.28 (central India) and 0.34 (Northeast) vs 0.35 (active spells), all p <= 0.004 | **strong** |
| The bigger the ERE, the tighter the link | central-India track starts in phases 4-6: 51% (16-39 cells), 53% (40-90), 80% (91+), chance 40% | **strong** |
| The Northeast takes the opposite phases | 73% of 40+ cell Northeast tracks start in phases 1-3 (chance 37%) | **supported**, few events |
| A stronger BSISO brings more large EREs | strong vs moderate amplitude: x1.05 (p = 0.81) | **not supported** |
| The Northeast can be forecast from the BSISO | 0.001-0.003 bits/day | **not supported** |
| The link is stable over time | 0.040 bits/day in 2000-2012, 0.020 in 2013-2024 | **weaker recently** |

---

## 2. Data and terms

| dataset (Kaggle) | file | contents |
|---|---|---|
| IMERG rainfall | `imerg_india_025_jjas_2000_2024.nc` | daily rain (mm/day), 0.25 degree, 5-40N, 60-100E, 3,050 JJAS days |
| BSISO index | `BSISO.INDEX.NORM.LY.data` | daily BSISO1 (PC1, PC2), APCC index (Lee et al. 2013) |
| India mask | `india_mask_025.npy` | India land cells on the IMERG grid |
| ERE catalogue | `objdays_enriched.parquet`, `tracks_enriched.parquet` | 10,877 ERE object-days and 10,006 tracks from the summer tracker |
| BSISO forecasts (optional) | `lasso_bsiso_forecasts.parquet` | statistical (lasso) BSISO forecasts 1-20 days ahead, made without the test years |

The data files are not stored in this repository.

| term | meaning |
|---|---|
| **ERE** | a group of at least 16 connected IMERG cells (about 12,000 km2), each above its own wet-day 99th percentile (at least 50 mm/day). Built exactly as in the summer tracker |
| **study event** | a day with at least one ERE centred in central India (17-26.5N, 74-88.5E) |
| **size class** | the size of the largest ERE: 16-39, 40-90 or 91+ cells |
| **track** | one ERE followed from day to day by the summer tracker (objects linked by overlap) |
| **phase** | one of 8 sectors of the BSISO1 plane. Phases 4-6 = the rain band over India |
| **amplitude** | how strong the BSISO is. Below 1 it is treated as weak |
| **multiplier** | chance of the event in that phase divided by its normal chance for that time of the season |
| **bits per day** | how much better the forecast's probabilities were than the seasonal chance, on the days that happened. About 0.6 bits/day of uncertainty is left after the seasonal chance |
| **RPSS** | skill of a forecast with several classes against the seasonal chance (0 = no better, 1 = perfect) |
| **season re-dating** | the test behind every p-value: each summer is given another summer's BSISO, which keeps the seasonal cycle but breaks the real timing |

---

## 3. The figures

All figures are in [`figures/`](figures).

### Figure 1. Why this definition of an ERE (`fig01_ere_definition.png`)

Each candidate event was scored by how much the BSISO, 5 days earlier, tells us about it (each summer predicted
from the other 24):

| event | days per summer | bits per day |
|---|---|---|
| **16+ cell ERE in central India (this study)** | 18.7 | **0.030** |
| Nikumbh-style 1-degree box above its 99.5th percentile | 20.5 | 0.028 |
| central-India mean rain in its top 20% | 24.4 | 0.018 |
| active spell day | 11.4 | 0.016 |
| any extreme cell (99th percentile, any size) | 64.8 | 0.008 |

The right panel repeats the score for thresholds from the 98th to the 99.5th percentile and minimum sizes of 8 to
40 cells. The scores form a flat ridge (lower threshold with bigger size works as well), and the 99th
percentile with 16 cells sits on it. Nothing nearby is clearly better. So the summer tracker's definition is kept
as it is.

### Figure 2. The Bayes view: how much and when (`fig02_bayes_multiplier.png`)

The normal (prior) chance of a large-ERE day in central India is 15.3%. Reading the BSISO phase 5 days earlier:

| BSISO phase 5 days earlier | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | weak |
|---|---|---|---|---|---|---|---|---|---|
| chance of a large-ERE day | 3.7% | 6.6% | 5.6% | 23.5% | **28.4%** | 27.1% | 17.9% | 15.3% | 13.6% |
| x the seasonal chance | 0.26 | 0.41 | 0.35 | 1.60 | **1.70** | 1.59 | 1.10 | 0.98 | 0.94 |

The right panel asks how many days before the rain the BSISO is most useful. It peaks at 4-5 days and stays high
from about 2 to 9 days. Central-India rain itself, read on the same earlier day, does the opposite: it is very
informative the day before (0.10 bits/day, rain lasting from one day to the next) and fades fast (0.017 by 5
days). So the 5-day signal is the BSISO arriving, not rain carrying on.

Checks printed under the figure:

| check | bits per day |
|---|---|
| BSISO 5 days earlier | 0.030 |
| rain persistence on that day, no BSISO | 0.017 |
| BSISO + rain persistence | 0.041 |
| BSISO shifted in time within its own season (99th percentile of 200 shifts) | 0.005 (p = 0.005) |
| 2000-2012 only / 2013-2024 only | 0.040 / 0.020 |
| Northeast, BSISO same day / 5 days earlier | 0.003 / 0.001 |

### Figure 3. Probability maps for each phase (`fig03_phase_maps.png`)

One map per BSISO phase: where a large ERE within about 90 km becomes more likely (blue) or less likely (brown)
5 days later, compared with the seasonal chance. Solid colour beats chance (p < 0.05). The rain band can be seen
moving north as the phase advances: central India is strongly suppressed in phases 1-3, lights up in phase 5
(central India and Odisha), the favoured area moves to the Gangetic plain in phase 7 while the south-centre
dries, and the peninsula is suppressed in phase 8. The weak-BSISO map is almost blank.
Share of mapped places beating chance: phase 1 24%, phase 2 17%, phase 3 17%, phase 5 15%, phase 8 10%; phases 4
and 6 only 1%, because their effect is spread thin per place and only shows over central India as a whole.

### Figure 4. Clustering compared with spells (`fig04_clustering_vs_spells.png`)

How tightly events gather in one part of the BSISO cycle (0 = spread evenly):

| events | clustering | chance | p | centre phase |
|---|---|---|---|---|
| active spell days | 0.35 | 0.13 | 0.001 | 5 |
| break spell days | 0.25 | 0.13 | 0.058 | 8 |
| central-India ERE events | 0.28 | 0.08 | 0.001 | 6 |
| Northeast ERE events | 0.34 | 0.15 | 0.004 | 2 |

Large EREs are as phase-locked as active spells, and the Northeast sits on the other side of the cycle.

### Figure 5. Where the events sit in each phase (`fig05_latitude_by_phase.png`)

In phases 4-6 large events sit inside the central-India box, with a median latitude near 21N; in the other phases
the median shifts north to about 22-24.5N, towards the foothills (from the figure).

### Figures 6-9. ERE tracks on the BSISO phase plane

These use the summer tracker's catalogue: 855 tracks reach 16+ cells.

**Figure 6, the 15 longest tracks** (`fig06_longest_tracks.png`). The biggest central-India tracks (2 Aug 2006,
279 cells; 31 Jul 2015, 204; 20 Jul 2014, 198; 30 Jun 2006, 187; 4 Aug 2007, 183) all happen with a strong BSISO
in phases 5-6. A track lasts 4-7 days and the BSISO moves about one phase in that time; the ERE grows while the
BSISO is in phases 5-6. The exceptions are informative: Northeast tracks in phases 1-2, Western Ghats tracks in
phases 5-7, a late-season central-India track in phase 2.

**Figure 7, where tracks start** (`fig07_track_starts.png`). Central-India starts (405 tracks) pile up in phases
4-7, with the large ones in phase 5 and outside the amplitude-1 circle. Northeast starts (110 tracks) sit on the
opposite side, phases 1-3.

**Figure 8, starts by size** (`fig08_track_size_by_phase.png`). Each slice shows the number and the percentage of
that size class starting in the phase:

| region, size | tracks (BSISO strong) | starts in the favoured phases | chance | p |
|---|---|---|---|---|
| central India, 16-39 cells | 153 | 51% (phases 4-6) | 40% | 0.002 |
| central India, 40-90 cells | 85 | 53% | 40% | 0.024 |
| central India, 91+ cells | 25 | **80%** (15 in phase 5 alone) | 40% | < 0.001 |
| Northeast, 16-39 cells | 63 | 52% (phases 1-3) | 37% | 0.048 |
| Northeast, 40+ cells | 11 | 73% | 37% | 0.032 |

**Figure 9, the average BSISO path around a track's start** (`fig09_mean_path_by_size.png`). For the 91+ cell
central-India tracks, the BSISO swings from phase 4 into phase 5 and the track starts right at amplitude 1. For
40-90 cells the start is inside the circle near phase 6; for 16-39 cells it is close to the centre. In the
Northeast the 40+ cell tracks start in phase 2 after the BSISO passes through phases 8 and 1. Averaging pulls the
paths towards the centre, so read their direction, not their distance.

### Figure 10. A stronger BSISO does not matter (`fig10_amplitude.png`)

Within the favoured phases, a strong BSISO (amplitude 1.75 or more) brings large EREs x1.05 as often as a
moderate one in central India (p = 0.81) and x1.06 in the Northeast (p = 0.92). This differs from Lee et al.
(2017), who found amplitude mattered for extreme rain over Asia; over India, the phase carries the signal.

### Figure 11. The forecast target: daily ERE size class (`fig11_daily_size_forecast.png`)

The class of a day is the size of its largest central-India ERE. Normal shares: no ERE 84.7%, 16-39 cells 8.9%,
40-90 cells 5.0%, 91+ cells 1.4%.

The BSISO 5 days earlier changes each class, and bigger EREs swing more: in phase 5 the multipliers are x1.32
(16-39 cells), x1.96 (40-90) and x3.19 (91+); in phases 1-3 they fall to x0.16-0.49, and there was no 91+ cell
day after phases 1-2.

Skill of the 4-class daily forecast (RPSS) by days ahead:

| days ahead | 1 | 4 | 7 | 10 | 14 |
|---|---|---|---|---|---|
| perfect BSISO 5 days before the target (ceiling) | 0.035 | 0.035 | 0.035 | 0.035 | 0.035 |
| lasso BSISO forecast | 0.035 | 0.035 | 0.032 | 0.025 | 0.009 |
| BSISO on the issue day (no forecast) | 0.027 | 0.035 | 0.031 | 0.016 | 0.001 |

Up to 5 days ahead the BSISO day that matters is already observed. Up to about 8 days, today's BSISO is nearly as
good as a perfect one. A BSISO forecast only helps beyond that. For "any large ERE that day", the share of the
day-to-day uncertainty removed is 5.0% (perfect BSISO), 3.2% (lasso, 10 days ahead) and 2.2% (today's BSISO, 10
days ahead).

---

## 4. What this means for the project

1. **The forecast target is fixed.** The second half of the project forecasts, for each day up to 14 days ahead,
   the chance of no ERE, a 16-39, a 40-90 or a 91+ cell ERE in central India. The event is the summer tracker's
   ERE, unchanged.
2. **The BSISO is a real but modest source of skill.** It moves the odds a lot (4% to 28% for a large-ERE day,
   and x3 for the biggest events), but it explains only about 5% of the day-to-day uncertainty. The rest is
   weather that the BSISO does not control.
3. **The transformer has a narrow, clear job.** Up to about 8 days ahead the observed BSISO is enough. The
   transformer can only help 9-14 days ahead, by forecasting the BSISO better than the lasso does. The room left is
   the gap between 0.025 and 0.035 RPSS at 10 days.
4. **Phase, not amplitude.** The forecast needs the BSISO phase; a stronger BSISO adds nothing.
5. **Central India only.** The Northeast follows the opposite phases, but the BSISO tells too little about it
   ahead of time to forecast it this way.
6. **Rain helps close in.** Today's rain adds a lot 1-3 days ahead and the BSISO adds on top of it, so the final
   forecast can use both.

---

## 5. Caveats

- One rainfall dataset (IMERG), 25 summers. The 91+ cell class has only 44 days and 33 tracks, and the 40+ cell
  Northeast class only 19 tracks.
- The link is about half as strong in 2013-2024 as in 2000-2012. That could be chance in 12-year samples or
  changes between IMERG versions, but it should be kept in mind.
- The ERE definition was compared against about 60 alternatives on the same data, so the best scores are slightly
  flattering; the comparisons that matter (against spells and single-cell extremes) are large.
- The lasso BSISO forecasts are out of sample by blocks of years; the rain models built on them are tested one
  summer at a time.
