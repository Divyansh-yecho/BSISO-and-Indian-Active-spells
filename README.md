# BSISO and Indian active/break monsoon spells

**Can the slow tropical wave that paces the monsoon be forecast weeks ahead, and does that tell us
when India will get heavy-rain or dry spells?**

> **Status: work in progress.** Part 1 (does the physical chain exist?) is done. Part 2 (forecasting it
> with a deep-learning model) has **not been started**. See [Where things stand](#where-things-stand).

---

## The question in plain words

During the Indian summer monsoon (June-September) rain does not fall evenly. Spells of heavy rain over
central India (**active spells**) alternate with spells of little rain (**break spells**), each lasting
from a few days to about two weeks. Knowing weeks in advance which is coming matters for farming, water
management and flood preparedness.

A slow tropical oscillation, the **boreal summer intraseasonal oscillation (BSISO)**, drifts northward
over India every 30-60 days and is known to pace these spells. It is summarised by two daily numbers
(called PC1 and PC2) that give its **phase** (where it is in its cycle) and its **amplitude** (how strong
it is).

## Goal

The submitted project description:

> Active and break spells of the Indian summer monsoon are strongly modulated by the boreal summer
> intraseasonal oscillation (BSISO), summarised by a two-component index giving its daily amplitude and
> phase. This project develops a deep-learning approach that reads maps of the tropical atmosphere to
> forecast this index at two-to-four-week leads, and then examines how far that skill carries through to
> predicting Indian active and break spells.

Concretely, three questions, with their status in the literature:

| question | status |
|---|---|
| Does the BSISO modulate Indian active/break spells? | established in the literature, and reproduced in earlier work of this project |
| Can the BSISO be forecast with deep learning? | established (Maeda & Satoh 2025) |
| Can a deep-learning BSISO forecast be turned into a forecast of Indian active/break spells? | **we did not find a published study. This is the project** |

## The physical chain being tested

```
BSISO phase and amplitude
        |   (changes how many low-pressure systems form, and how strong they are)
        v
Monsoon low-pressure systems (LPS)
        |   (LPS deliver more than half of India's monsoon rain)
        v
Heavy rain over central India
        |
        v
Active and break spells
```

The central idea is that the BSISO does not rain on India directly. It changes the **storms** that do.
So a good forecast of the BSISO should carry through to the storms and then to the spells, and the
project measures how far it carries.

## Plan

**Part 1: does the chain exist? (largely done)**
Measure each link from observations, with a statistical test that respects the fact that storms and
spells last several days. Every p-value in this repository comes from a shuffling test, never from
textbook formulas (which are far too lenient for multi-day events).

**Part 2: forecast it (not started)**
1. Baselines first: persistence, climatology, a linear (VAR) model and a lasso.
2. A small transformer reading 60 days of the BSISO index (PC1, PC2) and predicting the next 30, trained
   with a loss that also protects the amplitude (`0.8 * MSE(index) + 0.2 * MSE(amplitude)`).
3. Turn the forecast phase, amplitude and the storm state into a **probability** of an active or break
   spell at each lead, using logistic regression.
4. Verify it as a probabilistic forecast: Brier skill score against climatology, a reliability diagram,
   and a phase diagram. At an 8% base rate, even a small positive skill at 10 days would be a
   respectable result.

The transformer's job is specific. Earlier diagnostics showed that a linear model already reproduces the
**average** advance of the BSISO phase; what it cannot do is predict when the wave runs fast or slow.
That departure from the average advance is what the transformer has to learn.

## Where things stand

| piece | status | where |
|---|---|---|
| BSISO phase calculation verified | done | [`BSISO-LPS`](BSISO-LPS) |
| BSISO to LPS: genesis and strength | done | [`BSISO-LPS`](BSISO-LPS) |
| LPS to extreme rain (IMERG) | done | [`BSISO-LPS`](BSISO-LPS) |
| LPS to active/break spells; mediation; circularity test | done in earlier work | scripts **not yet in this repo** |
| Where the forecastable information lives (lead 1-20 days) | done in earlier work | scripts **not yet in this repo** |
| Linear forecast baselines | done in earlier work | scripts **not yet in this repo** |
| Transformer, probability layer, verification | **not started** | |

## Key findings so far

Full numbers, figures and caveats are in
[`BSISO-LPS/BSISO_LPS_RESULTS.md`](BSISO-LPS/BSISO_LPS_RESULTS.md).

- **BSISO phase changes how strong LPS are** (strong evidence): on BSISO-active days, storms are
  12-27% weaker in phases 1-3 and 10-15% stronger in phases 5-7 (June-September 2000-2022).
- **BSISO phase modestly changes how often LPS form**: a hump peaking near phase 5, about 30% above the
  average rate.
- **BSISO amplitude does not change how often storms form**, and acts on their strength only by
  amplifying the effect of the phase.
- **Extreme rain clusters around storm centres**: within 500 km, 16% of the heaviest rain days fall in
  3.8% of the area.
- **Not supported:** a BSISO-caused lag between storm formation and peak strength, and the idea of one
  clean BSISO track per summer. Under a strict "amplitude below 1 ends the event" rule the BSISO breaks
  into about 10 short events per summer, and none completes a full cycle.

From earlier work (not yet in this repository): about 90% of active-spell days have an LPS, against 35%
of break-spell days; and a diagnostic using the true BSISO phase shows information about spells exists at
a 10-day lead, which six linear forecasts all fail to capture. That gap is the case for a non-linear
model.

## What is in this repository

```
README.md                      this file
BSISO-LPS/
  bsiso_lps_analysis.ipynb     the Kaggle notebook: BSISO and LPS analysis
  BSISO_LPS_RESULTS.md         explains every number, what is and is not supported
  figures/                     the 12 figures from the notebook
```

## Data

The data files are **not stored in this repository** (they are large and have their own licences).

| data | used for | source |
|---|---|---|
| BSISO index, daily PC1 and PC2, 1981-2026 | phase and amplitude | APCC real-time BSISO index (Lee et al. 2013) |
| LPS tracks, June-September 2000-2022 (345 storms) | storm genesis and strength | moist LPS catalogue (Vishnu et al.; exact citation to be confirmed) |
| IMERG daily rain, 0.25 degree, 5-40N, 60-100E, 2000-2024 | rain around storms, spell labels | NASA GPM IMERG |

## Key terms

| term | meaning |
|---|---|
| **BSISO phase** | which of 8 sectors the wave is in. Phases 4-6 are when it sits over India |
| **BSISO amplitude** | how strong the wave is that day. Below 1 it is treated as inactive and its phase is not meaningful |
| **LPS** | monsoon low-pressure system: a storm of the kind that brings India much of its monsoon rain |
| **genesis** | the first day of a storm's track |
| **active / break spell** | rainfall over the central monsoon zone (18-28N, 73-88E) at least one standard deviation above / below normal for at least three days |

## Roadmap

- [ ] Fix a small look-ahead in the spell labels (the rainfall standardisation currently uses the whole
      record).
- [ ] Add confidence intervals to every correlation.
- [ ] Decide and justify a definition of a BSISO event (see the results file).
- [ ] Test whether the storm-strength result is partly a shared-wind effect (the index and the storm
      strength both use 850 hPa circulation).
- [ ] Build and train the transformer; compare against persistence and the lasso.
- [ ] Probability layer and verification (Brier skill, reliability diagram).
- [ ] Move the earlier analysis scripts into this repository.

## Honest limits

- The dataset is small: 345 storms (223 when restricted to BSISO-active days) and about 25 summers, so
  several effects are modest and p-values near 0.02-0.04 would not survive a correction for the many
  tests run.
- The BSISO and the storm strength share a wind input, so part of the strength result may be the
  large-scale circulation itself. Genesis, a count of events, is much less exposed.
- Skill on active and break spells is expected to be modest. A project that beats simple baselines
  honestly is the aim, not a large headline number.

## References

- Lee, J.-Y., et al. (2013). Real-time multivariate indices for the boreal summer intraseasonal
  oscillation over the Asian summer monsoon region. *Climate Dynamics*, 40, 493-509.
- Maeda, S. & Satoh, M. (2025), *Geophysical Research Letters*: CNN-based BSISO forecasting.
- Anirudh, et al. (2025), *JGR: Machine Learning and Computation*: transformer forecasting of the MISO
  index.
- Hunt, K. M. R. & Turner, A. G. (2022), *Weather and Climate Dynamics*, 3, 1341-1358: BSISO and
  monsoon low-pressure systems.
- Dey, et al. (2022), *Theoretical and Applied Climatology*, 148, 819-831: active/break spells and the
  intraseasonal oscillation.

*Add full titles before sharing publicly.*

*Author: [@Divyansh-yecho](https://github.com/Divyansh-yecho). Licence: not yet chosen.*
