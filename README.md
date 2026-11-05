# BSISO and Indian Active/Break Spells

Rain during the Indian summer monsoon doesn't fall evenly. It comes in spells: active spells with
heavy rain over central India, and break spells where it stays dry. One of the things that sets the
rhythm of these spells is the boreal summer intraseasonal oscillation (BSISO), a slow tropical wave that
moves north over India roughly every 30-60 days.

This project looks at how the BSISO affects the spells, and whether forecasting the BSISO a few weeks
ahead can help forecast the spells.

## The idea

The BSISO doesn't rain on India directly. It changes the monsoon low-pressure systems (LPS) that do.

BSISO -> low-pressure systems -> heavy rain -> active/break spells

So the plan has two parts:

1. Check each link of this chain in observations.
2. Forecast the BSISO 2-4 weeks ahead with a transformer, and use that forecast together with the
   storm information to estimate the probability of an active or break spell.

## Where it stands

Still work in progress. Part 1 is mostly done: the BSISO-to-LPS and LPS-to-rain parts are in this repo.
Part 2, the forecasting, has started: a first forecast of large extreme-rain days over central India from the
BSISO alone is in `BSISO-daily-ERE-Prediction/`, and adding the observed monsoon rain band on the forecast day
is in `BSISO_ERE_PREDICTION_RAIN-BANDS/`. The rain band helps a lot in the first week; beyond a week the skill
still comes only from the BSISO forecast. The rain-based MISO index was also tested in place of the BSISO: it
forecasts itself further ahead, but it is worse for forecasting extreme rain beyond a few days
(`BSISO_MISO_Comparison_Active_ERE/`).

So far, the BSISO phase clearly changes how strong the storms are and has a smaller effect on how often
they form, and the heaviest rain sits close to the storm centres.

## What's here

- `BSISO-LPS/` has the notebook, a write-up explaining the numbers and results, and the figures.
- `BSISO-Active/` links the BSISO to active and break spells, and tests how far ahead a spell can be
  predicted. Same layout: notebook, write-up and figures.
- `BSISO-Rain/` repeats the plots of Dey et al. (2022) for the BSISO and our spells, and looks at how the
  BSISO moves extreme rain around India. Same layout.
- `BSISO-ERE-Link/` links the BSISO to large extreme-rain events (16+ cells) and their tracks, and sets the
  forecast target for Part 2: the daily chance of no ERE or a small, medium or large ERE over central India.
  Same layout.
- `BSISO-daily-ERE-Prediction/` forecasts that target 1-14 days ahead from the BSISO alone: a small
  transformer forecasts the BSISO, and a regression turns it into daily and weekly ERE chances. It also lists
  every model variant tested and what mattered. Same layout.
- `BSISO_ERE_PREDICTION_RAIN-BANDS/` adds the monsoon rain band seen on the forecast day, and compares the best
  model with ridge, lasso, the BSISO-only model and the normal seasonal chance on the same days. It also has the
  code and result tables.
- `BSISO_MISO_Comparison_Active_ERE/` compares the BSISO with the MISO index (the rain-based index used by Dey et al.
  2022 and Anirudh et al. 2025). It covers how far ahead each index forecasts itself, how well each forecasts
  extreme-rain events, when each index "knows" about an event, and how active/break spells and EREs cluster in each
  phase diagram. MISO describes the monsoon very well on the day, but only the BSISO gives warning days ahead.
  Notebook, figures, result tables and code.

The data isn't stored in the repo. It comes from the APCC BSISO index, an LPS track catalogue and IMERG
rainfall.
