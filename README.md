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
Part 2, the forecasting, hasn't been started yet.

So far, the BSISO phase clearly changes how strong the storms are and has a smaller effect on how often
they form, and the heaviest rain sits close to the storm centres.

## What's here

- `BSISO-LPS/` has the notebook, a write-up explaining the numbers and results, and the figures.
- `BSISO-Active/` links the BSISO to active and break spells, and tests how far ahead a spell can be
  predicted. Same layout: notebook, write-up and figures.
- `BSISO-Rain/` repeats the plots of Dey et al. (2022) for the BSISO and our spells, and looks at how the
  BSISO moves extreme rain around India. Same layout.

The data isn't stored in the repo. It comes from the APCC BSISO index, an LPS track catalogue and IMERG
rainfall.
