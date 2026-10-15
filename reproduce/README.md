# Reproducing the paper

Code for *Particle transport by means of a vibrating string and membrane under variable tension* (K. L. Joiner). Every figure, table and the conveyor movie in the paper and supplement can be rebuilt from this folder.

## Setup

    pip install -r requirements.txt        # numpy, scipy, matplotlib, numba
    # the movie also needs ffmpeg on the PATH

Tested with Python 3.13, numpy 2.5, scipy 1.18, matplotlib 3.11, numba 0.68. All commands below run from the `code/` folder.

## Two stages

1. **Simulations** write data to `results/`. They are slow (about two days in total on 2 cores; the membrane conveyor runs take the longest).

       python reproduce/run_simulations.py --list          # every step, its command and what it feeds
       python reproduce/run_simulations.py                 # run them all
       python reproduce/run_simulations.py --only conveyor_d6

2. **Figures** are drawn from the data in `results/` and copied, under their paper names, to `../paper/submission/Figures/`.

       python reproduce/make_figures.py                    # all 19 figures, about 15 min
       python reproduce/make_figures.py --only Fig6 FigS6
       python reproduce/make_figures.py --movie            # also renders Movie S1

The full command list lives in `reproduce/steps.py`, so the two scripts and this file cannot drift apart.

## Figures

| Figure | Data (simulation step) | Figure command |
|---|---|---|
| Fig. 1, 2 | runs its own sweep | `forcing/run_paper_figures.py --mode 5` |
| Fig. 3 | `loop_string` | `forcing/run_main_figures.py --mode 5 --only loop` |
| Fig. 4 | `rates` | `forcing/run_main_figures.py --mode 5 --only rate` |
| Fig. 5 | `accel_sweep`, `accel_extra` | `acceleration_sweep/run_acceleration_density.py` |
| Fig. 6 | `ratchet5_rising`, `ratchet5_falling` | `particles/run_ratchet_figures.py --mode 5 --set _g3k` |
| Fig. 7 | `modes_5` … `modes_19` | `particles/run_conveyor_band_figure.py` |
| Fig. 8 | `conveyor_d6` | `particles/run_conveyor_snapshots.py --d 6 --cycle 30` |
| Fig. 9, 10 | `conveyor_d2` … `conveyor_d8`, `conveyor_d6_seed1`, `conveyor_d8_seed1` | `particles/run_reduced_model_figures.py` |
| Fig. S1, S2 | runs its own sweep | `forcing/run_paper_figures.py --mode 3` |
| Fig. S3, S4 | runs its own sweep | `forcing/run_paper_figures.py --mode 7` |
| Fig. S5 | `grains_mode3` | `particles/run_grain_figures.py` |
| Fig. S6, S7 | `ratchet11_forward_s0`, `ratchet11_reverse_s0` | `particles/run_ratchet_figures.py --mode 11 --names forward reverse` |
| Fig. S8 | `modes_5` … `modes_19` | `particles/run_conveyor_modes_figure.py` |
| Fig. S9 | `accel_sweep`, `accel_extra` | `acceleration_sweep/run_acceleration_density.py --raw` |
| Movie S1 | `conveyor_d6` | `particles/adjustable_conveyor.py movie --d 6 --tag _ret --crf 32` |

## Tables

| Table | Numbers come from |
|---|---|
| 1, 2 | `results/main_loop_string_mode5.json`; membrane column of Table 2 from `main_loop_membrane_mode51.json` |
| 3 | `results/main_rates_string_mode5.csv` (rows with beta = 0.015) |
| 4 | `results/grain_sweeps_mode5.csv` |
| 5 | `results/ratchet_summary_mode5_g3k.csv` (written by the Fig. 6 command) |
| 6, 7 | `results/reduced_model_summary.csv` (written by the Fig. 9 command) and the fit in `particles/conveyor_model.py fit --d 8 6 4 --tag _ret` |
| S1 | printed by `forcing/run_paper_figures.py --mode 3` and `--mode 7` |
| S2 | `results/string_grain_rate_scan.csv` |
| S3 | `results/conveyor_modes_summary.csv` (written by the Fig. S8 command) |

## Random seeds

Grain runs are seeded (`--seed`, default 0), so a rerun on the same machine gives the same data. The membrane conveyor run with `--seed 1` and the mode-11 ratchet runs with seeds 1 and 2 are the repeat runs quoted in the text. Compiled (numba) code and different library versions can change the last digits of chaotic grain paths; the statistics in the paper do not depend on this.

## Folder map

| Folder | Contents |
|---|---|
| `forcing/` | String and membrane solvers, drives (lookup, phase-locked loop, amplitude control), drive figures |
| `particles/` | Grain models, ratchet, mode scan, membrane conveyor, reduced model, their figures |
| `acceleration_sweep/` | Grains on a fixed standing wave (Fig. 5, S9) |
| `reproduce/` | This file, the step list and the two driver scripts |
| `results/` | Data and figures written by the scripts |
| `matlab_ports/`, `experimental/` | Thesis-era ports and side projects; not used by the paper |
