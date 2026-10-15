# Vibrating string and membrane: code

Three sections, each with its own README.

| Folder | Purpose |
|---|---|
| `forcing/` | Forced, damped string and 2D membrane with an adjustable tension gradient. This is where node transport is engineered: drive strategies, tension schedules, and the diagnostics that say whether the nodes moved smoothly. |
| `particles/` | Grains on the vibrating string and membrane. Uses the `forcing/` solvers for the surface, so a tension sweep moves the nodes under the grains. |
| `matlab_ports/` | Python ports of the original MATLAB bead-on-string scripts, kept as the reference set. Not used by the two folders above. |
| `tests/` | `run_checks.py`: quick regression checks for all three sections. |
| `results/` | Figures (`.png`) and data (`.npz`) written by the run scripts. |

## Reproducing the paper

`reproduce/README.md` maps every figure, table and the conveyor movie of the paper to the commands that make them. In short:

    python reproduce/run_simulations.py      # data, slow
    python reproduce/make_figures.py --movie  # figures and movie from the data

## Setup

    pip install -r requirements.txt        # numpy, scipy, matplotlib

## Run

From this folder (add `--model membrane` for the membrane):

    python forcing/run_sweep.py                  # PLL against baseline, ~1 min
    python forcing/run_sweep.py --legacy         # the approaches that fail
    python forcing/run_amplitude.py              # holding the amplitude
    python forcing/run_rate_scan.py              # how fast the sweep can go
    python particles/run_grains.py               # grains through a sweep, ~2 min
    python particles/run_grain_rate_scan.py      # grains against sweep rate
    python tests/run_checks.py                   # ~2 min

Every run script takes `--help`. Add `--show` to open the figure as well as
saving it.

## Shared conventions

* Tension `T(x) = a * (x - L/2) + b`, with `|a| < 2b/L`. `a` is the slope
  that gets swept; `b = 1`, `L = 10` as in the thesis.
* Units are the thesis's non-dimensional ones (unit density).
* Resonant frequencies and mode shapes come from the eigenmodes of the
  discrete operator that is actually integrated, so the drive is exactly on
  resonance for the grid in use.
* Time stepping is 4th-order Runge-Kutta.

## Where things stand

* **Working drive:** a phase-locked loop keeps the mode through a tension
  sweep with no table of the resonance, on the string and the membrane. The
  table-lookup drive is kept as the baseline.
* **Amplitude:** a feed-forward gain holds the peak displacement within 1%;
  the loop's own amplitude control holds it within about 3% with no table.
* **Sweep rate:** limited by the wave travel time, not the damping. Up to
  `da/dt` of about `1e-3` the mode stays above 99.8% pure.
* **Grains:** with some roughness they gather at the nodes and are carried
  along when the nodes move slowly (`1e-4`), trailing by the half-width of
  the quiet zone; about half stay with their node. On the membrane more are
  lost, to the quiet bands along the edges.
* Details are in `forcing/README.md` and `particles/README.md`. The paper
  draft is in `../paper/`.
