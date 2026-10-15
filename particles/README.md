# Particles: grains on a vibrating string and membrane

Goal: find out whether grains collect at the nodes of the driven surface and
whether they follow the nodes when the tension is swept.

## Files

| File | What it holds |
|---|---|
| `grains.py` | `Grains`: free flight, inelastic impact with the moving surface, resting contact |
| `surface.py` | Interpolates the solver's height, velocity and slope to the grain positions |
| `experiment.py` | `settle` (wave + grains at fixed tension), `sweep` (continue from a settled state), statistics |
| `run_grains.py` | One run: settle, sweep, hold. `--model string` or `membrane` |
| `run_grain_rate_scan.py` | Settle once per drive, then sweep at several rates from that state |

The surface comes from the `forcing/` solvers. The default drive is the
phase-locked loop with its amplitude loop holding the peak acceleration
(`pll+agc`, no tables); `table+gain` is the baseline.

## The grain model

* Free flight under gravity between contacts.
* Impact: in the frame of the surface point, the normal velocity is reversed
  and multiplied by `restitution`; the tangential velocity is multiplied by
  `tangential_restitution`. A tilted surface kicks the grain sideways.
* `roughness`: each impact adds a random sideways velocity of size
  `roughness * impact speed`, standing in for irregular grains and a rough
  surface.
* A grain that lands slowly stays on the surface, slides with a friction
  rate, and lifts off when the surface falls away faster than `g`.
* Gravity is set from `--gamma`, the peak surface acceleration in units of
  `g`. Around each node there is a quiet zone where the local value is below
  1 and a resting grain cannot lift off.

Not in the model: grain-grain collisions, any force from the grains back on
the surface, exact impact times (contacts are resolved at the end of each
time step), and in-plane motion of the surface.

## Results

Defaults: amplitude 0.05, peak acceleration 4 g, restitution 0.4, roughness
0.3, `beta = 0.05`, slope `a` from -0.15 to 0, hold 500. String: mode 3, 400
grains. Membrane: mode (3, 1), 1500 grains.

**Fixed tension.** Grains leave the antinodes and gather around the nodes and
the clamped ends. String: mean distance to the nearest still point falls from
0.85 to about 0.38 (evenly scattered grains: 0.88). The bands are as wide as
the quiet zone (half-width 0.375) plus a fringe.

**Roughness decides this.** With `--roughness 0` the grains do not go to the
nodes. They form narrow clusters, two at antinodes and one between an
antinode and a node, and the mean distance to a still point ends at 0.96,
above the even-scatter value. Seen at amplitude 0.05 / 4 g and at
0.15 / 3 g.

**Moving the nodes, string** (`run_grain_rate_scan.py`). "Trailing" is the
median distance behind their node of the grains that started within 0.70 of
it; the nodes travel 1.0.

| `da/dt` | Trailing, end of sweep | Trailing, after hold | Still at their node |
|---|---|---|---|
| 3e-5 | 0.43 | 0.43 | 49% |
| 1e-4 | 0.48 | 0.42 | 51% |
| 3e-4 | 0.70 | 0.39 | 53% |
| 1e-3 | 0.79 | 0.38 | 59% |
| 3e-3 | 0.85 | 0.36 | 58% |

* At `1e-4` and slower the grains are carried along, trailing by about the
  half-width of the quiet zone. They are herded by its trailing edge: a
  resting grain stays put until the edge passes, is shaken loose, and comes
  to rest again just inside.
* About half the grains stay with their node at every rate. Most of the rest
  are picked up by the neighbouring node or the quiet zones at the ends.
* At `3e-4` and faster the grains are not carried: they are left almost a
  full node displacement behind and re-gather afterwards at the nearest
  still point. That is usually their own node only because the nodes move
  less than half the node spacing here.
* So grains need about `1e-4`, ten times slower than the wave itself allows.
* The baseline drive gives the same picture.

**Moving the nodes, membrane.** Worse: 27% to 46% of the grains that started
on a nodal line are still on it after the hold, and the share is lowest for
the slowest sweeps. A grain shaken loose wanders in two dimensions, and once
it reaches the quiet band along a clamped edge it never leaves.

Figures: `../results/<model>_grains.png`, `<model>_grain_rate_scan.png`,
`string_grains_smooth.png`.

## What to try next

1. Higher `--gamma`: a narrower quiet zone should hold grains closer to the
   node.
2. A longer node travel (more than half the node spacing), which would
   separate "carried" from "re-gathered" cleanly.
3. On the membrane, modes or edges without a permanent quiet band next to
   the moving lines.
4. Exact impact times, grain-grain collisions, and grains loading the
   surface.

## Added scripts

- `run_membrane_grains.py` – grains on the membrane during a sweep, driven from a look-up table written beforehand by `forcing/run_make_table.py`.
- `run_ratchet.py` – ratchet conveyor on the string. Default: slow tension sweep, drive off, reset, repeat, with a slow back-and-forth control. `--steps a1 a2 a3` runs the fast stepping version instead. `--amplitude` should scale with the node spacing (0.05 for mode 3, 0.014 for mode 11). `--tag` adds a suffix to the output names.

  Example: `python particles/run_ratchet.py --mode 11 --a0 -0.064 --a1 0.064 --rate 8.5e-5 --cycles 5 --grains 500 --amplitude 0.014`
