# Acceleration sweep: grains on a standing wave against antinode acceleration

Separate from `particles/`. Asks one question: for a fixed standing wave (mode 5, 20 Hz, L = 1 m, SI units), how does the place where grains end up depend on the antinode acceleration `Gamma0 = A w^2 / g`, and on what a grain does while it is in contact with the surface?

- `standing_wave_grains.py` – compiled grain model on the analytic mode shape (needs `numba`). Contact rules follow `particles/fast_grains.py`, plus rolling in contact with Coulomb rolling resistance `mu_r`.
- `run_acceleration_sweep.py` – the sweep. Writes `results/accel_sweep.npz` and `results/accel_sweep.csv`. About 70 min on 2 cores; `--quick` for a 15 s check.
- `run_acceleration_figure.py` – `results/fig_accel_sweep.png` / `.pdf` (grey scale).

## Protocol

2000 grains start uniformly along the string. Each run is 1800 drive periods (90 s), cut into six blocks of 300 periods; the first two blocks are discarded and the last four averaged. The standard error is taken over four independent subgroups of 500 grains (grains do not interact). `Gamma0` runs from 0.6 to 6.0 in steps of 0.2.

Six contact rules:

| Case | In contact | Roughness |
|---|---|---|
| `held` | grain stays where it landed | 0 |
| `mu0.05`, `mu0.01`, `mu0.002` | grain rolls, rolling resistance `mu_r` | 0 |
| `rough_held` | stays where it landed | 0.3 |
| `rough_mu0.01` | rolls, `mu_r = 0.01` | 0.3 |

The largest surface slope is `A k = 0.0098 * Gamma0`, so `mu_r = 0.05` holds grains almost everywhere and `mu_r = 0.002` almost nowhere.

## Results (uniform spread = mean node distance 5 cm; antinode = 10 cm)

- **Below `Gamma0 = 1.2` nothing moves** unless the grains roll freely.
- **Rolling sends grains to the antinodes, not the nodes.** A grain in contact is heaviest at the bottom of the surface's swing, where the slope points toward the antinode, so the cycle-averaged push is toward the antinode. With `mu_r = 0.002` more than 98% of grains sit within 2 cm of an antinode for `Gamma0` from 1.4 to 2.0.
- **Smooth, held grains barely move** (3.9 to 5.9 cm). The structure in that curve is about ten times the standard error (0.05 cm), so it is locked bouncing orbits, not noise.
- **Roughness is what collects grains at the nodes**, as `particles/README.md` already found. `rough_held` switches on at `Gamma0 = 2.2` and improves steadily to `Gamma0 = 4.4` (97% of grains within 2 cm of a node, mean distance 0.94 cm).
- **There is a bad band at `Gamma0` 4.6 to 5.0**, where collection collapses (47% near a node at 5.0, 39% to 45% parked near antinodes). It recovers above 5.2 and reaches 95% at 6.0.
- **Rolling undoes about half of the node collection** (`rough_mu0.01` plateaus near 3.2 to 3.6 cm).

## Caveats

- `rough_held` is still collecting at 90 s (panel d: 0.1 to 0.5 cm change across the averaging window), so its values understate the final collection and the best `Gamma0` may shift with a longer run.
- One standing wave only; no tension sweep, no moving nodes, no grain-grain collisions.
- The bad band at 4.6 to 5.0 has not been explained. It may depend on restitution (0.3) and the stick threshold.
