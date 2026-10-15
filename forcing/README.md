# Forcing: string and 2D membrane with a tension sweep

Goal: move the nodes of a driven standing wave by changing the tension
gradient, without losing the mode.

**Working drive:** the phase-locked loop (`PLLDrive`). It finds the
resonance from a sensor and needs no table.
**Baseline:** the table lookup (`TableDrive`), which reads the resonance
from the computed eigenvalues. It is exact when the system is known exactly
and is what the loop is compared against.

## Files

| | File | What it holds |
|---|---|---|
| Models | `string_model.py` | `TensionString`: equation, discrete eigenmodes, resonance, energy, node finder |
| | `membrane_model.py` | `TensionMembrane`: the same for a rectangular membrane, modes labelled `(n, m)` |
| | `bessel.py` | Analytic Bessel eigenvalues, mode shapes and nodes of the string (thesis Ch. 2-3) |
| Drives | `drive.py` | `PLLDrive` (working drive), `TableDrive` (baseline), `Sensor` |
| | `compensation.py` | Feed-forward amplitude gain from the mode table, with the rate correction |
| | `legacy_drives.py` | Approaches that fail: fixed frequency, phase-jumping update, plain velocity feedback |
| | `schedules.py` | Tension schedules `a(t)`: constant, step, linear ramp, smooth ramp, constant-frequency-rate |
| Engine | `simulation.py` | `Simulation` (RK4 stepping, per-cycle diagnostics), `make_drive`, `steady_state_energy` |
| | `sweep_experiment.py` | Shared start / sweep / hold experiment, saving and plotting |
| | `plotstyle.py` | Figure style and the fixed colour for each drive |
| Experiments | `run_sweep.py` | Loop against baseline through a sweep; `--legacy` for fixed frequency and direct substitution, `--velocity-feedback` to add plain velocity feedback |
| | `run_amplitude.py` | Holding the amplitude through a sweep |
| | `run_rate_scan.py` | How fast the sweep can go |
| | `run_bessel_check.py` | Bessel solution against the simulated string (string only) |

Every experiment takes `--model string` (default) or `--model membrane`.

Drive names used by `make_drive` and the `--drives` options:

| Name | Frequency from | Amplitude from |
|---|---|---|
| `pll` | phase-locked loop | not controlled |
| `pll+agc` | phase-locked loop | its own amplitude loop (no table at all) |
| `pll+gain` | phase-locked loop | feed-forward gain (amplitude table) |
| `table` | resonance table | not controlled |
| `table+gain` | resonance table | feed-forward gain (amplitude table) |
| `fixed`, `naive`, `self` | legacy, see below | |

## Equations

String:

    u_tt = (T(x) u_x)_x - beta u_t + F0 shape(x) c(t)

Membrane (isotropic tension varying along x only):

    u_tt = d/dx(T(x) u_x) + T(x) u_yy - beta u_t + F0 shape(x, y) c(t)

with `T(x) = a (x - L/2) + b`. The membrane modes separate as
`phi(x) sin(m pi y / Ly)`, but `phi` solves
`-(T phi')' + T (m pi / Ly)^2 phi = lam phi`, which has an extra term
compared with the string. The membrane modes are therefore not products of
the string's Bessel modes (the thesis conclusions suggest they are).

## Diagnostics

Once per drive cycle the simulation demodulates the motion at the drive
phase to get the standing-wave shape, then records:

* **energy**, compared with the steady-state energy the target mode would
  have at the current tension (`steady_state_energy`, a modal sum);
* **purity**, the share of the oscillation that is the target mode (0 to 1);
* **nodes**, measured against the target mode's nodes;
* **peak**, the largest displacement anywhere on the surface.

The shape is an average over one cycle, so it is compared with the target
mode at that cycle's mean tension. Runs start in the computed steady state
(`Simulation.start_in_steady_state`), so no settling run is needed.

## The phase-locked loop

The force is `F0 * shape * gain * cos(phi)`. A sensor reads displacement and
velocity (here averaged over the surface), which gives the phase lag of the
response directly. The loop steers its frequency until that lag is 90
degrees, which is the resonance.

* **One tuning number,** `bandwidth` (default 0.1). The loop needs
  `bandwidth^2` well above the rate at which the resonance moves. At 0.02 it
  lagged by 0.44 linewidths and lost 16% of the energy mid-sweep; 0.07 to
  0.15 matched the baseline.
* **It has to be started near the resonance,** on a steady state. Closing
  the loop from a fixed-frequency steady state, it pulled in from 1% off
  (half a linewidth) at bandwidth 0.1, and from 3% and 10% off at bandwidth
  0.03. From 3% and 10% off at bandwidth 0.1 it ran away (only a few cases
  tried).
* **One calibration.** The other modes add a constant phase offset to the
  sensor signal. `calibrate=True` reads that offset once at the start, on
  resonance. Uncalibrated, the loop settles 0.10 linewidths off and 4.5%
  low in energy. The offset grows with damping: at `beta = 0.05` the loop
  ends the sweep 1.8% off in energy.

## Results

Defaults: `beta = 0.015`, uniform forcing, sweep at `da/dt = 1e-4`, hold for
300. String: mode 3, `a` from -0.18 to 0 (nodes move by 1.4). Membrane:
10 x 7, mode (3, 1), `a` from -0.15 to 0.

### 1. Through the sweep (`run_sweep.py`)

| | Drive | Final energy | Lowest purity | Largest node error | Largest detuning |
|---|---|---|---|---|---|
| String | Phase-locked loop | 2000.5 | 1.0000 | 0.008 | 0.13 linewidths |
| | Table lookup | 2004.5 | 1.0000 | 0.006 | 0 |
| Membrane | Phase-locked loop | 11356.1 | 1.0000 | 0.006 | 0.08 linewidths |
| | Table lookup | 11358.9 | 1.0000 | 0.006 | 0 |

Predicted final energy: 2004.4 (string), 11356.4 (membrane).

### 2. Holding the amplitude (`run_amplitude.py`)

With the frequency alone under control the peak displacement drifts, because
the mode shape and its coupling to the force both change with `a`.

| Drive | Peak displacement, string | Peak displacement, membrane |
|---|---|---|
| `pll` | 0.675 to 1.000 | 0.753 to 1.000 |
| `pll+agc` | 0.966 to 1.000 | 0.975 to 1.000 |
| `pll+gain` | 0.989 to 1.002 | 0.992 to 1.004 |
| `table+gain` (baseline) | 0.992 to 1.002 | 0.993 to 1.005 |

* **Amplitude loop (`pll+agc`):** needs no table, only a measurement of the
  peak displacement. It has to be slow (`agc_bandwidth` about 0.01): faster
  settings made the two loops hunt, and on the membrane lose the pattern.
  So the error grows with the sweep rate, to about 13% at `1e-3`.
* **Feed-forward gain (`+gain`):** `gain = response(a0) / response(a)` from
  the mode table, times a rate correction
  `1 + (dln(w)/dt - 2 dln(m)/dt) / beta`. The correction is needed because a
  rising frequency lowers the amplitude by itself (the `w^(-1/2)` law from
  thesis Ch. 6). Without it the string's peak dips to 0.955; with it the
  peak stays within 1% even at `1e-3`.
* `--quantity acceleration` holds the peak acceleration instead of the peak
  displacement. Holding either one changes the energy: constant peak
  displacement raises the energy by a factor 2.4 on the string.

### 3. How fast the sweep can go (`run_rate_scan.py`)

Worst values from the start of the sweep to the end of a 400-unit hold.
String, `beta = 0.015`, phase-locked loop:

| Rate `da/dt` | Sweep lasts | Lowest purity | Largest node error | Largest energy deviation |
|---|---|---|---|---|
| 1e-4 | 1800 | 1.0000 | 0.008 | 0.8% |
| 3e-4 | 600 | 0.9998 | 0.022 | 1.8% |
| 1e-3 | 180 | 0.9980 | 0.066 | 3.1% |
| 3e-3 | 60 | 0.987 | 0.18 | 3.4% |
| 1e-2 | 18 | 0.929 | 0.40 | 6.5% |
| 3e-2 | 6 | 0.50 | 1.15 | 22% |

* **The limit is set by the wave, not by the damping or the drive.** The
  loop and the baseline give almost the same curves, as do `beta = 0.015`
  and `0.05`. The node error grows in proportion to the rate: the nodes
  trail their target by a fixed delay of about 9 time units, roughly the
  time a wave takes to cross the string.
* **Practical range:** up to about `1e-3` the mode stays above 99.8% pure
  and the nodes are within 5% of the distance they travel. That is the full
  sweep in about 25 drive periods. At `3e-2` and above (the whole sweep
  within one period) the mode is lost during the sweep.
* **Recovery:** the baseline always rebuilds the mode during the hold. The
  loop at `beta = 0.05` lost lock at `3e-2` and above and did not come back.
* **Membrane:** the same picture (purity 0.9990 and node error 0.05 at
  `1e-3`).

### 4. The Bessel solution on the string (`run_bessel_check.py`)

The thesis's Bessel modes, `J0` and `Y0` of `2 sqrt(lam T(x)) / |a|`, describe
the simulated string essentially exactly. Mode 3, 301 grid points:

| | Bessel against grid modes |
|---|---|
| Eigenvalue | relative difference 9e-5 or less |
| Resonant frequency | 0.0025 to 0.0027 linewidths (`beta = 0.015`) |
| Node positions | within 1e-4 |
| Mode shape | within 2e-4 of its peak |

* The difference falls as the square of the grid spacing, as expected for a
  second-order scheme (mode 3: 0.022 linewidths at 101 points, 0.0055 at
  201, 0.0006 at 601). Higher modes are less well resolved: mode 6 is 0.027
  linewidths off at 301 points.
* Driving the sweep from the Bessel resonance instead of the grid modes
  gives the same result: final energy 2004.6 against 2004.5, purity 1.0000,
  node error 0.007 against 0.006.
* `bessel.eigenvalue` reproduces the thesis values: 0.63337 for mode 3 at
  `a = 0.18`, and 1.7669 and 2.4674 for mode 5 at `a = 0.18` and `0`
  (thesis: 1.7668 and 2.4672).
* So for the string the Bessel solution can supply the resonance table. The
  grid modes are still used by default because the same code then serves
  the membrane, which has no Bessel solution, and coarser grids.

### What fails (`run_sweep.py --legacy`)

| Approach | What happens |
|---|---|
| Fixed frequency | The resonance moves away and the mode decays (thesis Fig. 6.2). |
| Direct substitution, `cos(w(t) t)` | The instantaneous frequency is `w + t dw/dt`, not `w`. With a smooth ramp the drive is detuned by `t dw/dt`, about 20 linewidths at the start of the default string sweep (t = 1000); with stepped updates the phase jumps by `t * (change in w)`. The mode collapses and regrows only after the tension stops changing. |
| Plain velocity feedback | Nothing selects the mode; the loop moves to mode 1. |

Also tried and dropped: an amplitude loop that regulates on the single
sensor's amplitude instead of the peak (unstable in the one setting tried).

Figures and data are in `../results/`: `<model>_sweep`, `<model>_sweep_legacy`,
`<model>_amplitude`, `<model>_rate_scan`.

## Open points

1. **A better sensor.** A pickup that is blind to mode 1 would remove the
   phase offset and probably let the amplitude loop run faster.
2. **Starting the loop.** A frequency scan to find the mode, then closing
   the loop, has not been automated.
3. **Schedules.** `--schedule smooth` and `constant-frequency-rate` exist
   but have not been compared with the linear ramp.
4. **Mode crossings on the membrane.** `run_sweep.py --model membrane`
   prints the nearest other mode along the sweep. Near `|a| = 0.18` mode
   (1, 3) comes within about 1.5 linewidths; a square membrane has exact
   degeneracies at `a = 0`.
5. **How the tension gradient is produced physically** is not modelled; the
   slope `a` is simply prescribed.

## Added scripts

- `run_paper_figures.py` – grey-scale drive-comparison and snapshot figures for one string mode (`--mode 7` for the higher mode).
- `run_make_table.py` – writes the look-up table for one membrane mode to `results/`; `--check` compares with the closed-form solution in `kummer.py` (needs `mpmath`).
