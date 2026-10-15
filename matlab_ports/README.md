# Python ports of the MATLAB string-bead scripts

Each `.py` file in this folder is a port of the `.m` file with the same name
from `Matlab Code 2/Membrane`, in the same folder layout. This is the
reference set: new work goes in `../forcing` and `../particles`. The numerics are kept
as they were written; nothing was "fixed" silently. Every place where the
port had to differ from the MATLAB source is marked `PORT NOTE` in the file
and listed below.

## Setup

    pip install -r ../requirements.txt     # numpy, scipy, matplotlib

## What is here

| File | MATLAB source | What it does |
|---|---|---|
| `membrane_1D.py` | `membrane_1D.m` | Main script: bead (radius 1) falling onto a string, grid-based contact |
| `centdiftt_mem.py` | `centdiftt_mem.m` | Right-hand side: `a*u_x + T(x)*u_xx` plus contact forces |
| `Runge_Kutta_mem.py` | `Runge_Kutta_mem.m` | RK4 step with fixed ends |
| `event_fun.py` | `event_fun.m` | Finds all string points inside the bead and the matching points |
| `force_fun.py` | `force_fun.m` | Reaction force from the string acceleration at the contact points |
| `correct_mem.py` | `correct_mem.m` | Corrector step (bead only; string part is multiplied by 0) |
| `contact_fun.py` | `contact_fun.m` | Line-circle intersection (matching point) |
| `circle.py`, `plot_fun.py` | same names | Plot helpers |
| `reaction_fun.py` | `reaction_fun.m` | Script: draws the reaction force as the tension slope changes (N = 5) |
| `sctratch.py` | `sctratch.m` | Scratch: an earlier analytic `force_fun` and two commented fragments |
| `mass_spring_sim/` | `mass_spring_sim/` | Self-contained analytic version: single-point `event_fun`, analytic `force_fun`, `reaction_fun` script (N = 50) |
| `old_functions/` | `old_functions/` | First, event-driven version: `membrane_1D_old`, `Bead_Membrane_Dynamics_old`, `wave_corr_old` (solves for the impact time and rewinds the string), plus penalty-force drafts |

## Running

From inside `matlab_ports/`:

    python membrane_1D.py                       # animates every step, like the MATLAB script
    python membrane_1D.py --plot-every 20       # much faster animation
    python membrane_1D.py --nsteps 3000 --no-plot
    python reaction_fun.py --frames 20 --pause 0.05
    python mass_spring_sim/reaction_fun.py
    python old_functions/membrane_1D_old.py

Regression checks against the Octave reference numbers are in
`../tests/run_checks.py`.

The functions can also be imported; the scripts expose `simulate(...)` or
`frames(...)` so they can be run without plotting.

## Conventions

* State array `u` has shape `(2, N)`: `u[0]` is velocity (MATLAB `u(1,:)`),
  `u[1]` is displacement (MATLAB `u(2,:)`).
* All indices are 0-based: `jj`, `indx_L`, `indx_R`, `j`, `k` are the MATLAB
  values minus one. `x(1:indx_L)` becomes `x[:indx_L + 1]`.
* The bead is `B = [x, y, x0, y0]`, a 1-D array.
* Square roots of negative numbers return complex values, as in MATLAB
  (for example a matching point when the string line misses the bead).

## How the ports were checked

The original `.m` files were run in GNU Octave 8.4 with plotting stubbed
out, and compared with the Python output.

* `membrane_1D`: 3000 steps. Bead position, bead velocity, string norms and
  the number of contact points agree to about 1e-12 through the first 250
  steps (which include contact with up to 43 string points). After that the
  two runs drift apart: the difference grows smoothly from round-off level,
  doubling roughly every 15 to 20 steps, and reaches order one near step 900. That
  is round-off being amplified by the simulation, not a porting difference;
  see the first item under "Things found".
* `event_fun`, `contact_fun`, `force_fun` (both versions), `tangent_fun`,
  `bead_segment_distance_old`, `event_fun_old`: 40 random cases each, agree
  to 1e-14 or exactly.
* `old_functions/membrane_1D_old`: needed three small patches to run in
  Octave at all (see below). With the same patches, all 167 steps before
  the run stops agree to 1e-15, including the impact-time solve at step 155.
* `Forces_old`, `bead_segment_force`, `sctratch` force: compared against
  patched Octave copies, agree to 1e-16.
* Not checked numerically: `plot_fun` and the drawing code (only checked
  that they run), and `old_functions/reaction_fun` (the MATLAB script
  cannot run).

## Things found in the original code

These are all reproduced, not corrected.

1. **`membrane_1D`: the bead goes through the string.** The bead reaches
   y = -2 by step 300 and is in free fall below the string afterwards
   (y = -2144 at step 3000, identical behaviour in Octave). The string-side
   correction in `correct_mem` is multiplied by zero.
2. **Grid spacing.** `dx = L/points`, but `linspace(0, L, points)` has
   spacing `L/(points-1)`. All frequencies are about 0.3 % high at 300
   points (1 % at 100).
3. **`t` is never advanced** in `membrane_1D` (it is not used by the
   right-hand side, so there is no effect yet).
4. **`force_fun.m` prints every step** (missing semicolon). The port only
   prints with `--verbose`.
5. **`reaction_fun.m` (top level) no longer runs in MATLAB**: it calls the
   old 9-argument `force_fun`, which was later rewritten with 14 arguments.
   The port uses the matching analytic version from `mass_spring_sim`.
6. **`old_functions/wave_corr_old.m`** uses `u_mag_2`, `b_u1_mag_2`, `d_2`
   and `R` without receiving them; they are explicit arguments in the port.
   Its "bead - u2" branch has the same condition as "bead - u1" and can
   never run.
7. **`old_functions/membrane_1D_old` stops at step 168.** `event_fun_old`
   reports an impact both when contact starts and when it ends; on release
   `wave_corr` has no branch to take and `tf` is undefined. The port raises
   a `RuntimeError` at the same step. The script also calls a 7-argument
   `Runge_Kutta_mem` that is not in the archive (the current integrator is
   used), with `h1 = 1/dx` where the current code has `1/(2*dx)`.
8. **`old_functions/bead_segment_force.m`** is cut off (unclosed `if`, no
   output when there is no depression). The port returns zero force there.
9. **`old_functions/Forces_old.m`** uses `u_mag_2` without defining it; the
   port computes it from the two string points.
10. **`old_functions/reaction_fun.m`** uses undefined `r` and `ClB_mag2`.
    The port uses `Rad` and `|ClB|^2`; the projection would normally divide
    by `|ClCr|^2`.
11. **`old_functions/tangent_fun.m`** names the centre `k = B(1), h = B(2)`
    but uses `(h, k)` as the centre, so x and y are swapped relative to the
    other files.
12. **`sctratch.m`**: `length(jj+1)` equals `length(jj)`; `length(jj)+1` may
    have been intended.

None of these scripts contains the forced, damped string or the tension
sweep from thesis Chapters 4 to 6; that code is not in the archive.
The forced string and membrane are implemented fresh in `../forcing`.
