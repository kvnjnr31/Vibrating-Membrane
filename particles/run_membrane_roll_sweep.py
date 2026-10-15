"""Rough rolling grains on the 2D membrane while the tension slope is swept.

    python particles/run_membrane_roll_sweep.py

Grain rules are those of the acceleration study (particles/roll_grains.py):
roughness 0.3, rolling resistance 0.05 and a peak acceleration of 4 g, the
settings under which grains collect at the nodes.  The membrane is driven
from the look-up table of forcing/run_make_table.py.

Writes results/fig_membrane_roll_sweep_mode<n><m>.{png,pdf} and a .npz.
"""
import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.sweep_experiment import make_model                              # noqa: E402
from forcing import run_make_table                                          # noqa: E402
from particles.experiment import node_statistics, RESULTS                   # noqa: E402
from particles.roll_grains import RollGrains, driven_membrane               # noqa: E402
from particles.run_membrane_grains import figure                            # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, nargs=2, default=[5, 1])
    p.add_argument("--a", type=float, default=0.131, help="the slope is swept from -a to +a")
    p.add_argument("--rate", type=float, default=1e-4)
    p.add_argument("--amplitude", type=float, default=0.029)
    p.add_argument("--gamma", type=float, default=4.0)
    p.add_argument("--roughness", type=float, default=0.3)
    p.add_argument("--mu-r", type=float, default=0.05)
    p.add_argument("--held", action="store_true", help="grains in contact are held (no rolling)")
    p.add_argument("--restitution", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--grains", type=int, default=2000)
    p.add_argument("--t-settle", type=float, default=1000.0)
    p.add_argument("--t-hold", type=float, default=500.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)
    mode = tuple(args.mode)
    model = make_model("membrane", args.beta)[0]
    ts = 2 * args.a / args.rate
    a_of_t = lambda t: -args.a + 2 * args.a * min(max((t - args.t_settle) / ts, 0.0), 1.0)   # noqa: E731
    sim, g, w, t0 = driven_membrane(model, mode, a_of_t, lambda t: 1.0, -args.a, args.amplitude, args.gamma,
                                    args.dt, table=run_make_table.table_path(mode))
    grains = RollGrains(args.grains, model, g, w, restitution=args.restitution, roughness=args.roughness,
                        mu_r=args.mu_r, roll=not args.held, seed=args.seed)
    t_end = args.t_settle + ts + args.t_hold
    every = int(round(2.0 / args.dt))
    rec_t, rec_a, rec_X, rec_on = [], [], [], []
    clock = time.time()
    for i in range(int(round(t_end / args.dt)) + 1):
        if i % every == 0:
            rec_t.append(sim.t - t0); rec_a.append(a_of_t(sim.t - t0))
            rec_X.append(np.column_stack([grains.x, grains.y]).astype(np.float32)); rec_on.append(grains.on.mean())
        sim.step()
        grains.step(sim)
    print(f"ran {t_end:.0f} time units in {time.time() - clock:.0f} s", flush=True)
    res = dict(model=model, t=np.array(rec_t), a=np.array(rec_a), X=np.array(rec_X), mode=mode,
               t_sweep_start=args.t_settle, t_sweep_end=args.t_settle + ts)
    name = f"fig_membrane_roll_sweep_mode{mode[0]}{mode[1]}{args.tag}"
    mean_d, uniform_d = figure(res, name)
    t, X = res["t"], res["X"]
    k0, k1 = int(np.searchsorted(t, args.t_settle)) - 1, int(np.searchsorted(t, args.t_settle + ts)) - 1
    xs0, xs1 = model.still_points(-args.a, mode)[0], model.still_points(args.a, mode)[0]
    mid = (X[k0][:, 1] > 0.25 * model.Ly) & (X[k0][:, 1] < 0.75 * model.Ly)
    near0 = np.argmin(np.abs(X[k0][:, 0][:, None] - xs0[None, :]), axis=1)
    inner = mid & (near0 >= 1) & (near0 <= len(xs0) - 2)
    shift = (X[k1][:, 0] - X[k0][:, 0])[inner]
    on = np.array(rec_on)
    print(f"mean distance to a nodal line or edge: start {mean_d[0]:.3f}, sweep start {mean_d[k0]:.3f}, "
          f"sweep end {mean_d[k1]:.3f}, end {mean_d[-1]:.3f} (even scatter {uniform_d[-1]:.3f})")
    print(f"nodal lines move by {np.round(xs1[1:-1] - xs0[1:-1], 2)}; grains that start at an interior line "
          f"(middle half in y): mean shift in x {shift.mean():+.3f}, median {np.median(shift):+.3f}; "
          f"share in contact at the end {on[-1]:.2f}")
    np.savez_compressed(RESULTS / f"{name}.npz", t=t, a=res["a"], mean_distance=mean_d, uniform_distance=uniform_d,
                        X=X[::10], t_X=t[::10], in_contact=on, nodes_start=xs0, nodes_end=xs1, shift=shift)
    print("saved", RESULTS / f"{name}.png")


if __name__ == "__main__":
    main()
