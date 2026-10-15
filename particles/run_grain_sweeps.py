"""Grains on the string through a single tension sweep, at several sweep rates.

    python particles/run_grain_sweeps.py --mode 5 --a0 -0.13 --a1 0.13 --amplitude 0.029

Settles the grains once, then sweeps at each rate from the same settled
state.  Writes results/grain_sweeps_mode<n>.csv and, for the rate given by
--save-rate, results/string_grains_mode<n>.npz for the density figure.
Also runs a smooth-grain (roughness 0) case at fixed tension.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.string_model import TensionString                              # noqa: E402
from particles.surface import StringSurface                                 # noqa: E402
from particles.experiment import (settle, sweep, node_statistics, quiet_half_width,  # noqa: E402
                                  distance_to_still, RESULTS)


def grain_motion(res, lo=2.5, hi=7.5):
    """Displacement of the grains that start in the middle of the string,
    split into the part made at rest and the part made while moving."""
    t, X, nodes = res["t"], res["X"][:, :, 0], res["nodes"]
    i0 = int(np.searchsorted(t, res["t_sweep_start"])) - 1
    i1 = min(int(np.searchsorted(t, res["t_sweep_end"])), len(t) - 1)
    mid = (X[i0] > lo) & (X[i0] < hi)
    direction = np.sign(nodes[i1, nodes.shape[1] // 2] - nodes[i0, nodes.shape[1] // 2])
    Xs = X[i0:i1 + 1][:, mid]
    dx = np.diff(Xs, axis=0)
    step = t[1] - t[0]
    rest = np.abs(dx) < 0.002 * step
    return dict(node_travel=float(np.mean(np.abs(nodes[i1] - nodes[i0])[1:-1])),
                node_direction=float(direction),
                mean_displacement=float((X[i1][mid] - X[i0][mid]).mean()),
                median_displacement=float(np.median(X[i1][mid] - X[i0][mid])),
                displacement_after_hold=float((X[-1][mid] - X[i0][mid]).mean()),
                share_at_rest=float(rest.mean()),
                displacement_at_rest=float(dx[rest].sum() / mid.sum()),
                displacement_moving=float(dx[~rest].sum() / mid.sum()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, default=5)
    p.add_argument("--a0", type=float, default=-0.13)
    p.add_argument("--a1", type=float, default=0.13)
    p.add_argument("--amplitude", type=float, default=0.029)
    p.add_argument("--gamma", type=float, default=4.0)
    p.add_argument("--grains", type=int, default=500)
    p.add_argument("--rates", type=float, nargs="+", default=[3e-5, 8e-5, 3e-4, 1e-3, 3e-3])
    p.add_argument("--save-rate", type=float, default=8e-5)
    p.add_argument("--t-hold", type=float, default=500.0)
    args = p.parse_args(argv)
    model = TensionString(N=201, beta=0.05)
    surface = StringSurface(model)
    kw = dict(restitution=0.4, roughness=0.3)
    q = quiet_half_width(model, args.a0, args.mode, args.gamma)
    nodes0 = np.asarray(model.mode_nodes(args.a0, args.mode))
    print(f"quiet half-width {q:.3f}; node spacing {np.diff(nodes0).mean():.2f}", flush=True)
    s = settle(model, surface, args.mode, args.a0, drive="table+gain", gamma=args.gamma,
               amplitude=args.amplitude, n_grains=args.grains, t_settle=1000.0, dt=0.02, grain_kwargs=kw)
    rows = []
    for rate in args.rates:
        res = sweep(s, args.a1, rate, args.t_hold, verbose=False)
        m = grain_motion(res)
        mean_d, uniform_d = node_statistics(res)
        i0 = int(np.searchsorted(res["t"], res["t_sweep_start"])) - 1
        m.update(rate=rate, sweep_time=abs(args.a1 - args.a0) / rate,
                 mean_distance_start=float(mean_d[0]), mean_distance_settled=float(mean_d[i0]),
                 mean_distance_end=float(mean_d[-1]), uniform_distance=float(uniform_d[i0]))
        rows.append(m)
        print({k: round(v, 4) for k, v in m.items()}, flush=True)
        if abs(rate - args.save_rate) < 1e-12:
            np.savez_compressed(RESULTS / f"string_grains_mode{args.mode}.npz", t=res["t"], a=res["a"],
                                X=res["X"], nodes=res["nodes"], mean_distance=mean_d,
                                uniform_distance=uniform_d, quiet=q)
    keys = list(rows[0])
    with open(RESULTS / f"grain_sweeps_mode{args.mode}.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" for k in keys) + "\n")

    # smooth grains at fixed tension
    s0 = settle(model, surface, args.mode, args.a0, drive="table+gain", gamma=args.gamma,
                amplitude=args.amplitude, n_grains=args.grains, t_settle=1000.0, dt=0.02,
                grain_kwargs=dict(restitution=0.4, roughness=0.0), verbose=False)
    X = np.array(s0.rec["X"])[-1][:, 0]
    still = np.concatenate(([0.0], nodes0, [model.L]))
    d = np.min(np.abs(X[:, None] - still[None, :]), axis=1).mean()
    hist = np.histogram(X, bins=40, range=(0, model.L))[0]
    anti = 0.5 * (still[1:] + still[:-1])
    print(f"smooth grains: mean distance to a still point {d:.3f} (even scatter {rows[0]['uniform_distance']:.3f}); "
          f"largest clusters at x = {np.round(np.sort((np.argsort(hist)[-4:] + 0.5) * model.L / 40), 2)}; "
          f"antinodes at {np.round(anti, 2)}; nodes at {np.round(nodes0, 2)}", flush=True)


if __name__ == "__main__":
    main()
