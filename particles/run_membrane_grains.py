"""Grains on the 2D membrane while the tension slope is swept, driven from a
look-up table written beforehand by forcing/run_make_table.py.

    python forcing/run_make_table.py --mode 3 1
    python particles/run_membrane_grains.py --mode 3 1

Writes results/fig_membrane_grains_mode<n><m>.{png,pdf} (grey scale) and a
small .npz with the statistics.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.sweep_experiment import make_model                              # noqa: E402
from forcing import run_make_table                                          # noqa: E402
from forcing.run_paper_figures import style                                 # noqa: E402
from particles.surface import MembraneSurface                               # noqa: E402
from particles.experiment import (settle, sweep, node_statistics,           # noqa: E402
                                  following_summary, RESULTS)


def figure(res, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    model, t, X, mode = res["model"], res["t"], res["X"], res["mode"]
    t0, t1 = res["t_sweep_start"], res["t_sweep_end"]
    picks = [int(np.searchsorted(t, tt)) - 1 for tt in (t0, 0.5 * (t0 + t1), t1)] + [len(t) - 1]
    fig = plt.figure(figsize=(11.0, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1], hspace=0.3, wspace=0.1,
                          left=0.075, right=0.985, top=0.94, bottom=0.115)
    for col, (k, tag) in enumerate(zip(picks, "ABCD")):
        ax = fig.add_subplot(gs[0, col])
        xs, ys = model.still_points(res["a"][k], mode)
        ax.scatter(X[k][:, 0], X[k][:, 1], s=7, color="0.45", linewidths=0, rasterized=True)
        for xn in xs[1:-1]:
            ax.plot([xn, xn], [0, model.Ly], color="k", linestyle=(0, (4, 3)), linewidth=1.6)
        for yn in ys[1:-1]:
            ax.plot([0, model.Lx], [yn, yn], color="k", linestyle=(0, (4, 3)), linewidth=1.6)
        ax.set_xlim(0, model.Lx)
        ax.set_ylim(0, model.Ly)
        ax.set_xticks([0, 4, 8])
        ax.set_aspect("equal")
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
        ax.set_title(f"{tag}: t = {t[k]:.0f}, a = {res['a'][k]:+.2f}", loc="left", fontsize=14.5)
        ax.set_xlabel("x")
        if col == 0:
            ax.set_ylabel("y")
        else:
            ax.set_yticklabels([])
    ax = fig.add_subplot(gs[1, :])
    mean_d, uniform_d = node_statistics(res)
    ax.plot(t, mean_d, color="0.45", linewidth=2.6, label="Simulated grains")
    ax.plot(t, uniform_d, color="k", linestyle=(0, (4, 3)), linewidth=1.6, label="Evenly scattered grains")
    for tt in (t0, t1):
        ax.axvline(tt, color="0.8", linewidth=0.9, zorder=0)
    ax.set_ylim(0, 1.25 * uniform_d.max())
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time")
    ax.set_ylabel("mean distance")
    ax.set_title("E: Mean Distance to the Nearest Nodal Line or Edge", loc="left")
    ax.legend(loc="upper right", ncol=2)
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"{name}.{ext}")
    plt.close(fig)
    return mean_d, uniform_d


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, nargs=2, default=[3, 1])
    p.add_argument("--a0", type=float, default=-0.15)
    p.add_argument("--a1", type=float, default=0.15)
    p.add_argument("--rate", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--grains", type=int, default=1500)
    p.add_argument("--drive", default="table+gain", choices=["table", "table+gain"])
    p.add_argument("--gamma", type=float, default=4.0)
    p.add_argument("--roughness", type=float, default=0.3)
    p.add_argument("--t-settle", type=float, default=1000.0)
    p.add_argument("--t-hold", type=float, default=500.0)
    args = p.parse_args(argv)
    mode = tuple(args.mode)
    model, _, dt, _ = make_model("membrane", args.beta)
    path = run_make_table.table_path(mode)
    if not path.exists():
        sys.exit(f"no look-up table: run  python forcing/run_make_table.py --mode {mode[0]} {mode[1]}")
    assert model.load_table(path) == mode
    print(f"look-up table {path.name} loaded")
    s = settle(model, MembraneSurface(model), mode, args.a0, drive=args.drive, gamma=args.gamma,
               n_grains=args.grains, t_settle=args.t_settle, dt=dt,
               grain_kwargs=dict(restitution=0.4, roughness=args.roughness))
    res = sweep(s, args.a1, args.rate, args.t_hold)
    name = f"fig_membrane_grains_mode{mode[0]}{mode[1]}"
    mean_d, uniform_d = figure(res, name)
    f = following_summary(res)
    print(f"mean distance: start {mean_d[0]:.3f}, sweep start "
          f"{mean_d[np.searchsorted(res['t'], res['t_sweep_start']) - 1]:.3f}, end {mean_d[-1]:.3f} "
          f"(even scatter {uniform_d[-1]:.3f})")
    print({k: (round(float(v), 3) if np.isscalar(v) else v) for k, v in f.items()})
    np.savez_compressed(RESULTS / f"{name}.npz", t=res["t"], a=res["a"], mean_distance=mean_d,
                        uniform_distance=uniform_d, X=res["X"][::25], t_X=res["t"][::25], **{k: v for k, v in f.items() if np.isscalar(v)})
    print("saved", RESULTS / f"{name}.png")


if __name__ == "__main__":
    main()
