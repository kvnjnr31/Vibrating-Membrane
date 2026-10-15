"""Grains on a vibrating string or membrane, with a tension sweep.

Settles the grains at slope a0, sweeps the slope to a1 and holds.
Writes results/<model>_grains.{png,npz}.

    python particles/run_grains.py
    python particles/run_grains.py --model membrane
    python particles/run_grains.py --no-sweep                  # fixed tension
    python particles/run_grains.py --drive table+gain --rate 3e-4
    python particles/run_grains.py --roughness 0               # smooth grains
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.sweep_experiment import make_model                         # noqa: E402
from particles.surface import StringSurface, MembraneSurface            # noqa: E402
from particles.experiment import (settle, sweep, node_statistics,       # noqa: E402
                                  following_summary, RESULTS)

GRAIN_DEFAULTS = dict(string=dict(n=400), membrane=dict(n=1500))


def _finish(fig, res, ax_d, name, show):
    import matplotlib.pyplot as plt
    from forcing import plotstyle as ps
    t = res["t"]
    mean_d, uniform_d = node_statistics(res)
    ax_d.plot(t, uniform_d, color=ps.REFERENCE, linestyle=(0, (4, 3)), linewidth=1.4,
              label="Evenly scattered grains")
    ax_d.plot(t, mean_d, color=ps.GRAIN, label="Simulated grains")
    ax_d.set_ylim(bottom=0)
    ax_d.set_title("Mean distance from a grain to the nearest still point (node or clamped edge)")
    ax_d.set_xlabel("time")
    ax_d.legend(loc="center right", fontsize=9)
    if res["sweeping"]:
        for tt in (res["t_sweep_start"], res["t_sweep_end"]):
            ax_d.axvline(tt, color=ps.AXIS, linewidth=1.0)
        ax_d.annotate("sweep", (0.5 * (res["t_sweep_start"] + res["t_sweep_end"]), 0.0),
                      xycoords=("data", "axes fraction"), ha="center", xytext=(0, 6),
                      textcoords="offset points", fontsize=8.5, color=ps.MUTED)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.png"
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def _title(res, roughness):
    from forcing import plotstyle as ps
    return (f"mode {res['mode']}, {ps.DRIVE_LABEL[res['drive']].lower()}\n"
            f"peak acceleration {res['gamma']:g} g, roughness {roughness:g}")


def plot_string(res, roughness, name, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from forcing import plotstyle as ps
    ps.apply()
    model = res["model"]
    t, X, nodes = res["t"], res["X"][:, :, 0], res["nodes"]
    fig, (ax_m, ax_d) = plt.subplots(2, 1, figsize=(8.6, 7.4), sharex=True,
                                     gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.22,
                                                      left=0.09, right=0.9, top=0.87, bottom=0.08))
    bins = np.linspace(0.0, model.L, 101)
    dens = np.array([np.histogram(row, bins)[0] for row in X]).T / X.shape[1]
    im = ax_m.pcolormesh(t, bins, np.pad(dens, ((0, 1), (0, 0))), cmap=ps.SEQUENTIAL, shading="auto",
                         vmin=0, vmax=np.percentile(dens, 99.5), rasterized=True)
    ax_m.grid(False)
    for k in range(nodes.shape[1]):
        ax_m.plot(t, nodes[:, k], color=ps.INK, linestyle=(0, (4, 3)), linewidth=1.2,
                  label="Node of the target mode" if k == 0 else None)
    ax_m.set_ylabel("x")
    ax_m.set_title("Where the grains are (share of grains per 1% of the string)")
    ax_m.legend(loc="upper right", fontsize=9, facecolor=ps.SURFACE, frameon=True, edgecolor=ps.GRID)
    fig.colorbar(im, cax=fig.add_axes([0.915, 0.41, 0.015, 0.46]))
    fig.suptitle("Grains on the string: " + _title(res, roughness), x=0.09, y=0.985, ha="left",
                 fontsize=11.5, fontweight="bold")
    return _finish(fig, res, ax_d, name, show)


def plot_membrane(res, roughness, name, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from forcing import plotstyle as ps
    ps.apply()
    model = res["model"]
    t, X = res["t"], res["X"]
    if res["sweeping"]:
        picks = [0, int(np.searchsorted(t, res["t_sweep_start"])) - 1, len(t) - 1]
    else:
        picks = [0, len(t) // 4, len(t) - 1]
    fig = plt.figure(figsize=(10.5, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.34, wspace=0.12,
                          left=0.07, right=0.98, top=0.86, bottom=0.09)
    for col, k in enumerate(picks):
        ax = fig.add_subplot(gs[0, col])
        xs, ys = model.still_points(res["a"][k], res["mode"])
        ax.scatter(X[k][:, 0], X[k][:, 1], s=5, color=ps.GRAIN, alpha=0.6, linewidths=0)
        for xn in xs[1:-1]:
            ax.plot([xn, xn], [0, model.Ly], color=ps.INK, linestyle=(0, (4, 3)), linewidth=1.2)
        for yn in ys[1:-1]:
            ax.plot([0, model.Lx], [yn, yn], color=ps.INK, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.set_xlim(0, model.Lx)
        ax.set_ylim(0, model.Ly)
        ax.set_aspect("equal")
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
        ax.set_title(f"t = {t[k]:.0f}, slope a = {res['a'][k]:+.3f}", fontweight="normal")
        ax.set_xlabel("x")
        if col == 0:
            ax.set_ylabel("y")
        else:
            ax.set_yticklabels([])
    ax_d = fig.add_subplot(gs[1, :])
    fig.suptitle("Grains on the membrane: " + _title(res, roughness)
                 + "   (dashed: nodal lines)", x=0.07, y=0.985, ha="left", fontsize=11.5,
                 fontweight="bold")
    return _finish(fig, res, ax_d, name, show)


def add_grain_arguments(p):
    p.add_argument("--model", default="string", choices=["string", "membrane"])
    p.add_argument("--a0", type=float, default=-0.15)
    p.add_argument("--a1", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--grains", type=int, default=None)
    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=4.0, help="peak surface acceleration / g")
    p.add_argument("--restitution", type=float, default=0.4)
    p.add_argument("--roughness", type=float, default=0.3,
                   help="random sideways kick per impact, as a fraction of the impact speed")
    p.add_argument("--t-settle", type=float, default=1000.0)
    p.add_argument("--t-hold", type=float, default=500.0)
    p.add_argument("--seed", type=int, default=0)


def build(args, drive):
    """Model, surface and the settled state for the given arguments."""
    points = (201 if args.model == "string" else None)
    model, mode, dt, _ = make_model(args.model, args.beta, points=points)
    surface = StringSurface(model) if args.model == "string" else MembraneSurface(model)
    n = args.grains or GRAIN_DEFAULTS[args.model]["n"]
    return settle(model, surface, mode, args.a0, drive=drive, amplitude=args.amplitude,
                  gamma=args.gamma, n_grains=n, t_settle=args.t_settle, dt=dt, seed=args.seed,
                  grain_kwargs=dict(restitution=args.restitution, roughness=args.roughness))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_grain_arguments(p)
    p.add_argument("--rate", type=float, default=1e-4)
    p.add_argument("--drive", default="pll+agc",
                   choices=["pll", "pll+agc", "pll+gain", "table", "table+gain"])
    p.add_argument("--no-sweep", action="store_true")
    p.add_argument("--name", default=None)
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)

    settled = build(args, args.drive)
    res = sweep(settled, None if args.no_sweep else args.a1, args.rate, args.t_hold)
    mean_d, uniform_d = node_statistics(res)
    print(f"mean distance to nearest still point: start {mean_d[0]:.3f}, end {mean_d[-1]:.3f} "
          f"(evenly scattered grains: {uniform_d[-1]:.3f})")
    if res["sweeping"]:
        s = following_summary(res)
        print(f"nodes travel {s['node_travel']:.2f}. Grains that were at a node when the sweep "
              f"started trail it by {s['member_trailing_end_of_sweep']:+.3f} at the end of the sweep "
              f"and {s['member_trailing_after_hold']:+.3f} after the hold; "
              f"{100 * s['kept_after_hold']:.0f}% are still within {s['window']:.2f} of it")
        print(f"peak acceleration during the sweep: {s['acceleration_min']:.3f} to "
              f"{s['acceleration_max']:.3f} of its starting value")
    name = args.name or f"{args.model}_grains"
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / f"{name}.npz", t=res["t"], a=res["a"], X=res["X"],
                        nodes=res["nodes"], airborne=res["airborne"], w=res["w"],
                        mean_distance=mean_d, uniform_distance=uniform_d)
    plot = plot_string if args.model == "string" else plot_membrane
    print("saved", plot(res, args.roughness, name, args.show))
    return res


if __name__ == "__main__":
    main()
