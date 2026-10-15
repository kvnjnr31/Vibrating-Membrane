"""Grey-scale paper version of the string grain figure, from saved data.

    python particles/run_grain_figures.py

Reads results/string_grains.npz (written by run_grains.py) and writes
results/fig_string_grains.{png,pdf}.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import pub_style                    # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
DASH = (0, (4, 3))


def main(argv=None):
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, default=None, help="read string_grains_mode<n>.npz")
    args = p.parse_args(argv)
    pub_style()
    stem = "string_grains" if args.mode is None else f"string_grains_mode{args.mode}"
    d = np.load(RESULTS / f"{stem}.npz")
    t, X, nodes, a = d["t"], d["X"][:, :, 0], d["nodes"], d["a"]
    moving = np.where(np.diff(a) != 0)[0]
    t0, t1 = t[moving[0]], t[moving[-1] + 1]
    L = 10.0
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11.5, 13.0), sharex=True, gridspec_kw=dict(
        height_ratios=[1.5, 1], hspace=0.3, left=0.12, right=0.97, top=0.955, bottom=0.075))
    bins = np.linspace(0.0, L, 101)
    dens = np.array([np.histogram(row, bins)[0] for row in X]).T / X.shape[1]
    ax.pcolormesh(t, bins, np.pad(dens, ((0, 1), (0, 0))), cmap="Greys", shading="auto",
                  vmin=0, vmax=np.percentile(dens, 99.5), rasterized=True)
    for k in range(nodes.shape[1]):
        ax.plot(t, nodes[:, k], color="k", linestyle=DASH, linewidth=1.8, label="Node" if k == 0 else None)
    ax.set_ylim(0, L)
    ax.set_ylabel("position $x$")
    ax.set_title("(A) Grain Density")
    ax.legend(loc="center right")
    bx.plot(t, d["mean_distance"], color="0.5", linewidth=4.0, label="Simulation")
    bx.plot(t, d["uniform_distance"], color="k", linestyle=DASH, linewidth=2.2, label="Even spread")
    bx.set_ylim(0, 1.25 * d["uniform_distance"].max())
    bx.set_xlim(t[0], t[-1])
    bx.set_xlabel("time $t$")
    bx.set_ylabel("mean distance")
    bx.set_title("(B) Mean Distance to the Nearest Still Point")
    bx.legend(loc="lower right", ncol=2)
    for axis in (ax, bx):
        for tt in (t0, t1):
            axis.axvline(tt, color="0.6", linewidth=1.4, zorder=3 if axis is ax else 0)
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_{stem}.{ext}", dpi=200)
    plt.close(fig)
    print(f"sweep {t0:.0f} to {t1:.0f}; saved", RESULTS / f"fig_{stem}.png")


if __name__ == "__main__":
    main()
