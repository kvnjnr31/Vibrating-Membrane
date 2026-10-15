"""Summary figure for the conveyor, from hops between nodes.

    python particles/run_conveyor_hops_figure.py --modes 3 7 11 15

Reads results/conveyor_hops_mode<n>.csv (from run_conveyor_hops.py) and
writes results/fig_conveyor_hops.{png,pdf}.  Error bars are the standard
error over the measured grains.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import style                                  # noqa: E402
from particles.experiment import RESULTS                                     # noqa: E402

LINES = [":", "--", "-.", "-", (0, (5, 1, 1, 1, 1, 1))]
MARKERS = ["v", "s", "^", "o", "D"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--modes", type=int, nargs="+", default=[3, 7, 11, 15])
    args = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    fig, axes = plt.subplots(2, 1, figsize=(11, 10.4), sharex=True, gridspec_kw=dict(
        hspace=0.22, left=0.11, right=0.97, top=0.95, bottom=0.08))
    for k, mode in enumerate(args.modes):
        d = np.genfromtxt(RESULTS / f"conveyor_hops_mode{mode}.csv", delimiter=",", names=True)
        st = dict(color="k", linewidth=1.8, linestyle=LINES[k], marker=MARKERS[k], markersize=7,
                  markerfacecolor="white", markeredgewidth=1.4, capsize=2.5, elinewidth=1.0)
        speed, err = d["shift"] / d["period"], d["shift_se"] / d["period"]
        axes[0].errorbar(d["rate"], 1e4 * speed, yerr=1e4 * err, **st)
        axes[1].errorbar(d["rate"], d["hop"], yerr=d["hop_se"], label=f"Mode {mode} (node spacing {d['spacing'][0]:.1f})", **st)
        i = int(np.argmax(speed))
        lr = np.log(d["rate"])
        ok = speed > 0.2 * speed.max()
        res = np.std(np.log(speed[ok]) - np.polyval(np.polyfit(lr[ok], np.log(speed[ok]), 4), lr[ok]))
        print(f"mode {mode}: best rate {d['rate'][i]:.2e}, peak speed {speed[i]:.2e} +- {err[i]:.1e}, hop there {d['hop'][i]:.2f}; "
              f"hop at slowest {d['hop'][0]:.2f}, fastest {d['hop'][-1]:.2f}; scatter about a smooth curve {100 * res:.1f}%, "
              f"mean error bar {100 * np.mean((err / speed)[ok]):.1f}%, taken off at most {d['taken_share'].max():.2f}")
    axes[0].set_ylabel(r"transport speed ($10^{-4}$)")
    axes[0].set_title("A: Transport Speed")
    axes[1].set_ylabel("nodes passed per cycle")
    axes[1].set_title("B: Mean Hop in One Cycle")
    axes[1].set_xlabel(r"sweep rate $da/dt$ of the tension slope")
    axes[1].axhline(1.0, color="0.5", linewidth=1.0, linestyle=(0, (1, 2)))
    for ax in axes:
        ax.set_xscale("log")
        ax.axhline(0, color="0.6", linewidth=0.8)
    axes[1].legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_conveyor_hops.{ext}")


if __name__ == "__main__":
    main()
