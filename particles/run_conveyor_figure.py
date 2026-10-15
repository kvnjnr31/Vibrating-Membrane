"""Summary figure for the conveyor: transport against sweep rate, by mode.

    python particles/run_conveyor_figure.py --modes 3 7 11 15

Reads results/conveyor_scan_mode<n>.csv (from run_conveyor_scan.py) and
writes results/fig_conveyor_tradeoff.{png,pdf}.
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
        d = np.genfromtxt(RESULTS / f"conveyor_scan_mode{mode}.csv", delimiter=",", names=True)
        st = dict(color="k", linewidth=1.8, linestyle=LINES[k], marker=MARKERS[k], markersize=7,
                  markerfacecolor="white", markeredgewidth=1.4)
        speed = d["shift"] / d["period"]
        axes[0].plot(d["rate"], 1e4 * speed, label=f"Mode {mode}", **st)
        axes[1].plot(d["rate"], d["shift"] / d["spacing"], label=f"Mode {mode} (node spacing {d['spacing'][0]:.1f})", **st)
        i = int(np.argmax(speed))
        print(f"mode {mode}: best rate {d['rate'][i]:.2e}, peak speed {speed[i]:.2e}, shift there {d['shift'][i]:.2f} "
              f"= {d['shift'][i] / d['spacing'][i]:.2f} spacings; slowest-rate shift {d['shift'][0] / d['spacing'][0]:.2f} spacings, fastest {d['shift'][-1] / d['spacing'][-1]:.2f}; node travel {d['node_travel'][0] / d['spacing'][0]:.2f} spacings")
    axes[0].set_ylabel(r"transport speed ($10^{-4}$)")
    axes[0].set_title("A: Transport Speed")
    axes[1].set_ylabel("shift per cycle / node spacing")
    axes[1].set_title("B: Grain Shift in One Cycle")
    axes[1].set_xlabel(r"sweep rate $da/dt$ of the tension slope")
    axes[1].axhline(1.0, color="0.5", linewidth=1.0, linestyle=(0, (1, 2)))
    for ax in axes:
        ax.set_xscale("log")
        ax.axhline(0, color="0.6", linewidth=0.8)
    axes[1].legend(loc="upper right")
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_conveyor_tradeoff.{ext}")


if __name__ == "__main__":
    main()
