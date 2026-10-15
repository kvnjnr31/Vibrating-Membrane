"""Summary figure for the conveyor across modes (matched protocol).

    python particles/run_conveyor_modes_figure.py --modes 5 7 9 11 13 15 19

Reads results/conveyor_hops_mode<n>_v2.csv (from run_conveyor_hops.py --tag _v2)
and writes results/fig_conveyor_modes.{png,pdf} and conveyor_modes_summary.csv.

A: transport speed against sweep rate.   B: nodes passed in one cycle.
C: peak speed against mode number.
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import style, pub_style                                  # noqa: E402
from particles.experiment import RESULTS                                     # noqa: E402

LINES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)), (0, (8, 2)), (0, (1, 2))]
MARKERS = ["o", "s", "^", "v", "D", "P", "X"]


def peak(rate, speed):
    """Peak of a parabola through the best point and its neighbours (in log rate)."""
    i = int(np.clip(np.argmax(speed), 1, len(speed) - 2))
    x, y = np.log(rate[i - 1:i + 2]), speed[i - 1:i + 2]
    c = np.polyfit(x, y, 2)
    xb = -c[1] / (2 * c[0])
    return float(np.exp(xb)), float(np.polyval(c, xb))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--modes", type=int, nargs="+", default=[5, 7, 9, 11, 13, 15, 19])
    p.add_argument("--tag", default="_v2")
    args = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pub_style()
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 19), gridspec_kw=dict(
        height_ratios=[1.2, 1.2, 1], hspace=0.3, left=0.12, right=0.78, top=0.97, bottom=0.05))
    greys = np.linspace(0.0, 0.6, len(args.modes))
    rows = []
    for k, mode in enumerate(args.modes):
        d = np.genfromtxt(RESULTS / f"conveyor_hops_mode{mode}{args.tag}.csv", delimiter=",", names=True)
        st = dict(color=str(greys[k]), linewidth=2.4, linestyle=LINES[k % len(LINES)], marker=MARKERS[k % len(MARKERS)],
                  markersize=9, markerfacecolor="white", markeredgewidth=1.8, capsize=3, elinewidth=1.2)
        speed, err = d["shift"] / d["period"], d["shift_se"] / d["period"]
        axes[0].errorbar(d["rate"], 1e4 * speed, yerr=1e4 * err, label=f"Mode {mode}", **st)
        axes[1].errorbar(d["rate"], d["hop"], yerr=d["hop_se"], **st)
        r_best, v_best = peak(d["rate"], speed)
        i = int(np.argmax(speed))
        rows.append([mode, d["spacing"][0], d["a"][0], d["amplitude"][0], d["t_gather"][0], r_best, v_best, err[i],
                     d["hop"][i], d["dist_to_node"][i], d["hop"][0], d["taken_share"].max()])
    axes[0].set_ylabel(r"transport speed ($10^{-4}$)")
    axes[0].set_title("(A) Transport Speed")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0, handlelength=2.6)
    axes[1].set_ylabel("nodes passed per cycle")
    axes[1].set_title("(B) Mean Hop in One Cycle")
    axes[1].set_xlabel(r"sweep rate $da/dt$")
    axes[1].axhline(1.0, color="0.5", linewidth=1.4, linestyle=(0, (1, 2)))
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.minorticks_off()
    rows = np.array(rows)
    ax = axes[2]
    ax.errorbar(rows[:, 0], 1e4 * rows[:, 6], yerr=1e4 * rows[:, 7], color="k", linewidth=2.6, marker="o", markersize=12,
                markerfacecolor="white", markeredgewidth=2.0, capsize=4)
    ax.set_ylabel(r"peak speed ($10^{-4}$)")
    ax.set_xlabel("mode number")
    ax.set_xticks(rows[:, 0])
    ax.set_ylim(bottom=0)
    ax.set_title("(C) Peak Speed Against Mode Number")
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_conveyor_modes.{ext}", dpi=200)
    head = ["mode", "node_spacing", "slope_range", "amplitude", "gather_time", "best_rate", "peak_speed", "peak_speed_se",
            "hop_at_peak", "dist_to_node_at_peak", "hop_at_slowest_rate", "max_taken_share"]
    with open(RESULTS / "conveyor_modes_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows([[f"{x:.5g}" for x in r] for r in rows])
    print(" ".join(head))
    for r in rows:
        print(" ".join(f"{x:.4g}" for x in r))


if __name__ == "__main__":
    main()
