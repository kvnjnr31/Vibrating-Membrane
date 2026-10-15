"""Top-view snapshots of the self-regulating membrane conveyor through one cycle.

    python particles/run_conveyor_snapshots.py --d 6 --cycle 30

Reads results/adjustable_conveyor_ret_d<d>.npz and writes results/fig_conveyor_snapshots.{png,pdf}.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import style                                  # noqa: E402
from particles.experiment import RESULTS                                    # noqa: E402

import matplotlib                                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402
from matplotlib.patches import Rectangle                                     # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=float, default=6.0)
    p.add_argument("--tag", default="_ret")
    p.add_argument("--cycle", type=int, default=30)
    p.add_argument("--name", default="fig_conveyor_snapshots")
    args = p.parse_args()
    z = np.load(RESULTS / f"adjustable_conveyor{args.tag}_d{args.d:g}.npz")
    t, a, on = z["t"], z["a"], z["on"]
    X, Y, S, nodes = z["x"], z["y"], z["state"], z["nodes"]
    Lx, lane, wall, slot, box = float(z["Lx"]), z["lane"], float(z["wall"]), float(z["slot"]), z["box"]
    ret = float(z["return_slot"])
    T, (ts, fade, quiet, gather) = float(z["period"]), z["times"]
    t0 = args.cycle * T
    shots = [(t0 + 20, "Start of the sweep"),
             (t0 + 0.5 * ts, "Middle of the sweep"),
             (t0 + ts - 10, "End of the sweep (count)"),
             (t0 + ts + fade + 0.5 * quiet, "Drive off, slope reset")]
    style()
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.9), sharex=True, sharey=True,
                             gridspec_kw=dict(left=0.06, right=0.995, top=0.945, bottom=0.09, hspace=0.2, wspace=0.05))
    axes = axes.ravel()
    for ax, (tt, label), letter in zip(axes, shots, "ABCD"):
        k = int(np.argmin(np.abs(t - tt)))
        live = S[k] == 1
        ax.add_patch(Rectangle((0, 0), Lx, lane[0], color="0.9", linewidth=0))
        ax.add_patch(Rectangle((0, lane[1]), Lx, 7 - lane[1], color="0.9", linewidth=0))
        for yy in lane:
            ax.plot([wall, slot], [yy, yy], color="k", linewidth=2.0)
        ax.plot([wall + 0.03, wall + 0.03], lane, color="k", linewidth=3.0)
        ax.add_patch(Rectangle((wall + 0.03, lane[0]), ret - wall - 0.03, lane[1] - lane[0], facecolor="0.75",
                               edgecolor="none", hatch="////", zorder=0))
        ax.add_patch(Rectangle((slot, lane[0]), Lx - slot, lane[1] - lane[0], facecolor="0.75", edgecolor="none",
                               hatch="////", zorder=0))
        ax.add_patch(Rectangle((box[0], lane[0] + 0.15 * (lane[1] - lane[0])), box[1] - box[0], 0.7 * (lane[1] - lane[0]),
                               fill=False, edgecolor="0.35", linewidth=1.2, linestyle=(0, (2, 2))))
        if on[k] > 0.5:
            for xn in nodes[k][1:-1]:
                ax.plot([xn, xn], lane, color="k", linestyle=(0, (4, 3)), linewidth=1.2)
        ax.scatter(X[k][live], Y[k][live], s=4, color="0.15", linewidths=0, rasterized=True)
        if letter == "A":
            ax.text(ret / 2 + 0.05, lane[1] + 0.12, "return slot", ha="left", va="bottom", fontsize=11)
            ax.text(0.5 * (box[0] + box[1]), lane[1] + 0.12, "feeder", ha="center", va="bottom", fontsize=11)
            ax.text(Lx - 0.05, lane[1] + 0.12, "offload slot", ha="right", va="bottom", fontsize=11)
            ax.text(0.1, lane[0] - 0.12, "wall", ha="left", va="top", fontsize=11)
            xf = 0.5 * (box[0] + box[1])
            ax.annotate("", xy=(Lx, lane[0] - 0.35), xytext=(xf, lane[0] - 0.35),
                        arrowprops=dict(arrowstyle="<|-|>", color="k", linewidth=1.1))
            ax.text(0.5 * (xf + Lx), lane[0] - 0.45, f"$d = {args.d:g}$", ha="center", va="top", fontsize=12)
        n_on = int(live.sum())
        ax.set_title(f"({letter}) {label}", fontsize=15, loc="left")
        ax.text(Lx - 0.08, 6.2, f"$a = {a[k]:+.2f}$,  {n_on} grains", ha="right", va="top", fontsize=12)
        ax.set_xlim(0, Lx); ax.set_ylim(0.6, 6.3); ax.set_aspect("equal")
        if letter in "AC":
            ax.set_ylabel("$y$")
        if letter in "CD":
            ax.set_xlabel("$x$")
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"{args.name}.{ext}", dpi=200)
    print("saved", RESULTS / f"{args.name}.png")


if __name__ == "__main__":
    main()
