"""Figure for the bifurcation scan of the reduced conveyor model.

    python particles/run_model_bifurcation_figure.py

Reads results/model_bifurcation.npz and writes results/fig_model_bifurcation.{png,pdf}.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import style                                  # noqa: E402
from particles.experiment import RESULTS                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--mark", type=float, nargs="*", default=[2, 4, 6, 8], help="feeder positions run in the simulation")
    args = ap.parse_args()
    z = np.load(RESULTS / f"model_bifurcation{args.tag}.npz")
    d, N, S, per = z["d"], z["N"], z["s"], z["period"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    fig, axes = plt.subplots(4, 1, figsize=(12, 15.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.6, 1, 1, 1], left=0.09, right=0.955, top=0.965, bottom=0.05, hspace=0.25))
    ax = axes[0]
    for k in range(len(d)):
        on = S[k] > 0.5
        ax.scatter(np.full(on.sum(), d[k]), N[k][on], s=7, color="k", linewidths=0, zorder=2)
        ax.scatter(np.full((~on).sum(), d[k]), N[k][~on], s=9, facecolors="white", edgecolors="0.35", linewidths=0.7, zorder=2)
    ax.axhline(3000, color="0.5", linewidth=1.0, linestyle=":")
    ax.scatter([], [], s=20, color="k", label="feeder set on")
    ax.scatter([], [], s=20, facecolors="white", edgecolors="0.35", label="feeder set off")
    ax.set_ylabel("count at the end of each sweep")
    ax.set_title("(A) Counts at the Last 120 Sweep Ends", fontsize=15, loc="left")
    ax.legend(loc="lower left", fontsize=11, ncol=2)
    ax = axes[1]
    rep = per > 0
    ax.plot(d[rep], per[rep], linestyle="none", marker="o", markersize=6, color="k", label="pattern repeats")
    if (~rep).any():
        top = max(per.max(), 10) * 1.15
        ax.plot(d[~rep], np.full((~rep).sum(), top), linestyle="none", marker="x", markersize=7, color="0.4",
                label="no repeat within 60 cycles")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 5, 10, 20, 40]); ax.set_yticklabels(["1", "2", "3", "5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_ylabel("repeat length (cycles)")
    ax.set_title("(B) Repeat Length of the On/Off Pattern", fontsize=15, loc="left")
    ax.legend(loc="upper left", fontsize=11)
    ax = axes[2]
    ax.plot(d, z["on"], color="k", linewidth=0, marker="o", markersize=4)
    ax.step(d, z["on"], where="mid", color="k", linewidth=1.0)
    for frac, lab in ((1.0, "1"), (2 / 3, "2/3"), (1 / 2, "1/2"), (2 / 5, "2/5")):
        ax.axhline(frac, color="0.8", linewidth=0.8, linestyle=":")
        ax.text(d.max() + 0.12, frac, lab, va="center", fontsize=11, color="0.4")
    ax.set_ylabel("share of counts\nwith the feeder on")
    ax.set_title("(C) Share of Sweeps That Set the Feeder On", fontsize=15, loc="left")
    ax = axes[3]
    ax.plot(d, z["off"], color="k", linewidth=1.6, label="offloaded")
    ax.plot(d, z["ret"], color="0.5", linewidth=1.6, linestyle=(0, (4, 3)), label="returned")
    ax.set_ylabel("grains per cycle")
    ax.set_xlabel("feeder distance from the offload, $d$")
    ax.set_title("(D) Grains Offloaded and Returned per Cycle", fontsize=15, loc="left")
    ax.legend(loc="center left", fontsize=11)
    for a in axes:
        for m in args.mark:
            a.axvline(m, color="0.85", linewidth=1.0, zorder=0)
    axes[-1].set_xlim(d.min() - 0.1, d.max() + 0.1)
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_model_bifurcation{args.tag}.{ext}", dpi=200)
    print("saved", RESULTS / f"fig_model_bifurcation{args.tag}.png")


if __name__ == "__main__":
    main()
