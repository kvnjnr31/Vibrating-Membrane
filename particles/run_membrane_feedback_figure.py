"""Figure for the feedback conveyor: path in (input rate, sweep rate, output rate).

    python particles/run_membrane_feedback_figure.py

Reads results/membrane_feedback.csv (one row per cycle, from
run_membrane_feedback.py) and writes results/fig_membrane_feedback.{png,pdf}.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import style                                  # noqa: E402
from particles.experiment import RESULTS                                     # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", default="")
    p.add_argument("--target", type=float, default=None)
    args = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    d = np.genfromtxt(RESULTS / f"membrane_feedback{args.tag}.csv", delimiter=",", names=True)
    z = np.load(RESULTS / f"membrane_feedback{args.tag}.npz")
    target = float(z["target"]) if args.target is None else args.target
    fin, fout, r, n = 1e2 * d["input_rate"], 1e2 * d["output_rate"], 1e4 * d["sweep_rate"], d["grains_end"]
    t_end = d["t_start"] + d["period"]
    fig = plt.figure(figsize=(11, 9.4))
    ax = fig.add_axes([0.0, 0.33, 0.60, 0.66], projection="3d")
    ax.plot(fin, r, fout, color="0.55", linewidth=1.2)
    ax.scatter(fin, r, fout, c=np.linspace(0.75, 0.0, len(fin)).astype(str), s=34, edgecolors="k", linewidths=0.7, depthshade=False)
    ax.scatter([fin[0]], [r[0]], [fout[0]], marker="s", s=90, facecolor="white", edgecolor="k", linewidths=1.6, depthshade=False)
    ax.scatter([fin[-1]], [r[-1]], [fout[-1]], marker="*", s=330, facecolor="k", edgecolor="k", depthshade=False)
    ax.text(fin[0], r[0], fout[0], "  start", fontsize=13)
    ax.text(fin[-1] + 2.5, r[-1], fout[-1] + 1.2, "end", fontsize=13)
    ax.set_xlabel(r"input rate ($10^{-2}$)", labelpad=10, fontsize=14)
    ax.set_ylabel(r"sweep rate ($10^{-4}$)", labelpad=10, fontsize=14)
    ax.set_zlabel(r"output rate ($10^{-2}$)", labelpad=10, fontsize=14)
    ax.tick_params(labelsize=13)
    ax.view_init(elev=24, azim=-52)
    ax.set_box_aspect(None, zoom=0.86)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white"); axis.pane.set_edgecolor("0.6")
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
    fig.text(0.065, 0.955, "A: Path of the Conveyor Under Feedback", fontsize=18, fontweight="bold")
    bx = fig.add_axes([0.685, 0.60, 0.29, 0.31])
    bx.plot(fin, fout, color="0.55", linewidth=1.2)
    bx.scatter(fin, fout, c=np.linspace(0.75, 0.0, len(fin)).astype(str), s=30, edgecolors="k", linewidths=0.7, zorder=3)
    lim = 1.08 * max(fin.max(), fout.max())
    bx.plot([0, lim], [0, lim], color="k", linestyle=(0, (4, 3)), linewidth=1.2)
    bx.set_xlim(0, lim); bx.set_ylim(0, lim)
    bx.set_xlabel(r"input rate ($10^{-2}$)", fontsize=14); bx.set_ylabel(r"output rate ($10^{-2}$)", fontsize=14)
    bx.tick_params(labelsize=13)
    bx.set_title("B: Output Against Input", fontsize=16)
    cx = fig.add_axes([0.09, 0.075, 0.38, 0.235])
    cx.plot(d["cycle"], n, color="k", linewidth=2.0, marker="o", markersize=4.5, markerfacecolor="white")
    cx.axhline(target, color="k", linestyle=(0, (4, 3)), linewidth=1.2)
    cx.set_xlabel("cycle", fontsize=14); cx.set_ylabel("grains on the conveyor", fontsize=14); cx.tick_params(labelsize=13)
    cx.set_ylim(0, None); cx.set_xlim(0, d["cycle"][-1] + 0.5)
    cx.set_title("C: Grains on the Conveyor", fontsize=16)
    dx = fig.add_axes([0.585, 0.075, 0.335, 0.235])
    dx.plot(d["cycle"], fin, color="k", linewidth=2.0, label="Input")
    dx.plot(d["cycle"], fout, color="0.45", linewidth=2.0, linestyle=(0, (4, 3)), label="Output")
    dx.set_xlabel("cycle", fontsize=14); dx.set_ylabel(r"rate ($10^{-2}$)", fontsize=14); dx.tick_params(labelsize=13)
    dx.set_ylim(0, None); dx.set_xlim(0, d["cycle"][-1] + 0.5)
    dx2 = dx.twinx()
    dx2.plot(d["cycle"], r, color="k", linewidth=1.6, linestyle=":", label="Sweep rate (right)")
    dx2.set_ylabel(r"sweep rate ($10^{-4}$)", fontsize=14); dx2.set_ylim(0, None); dx2.tick_params(labelsize=13)
    dx2.spines["right"].set_visible(True)
    h1, l1 = dx.get_legend_handles_labels(); h2, l2 = dx2.get_legend_handles_labels()
    dx.set_ylim(0, 1.45 * max(fin.max(), fout.max())); dx2.set_ylim(0, 1.45 * r.max())
    dx.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=11, ncol=3, columnspacing=0.8, handlelength=1.3, handletextpad=0.4, borderaxespad=0.1)
    dx.set_title("D: Rates", fontsize=16)
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_membrane_feedback{args.tag}.{ext}")
    k = slice(-5, None)
    print(f"last five cycles: grains {n[k].mean():.0f} (target {target:.0f}), input {d['input_rate'][k].mean():.3f}, "
          f"output {d['output_rate'][k].mean():.3f}, sweep rate {d['sweep_rate'][k].mean():.2e}")
    print("grains at the end of each cycle:", n.astype(int).tolist())


if __name__ == "__main__":
    main()
