"""Figure for the membrane conveyor: grains on the membrane and grains offloaded.

    python particles/run_membrane_conveyor_figure.py

Reads results/membrane_conveyor.npz and results/membrane_conveyor_control.npz
(from run_membrane_conveyor.py, the second with --control) and writes
results/fig_membrane_conveyor.{png,pdf}.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import style                                  # noqa: E402
from particles.experiment import RESULTS                                     # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    r = np.load(RESULTS / "membrane_conveyor.npz")
    c = np.load(RESULTS / "membrane_conveyor_control.npz")
    fig = plt.figure(figsize=(11, 7.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.5, 1], hspace=0.3, wspace=0.08, left=0.09, right=0.975, top=0.985, bottom=0.095)
    lane, back, off = r["lane"], float(r["back"]), float(r["offload"])
    period = float(r["period"])
    picks = [int(np.searchsorted(r["t"], q * period)) for q in (2, 6, 13)]
    for col, (k, tag) in enumerate(zip(picks, "ABC")):
        ax = fig.add_subplot(gs[0, col])
        live = r["state"][k] == 1
        ax.scatter(r["x"][k][live], r["y"][k][live], s=5, color="0.35", linewidths=0, rasterized=True)
        for xn in r["nodes"][k][1:-1]:
            ax.plot([xn, xn], lane, color="k", linestyle=(0, (4, 3)), linewidth=1.5)
        ax.plot([back, off, off], [lane[0], lane[0], lane[0]], color="k", linewidth=2)
        ax.plot([back, off], [lane[1], lane[1]], color="k", linewidth=2)
        ax.plot([back, back], lane, color="k", linewidth=2)
        ax.plot([off, off], lane, color="k", linewidth=1.5, linestyle=(0, (1, 1.5)))
        ax.set_xlim(0.6, 9.9); ax.set_ylim(lane[0] - 0.5, lane[1] + 0.5)
        ax.set_aspect("equal")
        ax.set_xticks([2, 4, 6, 8])
        ax.set_xlabel("x")
        ax.set_title(f"{tag}: Start of Cycle {int(round(r['t'][k] / period)) + 1}", fontsize=15)
        if col == 0:
            ax.set_ylabel("y")
        else:
            ax.set_yticklabels([])
    ax = fig.add_subplot(gs[1, :])
    tt = np.linspace(0, r["t"][-1], 600)
    ax.plot(tt, np.searchsorted(r["fed"], tt), color="0.6", linewidth=1.6, linestyle=":", label="Fed (ratchet)")
    ax.plot(tt, np.searchsorted(r["exits"], tt), color="k", linewidth=2.4, label="Offloaded, ratchet")
    tc = np.linspace(0, c["t"][-1], 600)
    ax.plot(tc, np.searchsorted(c["exits"], tc), color="0.4", linewidth=2.0, linestyle=(0, (4, 3)), label="Offloaded, control (drive always on)")
    ax.set_xlim(0, r["t"][-1]); ax.set_ylim(0, None)
    ax.set_xlabel("time"); ax.set_ylabel("number of grains")
    ax.set_title("D: Grains Fed and Offloaded")
    ax.legend(loc="upper left")
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_membrane_conveyor.{ext}")
    for name, d in (("ratchet", r), ("control", c)):
        T = d["t"][-1]
        late = (d["exits"] > 0.7 * T).sum() / (0.3 * T)
        print(f"{name}: period {float(d['period']):.0f}, offloaded {len(d['exits'])} of {len(d['fed'])} fed, "
              f"rate over the last 30% of the run {late:.3f} grains per time unit, median transit {np.median(d['transit']):.0f}")


if __name__ == "__main__":
    main()
