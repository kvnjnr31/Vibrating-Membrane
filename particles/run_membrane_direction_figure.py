"""Offload count against time for the two sweep directions and the control.

    python particles/run_membrane_direction_figure.py

Reads results/membrane_conveyor_long.npz (slope rising in each sweep),
membrane_conveyor_reverse.npz (slope falling) and membrane_conveyor_control6.npz
(slow sweeps both ways, drive always on).  All three are fed one grain every
6 time units.  Writes results/fig_membrane_direction.{png,pdf}.
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
    runs = [("membrane_conveyor_long", "Slope rising (lines move to the feeder)", dict(color="k", linewidth=2.4)),
            ("membrane_conveyor_control6", "Control (drive always on)", dict(color="0.45", linewidth=2.0, linestyle=(0, (4, 3)))),
            ("membrane_conveyor_reverse", "Slope falling (lines move to the slot)", dict(color="k", linewidth=2.0, linestyle=":"))]
    T = float(np.load(RESULTS / "membrane_conveyor_reverse.npz")["t"][-1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(left=0.09, right=0.98, top=0.9, bottom=0.17, wspace=0.28))
    tt = np.linspace(0, T, 500)
    for name, label, st in runs:
        d = np.load(RESULTS / f"{name}.npz")
        axes[0].plot(tt, np.searchsorted(d["exits"], tt), label=label, **st)
        k = int(np.searchsorted(d["t"], T)) - 1
        x = d["x"][k][d["state"][k] == 1]
        h, e = np.histogram(x, bins=np.linspace(float(d["back"]), float(d["offload"]), 41), density=True)
        axes[1].plot(0.5 * (e[:-1] + e[1:]), h, **st)
        print(f"{name}: offloaded {int((d['exits'] < T).sum())} of {int((d['fed'] < T).sum())} fed by time {T:.0f}; mean x {x.mean():.2f}")
    axes[0].set_xlabel("time"); axes[0].set_ylabel("grains offloaded")
    axes[0].set_xlim(0, T); axes[0].set_ylim(0, None)
    axes[0].set_title("A: Grains Offloaded")
    axes[0].legend(loc="upper left", fontsize=11.5, handlelength=2.2)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("share of grains per unit length")
    axes[1].set_ylim(0, None)
    axes[1].set_title("B: Where the Grains Are at the End")
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_membrane_direction.{ext}")


if __name__ == "__main__":
    main()
