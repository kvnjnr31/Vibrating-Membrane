"""Sweep portions of the three nodal-line attractors, drawn against each other.

    python particles/run_membrane_sweep_space.py

Axes: grains in the zone of nodal line 1, of line 2 and of line 3 (each zone
runs between the line's two neighbouring antinodes).  Only the sweep part of
every cycle is drawn (drive on, slope rising); the reset and the gather are
left out.  One curve per cycle, light grey for early cycles and black for
late ones; the dot marks where each sweep starts.

A: steady feed, 30 cycles (results/membrane_attractors_long.csv).
B: feedback run, 40 cycles (results/membrane_feedback.npz and .csv).

Writes results/fig_membrane_sweep_space.{png,pdf} and membrane_sweep_space.csv.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.sweep_experiment import make_model                              # noqa: E402
from forcing.run_paper_figures import style                                  # noqa: E402
from particles.experiment import RESULTS                                     # noqa: E402
from particles.run_membrane_attractors import antinode_tables                # noqa: E402

A_MAX = 0.131


def steady():
    d = np.genfromtxt(RESULTS / "membrane_attractors_long.csv", delimiter=",", names=True)
    z = np.load(RESULTS / "membrane_conveyor_long.npz")
    period, ts = float(z["period"]), float(z["times"][0])
    cyc = (d["time"] // period).astype(int)
    sweep = (d["time"] % period) < ts
    n = np.column_stack([d[f"grains_line{j}"] for j in (1, 2, 3)])
    keep = sweep & (cyc < int(round(d["time"][-1] / period)))
    return cyc[keep] + 1, d["time"][keep], d["a"][keep], n[keep]


def feedback():
    z = np.load(RESULTS / "membrane_feedback.npz")
    t, a, X, S = z["frame_t"], z["frame_a"], z["frame_x"], z["frame_state"]
    model = make_model("membrane", 0.05)[0]
    grid, anti = antinode_tables(model, A_MAX * 1.001)
    n = np.zeros((len(t), 3), dtype=int)
    for k in range(len(t)):
        pos = np.array([np.interp(a[k], grid, anti[:, j]) for j in range(4)])
        x = X[k][S[k] == 1]
        for j in range(3):
            n[k, j] = int(np.sum((x > pos[j]) & (x < pos[j + 1])))
    cyc = np.searchsorted(z["t_start"], t, side="right")
    ts = 2 * A_MAX / z["sweep_rate"][np.clip(cyc - 1, 0, len(z["t_start"]) - 1)]
    keep = (cyc >= 1) & (t - z["t_start"][np.clip(cyc - 1, 0, len(z["t_start"]) - 1)] < ts)
    return cyc[keep], t[keep], a[keep], n[keep]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    fig = plt.figure(figsize=(11, 6.2))
    rows = []
    for q, (name, title, data) in enumerate((("steady", "A: Steady Feed, 30 Cycles", steady()),
                                             ("feedback", "B: Feedback, 40 Cycles", feedback()))):
        cyc, t, a, n = data
        ax = fig.add_axes([-0.01 + 0.5 * q, 0.02, 0.5, 0.88], projection="3d")
        ax.patch.set_alpha(0.0)
        last = cyc.max()
        for c in np.unique(cyc):
            m = cyc == c
            shade = str(0.78 * (1 - (c - 1) / max(last - 1, 1)))
            ax.plot(n[m, 0], n[m, 1], n[m, 2], color=shade, linewidth=1.0 + 1.1 * (c == last))
            ax.scatter([n[m, 0][0]], [n[m, 1][0]], [n[m, 2][0]], color=shade, s=16, edgecolors="k", linewidths=0.5, depthshade=False)
        ax.set_xlabel("grains at line 1", labelpad=9, fontsize=13)
        ax.set_ylabel("grains at line 2", labelpad=9, fontsize=13)
        ax.set_zlabel("grains at line 3", labelpad=9, fontsize=13)
        ax.tick_params(labelsize=11)
        ax.view_init(elev=22, azim=-60)
        ax.set_box_aspect(None, zoom=0.84)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor("white"); axis.pane.set_edgecolor("0.6")
            axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
        fig.text(0.25 + 0.5 * q, 0.93, title, ha="center", fontsize=17, fontweight="bold")
        rows += [(q, c_, t_, a_, *n_) for c_, t_, a_, n_ in zip(cyc, t, a, n)]
        m = cyc == last
        print(f"{name}: last sweep runs from {tuple(n[m][0])} to {tuple(n[m][-1])} (lines 1, 2, 3); sum {n[m][0].sum()} -> {n[m][-1].sum()}")
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_membrane_sweep_space.{ext}")
    np.savetxt(RESULTS / "membrane_sweep_space.csv", np.array(rows), delimiter=",", comments="", fmt="%.6g",
               header="run(0=steady;1=feedback),cycle,time,a,grains_line1,grains_line2,grains_line3")


if __name__ == "__main__":
    main()
