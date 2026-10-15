"""Membrane conveyor with the state of each nodal line drawn as a path in
(tension slope a, antinode acceleration, grain count).

    python particles/run_membrane_attractors.py            # movie and still figure

Reads results/membrane_conveyor.npz (from run_membrane_conveyor.py).  No new
simulation is run.  For each nodal line:

* its zone runs from the antinode on its left to the antinode on its right
  (antinodes of the mode at the current slope, along the centre of the lane);
* grain count  = grains on the membrane inside that zone;
* acceleration = mean peak acceleration of those two antinodes, in units of
  g, from the measured range of the membrane's motion.

Writes results/membrane_conveyor_attractors.mp4,
results/fig_membrane_attractors.{png,pdf} and results/membrane_attractors.csv.
"""
import argparse
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.sweep_experiment import make_model                              # noqa: E402
from forcing import run_make_table                                          # noqa: E402
from forcing.run_paper_figures import style                                  # noqa: E402
from particles.experiment import RESULTS                                     # noqa: E402

MODE = (5, 1)


def antinode_tables(model, a_max, samples=61):
    """Antinode positions along x for slopes between -a_max and +a_max."""
    grid = np.linspace(-a_max, a_max, samples)
    jc = len(model.y) // 2
    out = []
    for a in grid:
        prof = np.abs(model.mode_shape(a, MODE)[:, jc])
        nodes = np.concatenate(([model.x[0]], np.atleast_1d(model.mode_nodes(a, MODE)), [model.x[-1]]))
        row = []
        for lo, hi in zip(nodes[:-1], nodes[1:]):
            sel = np.flatnonzero((model.x >= lo) & (model.x <= hi))
            i = sel[np.argmax(prof[sel])]
            i = min(max(i, 1), len(model.x) - 2)
            p = np.polyfit(model.x[i - 1:i + 2], prof[i - 1:i + 2], 2)
            row.append(-p[1] / (2 * p[0]))
        out.append(row)
    return grid, np.array(out)


def node_states(d, model):
    """a, acceleration and grain count of every interior nodal line, per frame."""
    model.load_table(run_make_table.table_path(MODE))
    w = model.resonance_table(MODE)
    grid, anti = antinode_tables(model, float(d["a_max"]) * 1.001)
    t, a, xg, g = d["t"], d["a"], d["xgrid"], float(d["g"])
    env_all, x_all, state_all = d["env"], d["x"], d["state"]          # read once: an .npz is unpacked on every access
    n_nodes = anti.shape[1] - 1
    acc, count, zones = np.zeros((len(t), n_nodes)), np.zeros((len(t), n_nodes), dtype=int), np.zeros((len(t), n_nodes + 1))
    for k in range(len(t)):
        pos = np.array([np.interp(a[k], grid, anti[:, j]) for j in range(n_nodes + 1)])
        zones[k] = pos
        gam = np.interp(pos, xg, env_all[k]) * w(a[k]) ** 2 / g
        x = x_all[k][state_all[k] == 1]
        for j in range(n_nodes):
            acc[k, j] = 0.5 * (gam[j] + gam[j + 1])
            count[k, j] = int(np.sum((x > pos[j]) & (x < pos[j + 1])))
    return acc, count, zones


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lines", type=int, nargs=3, default=[1, 2, 3], help="which nodal lines to show, counted from the feeder")
    p.add_argument("--tag", default="", help="read membrane_conveyor<tag>.npz and add the tag to the output names")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--still", type=int, default=None, help="write one movie frame as a PNG and stop")
    p.add_argument("--no-movie", action="store_true")
    args = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    d = np.load(RESULTS / f"membrane_conveyor{args.tag}.npz")
    model = make_model("membrane", 0.05)[0]
    acc, count, zones = node_states(d, model)
    t, a, on = d["t"], d["a"], d["on"]
    X, Y, Z, S = d["x"], d["y"], d["z"], d["state"]
    line, env, nodes, xg = d["line"], d["env"], d["nodes"], d["xgrid"]
    Lx, lane, back, off = float(d["Lx"]), d["lane"], float(d["back"]), float(d["offload"])
    box, drop, period, a_max = d["feed_box"], float(d["drop"]), float(d["period"]), float(d["a_max"])
    ts, fade, quiet, gather = d["times"]
    fed_t, exit_t = d["fed"], d["exits"]
    n_cycles = int(round(t[-1] / period))
    per = int(round(period / (t[1] - t[0])))
    sel = [j - 1 for j in args.lines]
    head = "time,a," + ",".join(f"acceleration_line{j + 1},grains_line{j + 1}" for j in range(acc.shape[1]))
    np.savetxt(RESULTS / f"membrane_attractors{args.tag}.csv", np.column_stack([t, a] + [c for j in range(acc.shape[1]) for c in (acc[:, j], count[:, j])]),
               delimiter=",", header=head, comments="", fmt="%.5g")
    zmax = [1.08 * count[:, j].max() for j in sel]
    amax = 1.08 * acc.max()

    def setup3d(ax, j, zm, fs=11):
        ax.set_xlim(-a_max, a_max); ax.set_ylim(0, amax); ax.set_zlim(0, zm)
        ax.set_xlabel("slope a", fontsize=fs, labelpad=4); ax.set_ylabel("acceleration (g)", fontsize=fs, labelpad=4)
        ax.set_zlabel("grains", fontsize=fs, labelpad=6)
        ax.set_xticks([-0.1, 0, 0.1]); ax.set_yticks([0, 2, 4])
        ax.tick_params(labelsize=fs - 1, pad=1)
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect(None, zoom=0.82)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor("white"); axis.pane.set_edgecolor("0.6")
            axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
        ax.set_title(f"Nodal Line {j + 1}", fontsize=fs + 2, fontweight="bold", pad=0)

    # ---------------------------------------------------------------- still figure for the paper
    style()
    fig = plt.figure(figsize=(11, 4.6))
    for q, (j, zm) in enumerate(zip(sel, zmax)):
        ax = fig.add_axes([-0.005 + 0.33 * q, 0.04, 0.325, 0.9], projection="3d")
        setup3d(ax, j, zm, fs=11)
        ax.patch.set_alpha(0.0)
        ax.set_title(f"{'ABC'[q]}: Nodal Line {j + 1}", fontsize=15, fontweight="bold", pad=0)
        ax.plot(a[:-per], acc[:-per, j], count[:-per, j], color="0.65", linewidth=0.8)
        ax.plot(a[-per:], acc[-per:, j], count[-per:, j], color="k", linewidth=1.8)
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_membrane_attractors{args.tag}.{ext}")
    plt.close(fig)
    for j in range(acc.shape[1]):
        c = count[:, j]
        ends = [int(c[min(q * per, len(c) - 1)]) for q in range(n_cycles + 1)]
        print(f"line {j + 1}: grains at the start of each cycle {ends}")
    if args.no_movie:
        return

    # ---------------------------------------------------------------- movie
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
    W, H = 12.8, 11.6
    fig = plt.figure(figsize=(W, H), dpi=100)
    top = 7.2 / H                                              # the upper part keeps the layout of the first movie
    ax = fig.add_axes([0.06, 1 - top + top * 0.42, 0.60, top * 0.533])
    sx = fig.add_axes([0.06, 1 - top + top * 0.10, 0.60, top * 0.24])
    for rect in (((0, 0), Lx, lane[0]), ((0, lane[1]), Lx, 7 - lane[1]), ((0, lane[0]), back, lane[1] - lane[0]),
                 ((off, lane[0]), Lx - off, lane[1] - lane[0])):
        ax.add_patch(Rectangle(*rect, color="0.88", linewidth=0))
    for yy in lane:
        ax.plot([back, off], [yy, yy], color="k", linewidth=2.2)
    ax.plot([back, back], lane, color="k", linewidth=2.2)
    ax.plot([off, off], lane, color="k", linewidth=1.6, linestyle=(0, (1, 1.5)))
    ax.add_patch(Rectangle((box[0], lane[0] + 0.15 * (lane[1] - lane[0])), box[1] - box[0], 0.7 * (lane[1] - lane[0]),
                           fill=False, edgecolor="0.3", linewidth=1.2, hatch="////"))
    ax.text(0.5 * (box[0] + box[1]), lane[1] + 0.5, "feeder", ha="center", va="bottom", fontsize=12)
    ax.text(off, lane[1] + 0.5, "offload slot", ha="center", va="bottom", fontsize=12)
    node_lines = [ax.plot([0, 0], lane, color="k", linestyle=(0, (4, 3)), linewidth=1.5)[0] for _ in range(nodes.shape[1] - 2)]
    labels = [ax.text(0, lane[1] + 0.12, str(j + 1), ha="center", va="bottom", fontsize=12, fontweight="bold") for j in range(nodes.shape[1] - 2)]
    spans = [ax.plot([0, 0], [lane[0] - 0.22 - 0.16 * (q % 2)] * 2, color="k", linewidth=3, solid_capstyle="butt")[0] for q in range(len(sel))]
    span_txt = [ax.text(0, lane[0] - 0.3 - 0.16 * (q % 2), "", ha="center", va="top", fontsize=10) for q in range(len(sel))]
    dots = ax.scatter([], [], s=9, color="0.25", linewidths=0)
    ax.set_xlim(0, Lx); ax.set_ylim(1.0, 6.0); ax.set_aspect("equal"); ax.set_xticklabels([])
    ax.set_ylabel("y"); ax.set_title("Top View", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    zlo, zhi = -0.16, drop + 0.08
    band = [sx.fill_between(xg, -env[0], env[0], color="0.82", linewidth=0)]
    prof, = sx.plot(xg, line[0], color="k", linewidth=1.3)
    sdots = sx.scatter([], [], s=9, color="0.25", linewidths=0, zorder=3)
    sx.add_patch(Polygon([[box[0] - 0.25, zhi], [box[1] + 0.25, zhi], [box[1], drop + 0.03], [box[0], drop + 0.03]],
                         closed=True, facecolor="0.7", edgecolor="k", linewidth=1.2))
    sx.plot([back, back], [-0.01, 0.07], color="k", linewidth=2.2)
    sx.plot([off, off], [zlo, 0.0], color="k", linewidth=1.6, linestyle=(0, (1, 1.5)))
    sx.set_xlim(0, Lx); sx.set_ylim(zlo, zhi); sx.set_xlabel("x"); sx.set_ylabel("height")
    sx.set_title("Side View (height stretched)", loc="left")
    tx = fig.add_axes([0.70, 1 - top + top * 0.10, 0.28, top * 0.83]); tx.axis("off")
    tx.text(0.0, 0.97, "Ratchet Conveyor on the Membrane", fontsize=15, fontweight="bold", va="top")
    tx.text(0.0, 0.90, "State of each nodal line:\nslope, antinode acceleration\nand grains held", fontsize=12, va="top", linespacing=1.4)
    txt_cycle = tx.text(0.0, 0.70, "", fontsize=14, fontweight="bold", va="top")
    txt_phase = tx.text(0.0, 0.63, "", fontsize=13, va="top", linespacing=1.4)
    txt_a = tx.text(0.0, 0.47, "", fontsize=13, va="top")
    tx.add_patch(Rectangle((0.0, 0.385), 0.9, 0.025, fill=False, edgecolor="k", linewidth=1.0, transform=tx.transAxes))
    bar = Rectangle((0.0, 0.385), 0.0, 0.025, color="0.4", linewidth=0, transform=tx.transAxes); tx.add_patch(bar)
    tx.text(0.0, 0.37, r"$-a_{\max}$", fontsize=11, va="top"); tx.text(0.9, 0.37, r"$+a_{\max}$", fontsize=11, va="top", ha="right")
    txt_n = tx.text(0.0, 0.27, "", fontsize=13, va="top", linespacing=1.6)
    ax3, old, new, pt = [], [], [], []
    for q, (j, zm) in enumerate(zip(sel, zmax)):
        b = fig.add_axes([0.01 + 0.33 * q, 0.0, 0.32, (1 - top) * 0.93], projection="3d")
        setup3d(b, j, zm)
        b.set_title("")
        b.text2D(0.5, 0.955, f"Nodal Line {j + 1}", transform=b.transAxes, ha="center", va="top", fontsize=13, fontweight="bold")
        old.append(b.plot([], [], [], color="0.65", linewidth=0.8)[0])
        new.append(b.plot([], [], [], color="k", linewidth=1.8)[0])
        pt.append(b.plot([], [], [], marker="o", color="k", markersize=6, markerfacecolor="white", markeredgewidth=1.6, linestyle="")[0])
        ax3.append(b)
    fig.text(0.5, (1 - top) * 0.955, "Black: the current cycle.  Grey: earlier cycles.  Each line's zone runs between its two neighbouring antinodes (bars under the top view).",
             ha="center", va="bottom", fontsize=11, color="0.25")

    def draw(k):
        live = S[k] == 1
        dots.set_offsets(np.column_stack([X[k][live], Y[k][live]]))
        vis = S[k] >= 1
        sdots.set_offsets(np.column_stack([X[k][vis], Z[k][vis]]))
        show = on[k] > 0.5
        for ln, lb, xn in zip(node_lines, labels, nodes[k][1:-1]):
            ln.set_xdata([xn, xn]); ln.set_visible(show); lb.set_x(xn)
        for q, j in enumerate(sel):
            lo, hi = max(zones[k, j], back), min(zones[k, j + 1], off)
            spans[q].set_xdata([lo, hi]); span_txt[q].set_x(0.5 * (lo + hi)); span_txt[q].set_text(f"zone {j + 1}: {count[k, j]}")
        band[0].remove()
        band[0] = sx.fill_between(xg, -env[k], env[k], color="0.82", linewidth=0, zorder=0)
        prof.set_ydata(line[k])
        s = t[k] % period
        c = min(int(t[k] // period) + 1, n_cycles)
        if s < ts:
            ph = "Sweep: drive on, slope rising.\nNodal lines move left."
        elif s < ts + fade + quiet:
            ph = "Reset: drive off.\nSlope returns to the start."
        else:
            ph = "Gather: drive on.\nGrains settle at the nodal lines."
        txt_cycle.set_text(f"Cycle {c} of {n_cycles}"); txt_phase.set_text(ph)
        txt_a.set_text(f"tension slope a = {a[k]:+.3f}")
        bar.set_width(0.9 * (a[k] + a_max) / (2 * a_max))
        txt_n.set_text(f"time {t[k]:.0f}\nfed {np.searchsorted(fed_t, t[k], side='right')}\n"
                       f"on the membrane {int(live.sum())}\noffloaded {np.searchsorted(exit_t, t[k], side='right')}")
        k0 = (k // per) * per
        for q, j in enumerate(sel):
            old[q].set_data_3d(a[:k0 + 1], acc[:k0 + 1, j], count[:k0 + 1, j])
            new[q].set_data_3d(a[k0:k + 1], acc[k0:k + 1, j], count[k0:k + 1, j])
            pt[q].set_data_3d([a[k]], [acc[k, j]], [count[k, j]])

    if args.still is not None:
        draw(args.still)
        fig.savefig(RESULTS / f"membrane_conveyor_attractors{args.tag}_frame.png", dpi=100)
        print("saved still")
        return
    out = RESULTS / f"membrane_conveyor_attractors{args.tag}.mp4"
    w_, h_ = fig.canvas.get_width_height()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w_}x{h_}",
                           "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27",
                           "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    n = 0
    for k in range(0, len(t), args.stride):
        draw(k)
        fig.canvas.draw()
        ff.stdin.write(fig.canvas.buffer_rgba())
        n += 1
    ff.stdin.close(); ff.wait()
    print(f"saved {out} ({n} frames, {n / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
