"""Movie of the membrane conveyor: top view and side view.

    python particles/run_membrane_movie.py                 # results/membrane_conveyor.mp4

Reads results/membrane_conveyor<tag>.npz written by run_membrane_conveyor.py.
Top panel: the membrane seen from above, with the lane, the feeder, the
offload slot, the nodal lines (dashed, shown while the drive is on) and the
grains.  Bottom panel: the same seen from the side, with the height
stretched; the grey band is the range of the membrane's motion along the
centre of the lane and the line is its shape at that instant.
"""
import argparse
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import style                                  # noqa: E402
from particles.experiment import RESULTS                                     # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", default="")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--stride", type=int, default=1, help="use every n-th saved frame")
    p.add_argument("--frames", type=int, nargs=2, default=None, help="first and last frame (for a test)")
    p.add_argument("--no-slow", action="store_true", help="leave out the slow-motion part")
    p.add_argument("--still", type=int, default=None, help="write one frame as a PNG and stop")
    args = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    style()
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
                         "ytick.labelsize": 11})
    d = np.load(RESULTS / f"membrane_conveyor{args.tag}.npz")
    t, a, on = d["t"], d["a"], d["on"]
    X, Y, Z, S = d["x"], d["y"], d["z"], d["state"]
    line, env, nodes, xg = d["line"], d["env"], d["nodes"], d["xgrid"]
    Lx, Ly, lane, back, off = float(d["Lx"]), float(d["Ly"]), d["lane"], float(d["back"]), float(d["offload"])
    box, drop, period, amp = d["feed_box"], float(d["drop"]), float(d["period"]), float(d["amplitude"])
    ts, fade, quiet, gather = d["times"]
    fed_t, exit_t = d["fed"], d["exits"]
    has_slow = "slow_t" in d.files and not args.no_slow
    if has_slow:
        slow = {k: d["slow_" + k] for k in ("t", "a", "on", "x", "y", "z", "state", "line", "env", "nodes")}
        k_slow = int(np.searchsorted(t, slow["t"][0]))             # the slow part is shown before this frame
        factor = (t[1] - t[0]) / (slow["t"][1] - slow["t"][0])
    n_cycles = int(round((t[-1]) / period))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([0.06, 0.42, 0.60, 0.533])
    sx = fig.add_axes([0.06, 0.09, 0.60, 0.25])
    # ---- top view
    ax.add_patch(Rectangle((0, 0), Lx, lane[0], color="0.88", linewidth=0))
    ax.add_patch(Rectangle((0, lane[1]), Lx, Ly - lane[1], color="0.88", linewidth=0))
    ax.add_patch(Rectangle((0, lane[0]), back, lane[1] - lane[0], color="0.88", linewidth=0))
    ax.add_patch(Rectangle((off, lane[0]), Lx - off, lane[1] - lane[0], color="0.88", linewidth=0))
    for yy in lane:
        ax.plot([back, off], [yy, yy], color="k", linewidth=2.2)
    ax.plot([back, back], lane, color="k", linewidth=2.2)
    ax.plot([off, off], lane, color="k", linewidth=1.6, linestyle=(0, (1, 1.5)))
    ax.add_patch(Rectangle((box[0], lane[0] + 0.15 * (lane[1] - lane[0])), box[1] - box[0], 0.7 * (lane[1] - lane[0]),
                           fill=False, edgecolor="0.3", linewidth=1.2, hatch="////"))
    ax.text(0.5 * (box[0] + box[1]), lane[1] + 0.18, "feeder", ha="center", va="bottom", fontsize=12)
    ax.text(off, lane[1] + 0.18, "offload slot", ha="center", va="bottom", fontsize=12)
    ax.annotate("", xy=(off - 2.2, lane[0] - 0.4), xytext=(back + 2.2, lane[0] - 0.4),
                arrowprops=dict(arrowstyle="-|>", color="k", linewidth=1.6))
    ax.text(back + 0.2, lane[0] - 0.4, "grains move", ha="left", va="center", fontsize=12)
    node_lines = [ax.plot([0, 0], lane, color="k", linestyle=(0, (4, 3)), linewidth=1.5)[0] for _ in range(nodes.shape[1] - 2)]
    dots = ax.scatter([], [], s=9, color="0.25", linewidths=0)
    ax.set_xlim(0, Lx); ax.set_ylim(1.0, 6.0); ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_ylabel("y"); ax.set_title("Top View", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    # ---- side view
    zlo, zhi = -0.16, drop + 0.08
    band = [sx.fill_between(xg, -env[0], env[0], color="0.82", linewidth=0)]
    prof, = sx.plot(xg, line[0], color="k", linewidth=1.3)
    sdots = sx.scatter([], [], s=9, color="0.25", linewidths=0, zorder=3)
    sx.add_patch(Polygon([[box[0] - 0.25, zhi], [box[1] + 0.25, zhi], [box[1], drop + 0.03], [box[0], drop + 0.03]],
                         closed=True, facecolor="0.7", edgecolor="k", linewidth=1.2))
    sx.plot([back, back], [-0.01, 0.07], color="k", linewidth=2.2)
    sx.plot([off, off], [zlo, 0.0], color="k", linewidth=1.6, linestyle=(0, (1, 1.5)))
    sx.set_xlim(0, Lx); sx.set_ylim(zlo, zhi)
    sx.set_xlabel("x"); sx.set_ylabel("height")
    w_in, h_in = 12.8 * 0.60, 7.2 * 0.25
    stretch = (h_in / (zhi - zlo)) / (w_in / Lx)
    sx.set_title(f"Side View (height stretched {stretch:.0f} times)", loc="left")
    # ---- status panel
    tx = fig.add_axes([0.70, 0.085, 0.28, 0.815]); tx.axis("off")
    tx.text(0.0, 0.97, "Ratchet Conveyor on the Membrane", fontsize=15, fontweight="bold", va="top")
    tx.text(0.0, 0.90, "Mode (5,1), rough rolling grains,\npeak acceleration 4 g", fontsize=12, va="top", linespacing=1.4)
    txt_cycle = tx.text(0.0, 0.74, "", fontsize=14, fontweight="bold", va="top")
    txt_phase = tx.text(0.0, 0.67, "", fontsize=13, va="top", linespacing=1.4)
    txt_a = tx.text(0.0, 0.50, "", fontsize=13, va="top")
    tx.add_patch(Rectangle((0.0, 0.415), 0.9, 0.025, fill=False, edgecolor="k", linewidth=1.0, transform=tx.transAxes))
    bar = Rectangle((0.0, 0.415), 0.0, 0.025, color="0.4", linewidth=0, transform=tx.transAxes); tx.add_patch(bar)
    tx.text(0.0, 0.40, r"$-a_{\max}$", fontsize=11, va="top"); tx.text(0.9, 0.40, r"$+a_{\max}$", fontsize=11, va="top", ha="right")
    txt_n = tx.text(0.0, 0.32, "", fontsize=13, va="top", linespacing=1.6)
    tx.text(0.0, 0.055, "Dashed lines: nodal lines (drive on).\nGrey band: range of membrane motion.", fontsize=11,
            va="top", color="0.25", linespacing=1.4)
    a_max = float(d["a_max"])
    full = (t, a, on, X, Y, Z, S, line, env, nodes)

    txt_slow = ax.text(0.5, 0.955, "", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="center",
                       color="k", zorder=5, bbox=dict(facecolor="white", edgecolor="k", boxstyle="round,pad=0.35"))

    def draw(k, src=None):
        t, a, on, X, Y, Z, S, line, env, nodes = src if src is not None else full
        txt_slow.set_text(f"Slow motion ({factor:.0f} times slower)" if src is not None else "")
        txt_slow.set_visible(src is not None)
        live = S[k] == 1
        dots.set_offsets(np.column_stack([X[k][live], Y[k][live]]))
        vis = S[k] >= 1
        sdots.set_offsets(np.column_stack([X[k][vis], Z[k][vis]]))
        show = on[k] > 0.5
        for ln, xn in zip(node_lines, nodes[k][1:-1]):
            ln.set_xdata([xn, xn]); ln.set_visible(show)
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
        txt_cycle.set_text(f"Cycle {c} of {n_cycles}")
        txt_phase.set_text(ph)
        txt_a.set_text(f"tension slope a = {a[k]:+.3f}")
        bar.set_width(0.9 * (a[k] + a_max) / (2 * a_max))
        txt_n.set_text(f"time {t[k]:.0f}\nfed {np.searchsorted(fed_t, t[k], side='right')}\n"
                       f"on the membrane {int(live.sum())}\noffloaded {np.searchsorted(exit_t, t[k], side='right')}")

    if args.still is not None:
        draw(args.still)
        fig.savefig(RESULTS / f"membrane_conveyor{args.tag}_frame.png")
        print("saved still")
        return
    k0, k1 = args.frames if args.frames else (0, len(t) - 1)
    out = RESULTS / f"membrane_conveyor{args.tag}.mp4"
    w, h = fig.canvas.get_width_height()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
                           "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27",
                           "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    n_out = 0
    for k in range(k0, k1 + 1, args.stride):
        if has_slow and k == k_slow:
            src = tuple(slow[q] for q in ("t", "a", "on", "x", "y", "z", "state", "line", "env", "nodes"))
            for q in range(len(slow["t"])):
                draw(q, src)
                fig.canvas.draw()
                ff.stdin.write(fig.canvas.buffer_rgba())
                n_out += 1
        draw(k)
        fig.canvas.draw()
        ff.stdin.write(fig.canvas.buffer_rgba())
        n_out += 1
    ff.stdin.close(); ff.wait()
    print(f"saved {out} ({n_out} frames, {n_out / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
