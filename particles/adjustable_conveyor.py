"""The adjustable conveyor: a membrane ratchet conveyor whose feeder is
switched by the number of grains on the conveyor.

    python particles/adjustable_conveyor.py run --d 8        # one simulation
    python particles/adjustable_conveyor.py figure --d 8 6 4 # feed rate against time, one panel per d
    python particles/adjustable_conveyor.py movie --d 8      # top view, side view and feed rate
    python particles/adjustable_conveyor.py embed --d 8 6 4  # delay embedding of the grain count

Setup (membrane units: length 10, width 7; mode (5,1); peak acceleration 4 g;
rough rolling grains as in particles/roll_grains.py):

* back wall at the left edge, offload slot at the right edge, side rails at
  y = --lane;
* the feeder is a box --feed-width wide whose centre is a distance d from the
  right edge; it drops grains from a height --drop;
* each cycle is a ratchet cycle: sweep (slope rises, drive on), drive off and
  slope reset, drive on and gather;
* at the end of every sweep the grains on the conveyor are counted.  Below
  --target the feeder is commanded on, otherwise off;
* the feed rate does not jump.  With --valve smooth (default) it follows the
  command like a critically damped valve,
      dF/dt = G,   dG/dt = (F_max u - F - 2 tau_f G) / tau_f^2,
  with tau_f = --valve-tau.  With --valve ramp it moves toward the command at
  --feed-accel until it reaches --feed-max (full on) or zero (off);
* with --return-slot x_r > 0, a grain that crosses x = x_r near the left
  edge falls into a return chute and goes back to the feeder hopper; the
  feeder drops returned grains first and fresh grains only when the hopper
  is empty.  With --return-slot 0 the left edge is a wall, as in the first
  runs.

The feed rate is saved at every time step.
Outputs: results/adjustable_conveyor_d<d>.npz, fig_adjustable_conveyor.{png,pdf},
adjustable_conveyor_d<d>.mp4.
"""
import argparse
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.sweep_experiment import make_model                              # noqa: E402
from forcing import run_make_table                                          # noqa: E402
from forcing.run_paper_figures import style                                  # noqa: E402
from particles.experiment import RESULTS                                    # noqa: E402

MODE = (5, 1)


TAG = ""


def name(d):
    return f"adjustable_conveyor{TAG}_d{d:g}"


class Cycle:
    """Repeating ratchet cycle: sweep, fade out, quiet (slope reset), fade in, gather."""

    def __init__(self, a_max, rate, fade, quiet, gather):
        self.a_max, self.fade, self.quiet, self.gather = a_max, fade, quiet, gather
        self.ts = 2 * a_max / rate
        self.period = self.ts + 2 * fade + quiet + gather

    def a(self, t):
        s = t % self.period if t > 0 else 0.0
        if s < self.ts:
            return -self.a_max + 2 * self.a_max * s / self.ts
        q = (s - self.ts - self.fade - 0.5 * self.quiet) / (0.4 * self.quiet)
        return self.a_max - 2 * self.a_max * min(max(q, 0.0), 1.0)

    def env(self, t):
        s = t % self.period if t > 0 else 0.0
        if s < self.ts:
            return 1.0
        if s < self.ts + self.fade:
            return 1.0 - (s - self.ts) / self.fade
        if s < self.ts + self.fade + self.quiet:
            return 0.0
        return min((s - self.ts - self.fade - self.quiet) / self.fade, 1.0)


# ============================================================================ run
def run(args):
    from particles.roll_grains import RollGrains, driven_membrane
    d = args.d[0]
    model = make_model("membrane", args.beta)[0]
    cyc = Cycle(args.a, args.rate, args.fade, args.quiet, args.gather)
    sim, g, w, t0 = driven_membrane(model, MODE, cyc.a, cyc.env, -args.a, args.amplitude, args.gamma, args.dt,
                                    table=run_make_table.table_path(MODE))
    G = RollGrains(args.pool, model, g, w, restitution=args.restitution, roughness=args.roughness, mu_r=args.mu_r, seed=args.seed)
    wall, slot = 0.02, model.Lx - 0.15
    G.xlo, G.ylo, G.yhi = wall, args.lane[0], args.lane[1]
    G.alive[:] = False
    box = (model.Lx - d - 0.5 * args.feed_width, model.Lx - d + 0.5 * args.feed_width)
    rng = np.random.default_rng(100 + args.seed)
    state = np.zeros(args.pool, dtype=np.int8)                 # 0 = spare, 1 = on the conveyor, 2 = falling away
    steps = int(round(args.cycles * cyc.period / args.dt))
    per_n, sweep_n = int(round(cyc.period / args.dt)), int(round(cyc.ts / args.dt))
    frame_n = int(round(args.frame_every / args.dt))
    osc_n = int(round(2 * np.pi / w / args.dt))
    jc = len(model.y) // 2
    feed_rate = np.zeros(steps, dtype=np.float32)              # sampled at every time step
    count = np.zeros(steps, dtype=np.int32)
    command = np.zeros(steps, dtype=np.int8)
    F = dict(t=[], a=[], on=[], x=[], y=[], z=[], state=[], line=[], env=[])
    rate, rate_v, cmd_on, credit, blocked = 0.0, 0.0, True, 0.0, 0
    fed_t, fed_returned, exit_t, return_t, measured, recent = [], [], [], [], [], []
    hopper = np.zeros(steps, dtype=np.int32)
    clock = time.time()
    for i in range(steps):
        t = sim.t - t0
        if i % per_n == sweep_n:                               # the sweep has just ended: count and decide
            n_on = int((state == 1).sum())
            cmd_on = n_on < args.target
            measured.append((t, n_on, int(cmd_on)))
        if args.valve == "smooth":
            tau = args.valve_tau
            rate_v += args.dt * (args.feed_max * cmd_on - rate - 2 * tau * rate_v) / tau ** 2
            rate = min(max(rate + args.dt * rate_v, 0.0), args.feed_max)
        else:
            rate = min(max(rate + (args.feed_accel if cmd_on else -args.feed_accel) * args.dt, 0.0), args.feed_max)
        feed_rate[i], command[i] = rate, cmd_on
        credit += rate * args.dt
        while credit >= 1.0:
            credit -= 1.0
            free = np.flatnonzero(state == 3)                   # returned grains first
            from_hopper = free.size > 0
            if not from_hopper:
                free = np.flatnonzero(state == 0)
            if not free.size:
                blocked += 1
                continue
            k = free[0]
            G.place(k, box[0] + (box[1] - box[0]) * rng.random(),
                    args.lane[0] + (0.15 + 0.7 * rng.random()) * (args.lane[1] - args.lane[0]), args.drop, sim)
            state[k] = 1
            fed_t.append(t)
            fed_returned.append(from_hopper)
        if i % 25 == 0:
            count[i], hopper[i] = int((state == 1).sum()), int((state == 3).sum())
        else:
            count[i], hopper[i] = count[i - 1], hopper[i - 1]
        recent.append(np.abs(sim.u[:, jc]))
        if len(recent) > osc_n:
            recent.pop(0)
        if i % frame_n == 0:
            F["t"].append(t); F["a"].append(cyc.a(t)); F["on"].append(cyc.env(t))
            F["x"].append(G.x.astype(np.float32)); F["y"].append(G.y.astype(np.float32)); F["z"].append(G.z.astype(np.float32))
            F["state"].append(state.copy()); F["line"].append(sim.u[:, jc].astype(np.float32))
            F["env"].append(np.max(recent, axis=0).astype(np.float32))
        sim.step()
        G.step(sim)
        out = G.alive & (G.x > slot)
        if out.any():
            G.alive[out] = False
            state[out] = 2
            G.vz[out] = np.minimum(G.vz[out], 0.0)
            exit_t.extend([t] * int(out.sum()))
        if args.return_slot > 0:
            back = G.alive & (G.x < args.return_slot)
            if back.any():                                     # into the return chute and back to the hopper
                G.alive[back] = False
                state[back] = 3
                return_t.extend([t] * int(back.sum()))
        fall = state == 2
        if fall.any():
            G.vz[fall] -= g * args.dt
            G.x[fall] += G.vx[fall] * args.dt
            G.z[fall] += G.vz[fall] * args.dt
            state[fall & (G.z < -0.2)] = 0
        if i % per_n == per_n - 1:
            c = i // per_n + 1
            print(f"d = {d:g}, cycle {c}: on the conveyor {int((state == 1).sum())}, feed rate {rate:.3f}, feeder "
                  f"{'on' if cmd_on else 'off'}, fed {len(fed_t)} ({int(np.sum(fed_returned))} returned), offloaded {len(exit_t)}, "
                  f"returned {len(return_t)}, {time.time() - clock:.0f} s", flush=True)
    if blocked:
        print(f"warning: the feeder was blocked {blocked} times because the array was full")
    xs = np.array([model.still_points(a, MODE)[0] for a in F["a"]]).astype(np.float32)
    np.savez_compressed(RESULTS / f"{name(d)}.npz", **{k: np.array(v) for k, v in F.items()}, nodes=xs, xgrid=model.x,
                        feed_rate=feed_rate, command=command, count=count, dt=args.dt, period=cyc.period,
                        times=np.array([cyc.ts, args.fade, args.quiet, args.gather]), measured=np.array(measured),
                        fed=np.array(fed_t), exits=np.array(exit_t), d=d, box=np.array(box), wall=wall, slot=slot,
                        lane=np.array(args.lane), drop=args.drop, target=args.target, feed_max=args.feed_max,
                        feed_accel=args.feed_accel, a_max=args.a, rate=args.rate, g=g, omega=w, Lx=model.Lx, Ly=model.Ly,
                        blocked=blocked, returns=np.array(return_t), fed_returned=np.array(fed_returned), hopper=hopper,
                        return_slot=args.return_slot, valve=args.valve, valve_tau=args.valve_tau)
    keep = ("feed_rate", "command", "count", "hopper", "dt", "period", "times", "measured", "fed", "fed_returned", "exits",
            "returns", "d", "box", "wall", "slot", "target", "feed_max", "feed_accel", "valve_tau", "return_slot", "a_max", "rate")
    z = np.load(RESULTS / f"{name(d)}.npz")
    np.savez_compressed(RESULTS / f"{name(d)}_series.npz", **{k: z[k] for k in keep})
    print("saved", RESULTS / f"{name(d)}.npz")


# ============================================================================ figure
def figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    height = 3.9 * len(args.d) + 0.6
    fig, axes = plt.subplots(len(args.d), 1, figsize=(11, height), sharex=True,
                             gridspec_kw=dict(left=0.09, right=0.905, top=1 - 0.5 / height, bottom=0.85 / height, hspace=0.3))
    for ax, d, tag in zip(np.atleast_1d(axes), args.d, "ABCDEF"):
        z = np.load(RESULTS / f"{name(d)}.npz")
        dt, period = float(z["dt"]), float(z["period"])
        f, n = z["feed_rate"], z["count"]
        t = np.arange(len(f)) * dt
        n_cyc = int(round(len(f) * dt / period))
        ax.plot(t / period, f, color="k", linewidth=1.4, rasterized=True)
        ax.set_ylim(0, 1.12 * float(z["feed_max"]))
        ax.set_ylabel("feed rate")
        ax.set_xlim(0, n_cyc)
        ax2 = ax.twinx()
        ax2.plot(t[::50] / period, n[::50], color="0.6", linewidth=1.4)
        ax2.axhline(float(z["target"]), color="0.6", linewidth=1.0, linestyle=(0, (4, 3)))
        ax2.set_ylim(0, 1.25 * max(float(z["target"]), n.max()))
        ax2.set_ylabel("grains on the conveyor", color="0.35")
        ax2.spines["right"].set_visible(True)
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.set_title(f"{tag}: Feeder at d = {d:g} (x = {float(z['Lx']) - d:g})")
        c = (t // period).astype(int)
        mean = np.array([f[c == q].mean() for q in range(n_cyc)])
        late = mean[-10:]
        m = z["measured"]
        print(f"d = {d:g}: mean feed rate in each of the last 10 cycles {np.round(late, 3)}; overall last-10 mean {late.mean():.3f}, "
              f"spread {late.std():.3f}; grains at the last 10 counts {m[-10:, 1].astype(int).tolist()}; "
              f"fed {len(z['fed'])}, offloaded {len(z['exits'])}; feeder on in {int(m[:, 2].sum())} of {len(m)} decisions")
    np.atleast_1d(axes)[-1].set_xlabel("time (cycles)")
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_adjustable_conveyor.{ext}")
    print("saved", RESULTS / "fig_adjustable_conveyor.png")


# ============================================================================ movie
def movie(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    style()
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
    for d in args.d:
        z = np.load(RESULTS / f"{name(d)}.npz")
        t, a, on = z["t"], z["a"], z["on"]
        X, Y, Z, S = z["x"], z["y"], z["z"], z["state"]
        line, env, nodes, xg = z["line"], z["env"], z["nodes"], z["xgrid"]
        Lx, lane, wall, slot, box, drop = float(z["Lx"]), z["lane"], float(z["wall"]), float(z["slot"]), z["box"], float(z["drop"])
        period, dt, target, fmax = float(z["period"]), float(z["dt"]), float(z["target"]), float(z["feed_max"])
        ts, fade, quiet, gather = z["times"]
        feed, count, command = z["feed_rate"], z["count"], z["command"]
        fed_t, exit_t = z["fed"], z["exits"]
        ret_t = z["returns"] if "returns" in z.files and float(z["return_slot"]) > 0 else None
        n_cycles = int(round(len(feed) * dt / period))
        tf = np.arange(len(feed)) * dt
        sub = slice(None, None, 25)
        tfs, fs, cs = tf[sub] / period, feed[sub], count[sub]
        W, H = 12.8, 10.4
        fig = plt.figure(figsize=(W, H), dpi=100)
        top = 7.2 / H
        ax = fig.add_axes([0.06, 1 - top + top * 0.42, 0.60, top * 0.533])
        sx = fig.add_axes([0.06, 1 - top + top * 0.10, 0.60, top * 0.24])
        ax.add_patch(Rectangle((0, 0), Lx, lane[0], color="0.88", linewidth=0))
        ax.add_patch(Rectangle((0, lane[1]), Lx, 7 - lane[1], color="0.88", linewidth=0))
        for yy in lane:
            ax.plot([wall, slot], [yy, yy], color="k", linewidth=2.2)
        ax.plot([wall + 0.03, wall + 0.03], lane, color="k", linewidth=3.0)
        ax.plot([slot, slot], lane, color="k", linewidth=1.6, linestyle=(0, (1, 1.5)))
        ax.add_patch(Rectangle((box[0], lane[0] + 0.15 * (lane[1] - lane[0])), box[1] - box[0], 0.7 * (lane[1] - lane[0]),
                               fill=False, edgecolor="0.3", linewidth=1.2, hatch="////"))
        ax.text(0.5 * (box[0] + box[1]), lane[1] + 0.18, "feeder", ha="center", va="bottom", fontsize=12)
        ax.text(slot - 0.1, lane[1] + 0.18, "offload slot", ha="right", va="bottom", fontsize=12)
        if ret_t is not None:
            xr = float(z["return_slot"])
            ax.add_patch(Rectangle((wall + 0.03, lane[0]), xr - wall - 0.03, lane[1] - lane[0], facecolor="0.75",
                                   edgecolor="none", hatch="////", zorder=0))
            ax.text(wall + 0.1, lane[1] + 0.18, "wall and return slot", ha="left", va="bottom", fontsize=12)
        else:
            ax.text(wall + 0.1, lane[1] + 0.18, "wall", ha="left", va="bottom", fontsize=12)
        ax.annotate("", xy=(slot, lane[0] - 0.4), xytext=(0.5 * (box[0] + box[1]), lane[0] - 0.4),
                    arrowprops=dict(arrowstyle="<|-|>", color="k", linewidth=1.3))
        ax.text(0.5 * (0.5 * (box[0] + box[1]) + slot), lane[0] - 0.48, f"d = {d:g}", ha="center", va="top", fontsize=12)
        node_lines = [ax.plot([0, 0], lane, color="k", linestyle=(0, (4, 3)), linewidth=1.5)[0] for _ in range(nodes.shape[1] - 2)]
        dots = ax.scatter([], [], s=9, color="0.25", linewidths=0)
        ax.set_xlim(0, Lx); ax.set_ylim(1.0, 6.0); ax.set_aspect("equal"); ax.set_xticklabels([])
        ax.set_ylabel("y"); ax.set_title("Top View", loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
        zlo, zhi = -0.16, drop + 0.08
        band = [sx.fill_between(xg, -env[0], env[0], color="0.82", linewidth=0)]
        prof, = sx.plot(xg, line[0], color="k", linewidth=1.3)
        sdots = sx.scatter([], [], s=9, color="0.25", linewidths=0, zorder=3)
        hopper = Polygon([[box[0] - 0.25, zhi], [box[1] + 0.25, zhi], [box[1], drop + 0.03], [box[0], drop + 0.03]],
                         closed=True, facecolor="0.7", edgecolor="k", linewidth=1.2)
        sx.add_patch(hopper)
        sx.plot([wall + 0.03, wall + 0.03], [-0.01, 0.07], color="k", linewidth=3.0)
        sx.plot([slot, slot], [zlo, 0.0], color="k", linewidth=1.6, linestyle=(0, (1, 1.5)))
        sx.set_xlim(0, Lx); sx.set_ylim(zlo, zhi); sx.set_xlabel("x"); sx.set_ylabel("height")
        sx.set_title("Side View (height stretched)", loc="left")
        tx = fig.add_axes([0.70, 1 - top + top * 0.10, 0.28, top * 0.83]); tx.axis("off")
        tx.text(0.0, 0.97, "Self-Regulating Conveyor", fontsize=15, fontweight="bold", va="top")
        tx.text(0.0, 0.90, f"Feeder at d = {d:g} from the slot.\nTarget: {target:.0f} grains.", fontsize=12, va="top", linespacing=1.4)
        txt_cycle = tx.text(0.0, 0.74, "", fontsize=14, fontweight="bold", va="top")
        txt_phase = tx.text(0.0, 0.67, "", fontsize=13, va="top", linespacing=1.4)
        txt_feed = tx.text(0.0, 0.50, "", fontsize=14, fontweight="bold", va="top")
        tx.add_patch(Rectangle((0.0, 0.405), 0.9, 0.025, fill=False, edgecolor="k", linewidth=1.0, transform=tx.transAxes))
        bar = Rectangle((0.0, 0.405), 0.0, 0.025, color="0.4", linewidth=0, transform=tx.transAxes); tx.add_patch(bar)
        tx.text(0.0, 0.39, "feed rate: 0", fontsize=11, va="top"); tx.text(0.9, 0.39, "full on", fontsize=11, va="top", ha="right")
        txt_n = tx.text(0.0, 0.29, "", fontsize=13, va="top", linespacing=1.6)
        bx = fig.add_axes([0.06, 0.065, 0.86, (1 - top) * 0.74])
        bx.set_xlim(0, n_cycles); bx.set_ylim(0, 1.12 * fmax)
        bx.set_xlabel("time (cycles)"); bx.set_ylabel("feed rate")
        bx.set_title("Feed Rate (black, left) and Grains on the Conveyor (grey, right)", loc="left")
        fl, = bx.plot([], [], color="k", linewidth=1.5)
        bx2 = bx.twinx()
        bx2.set_ylim(0, 1.25 * max(target, count.max())); bx2.set_ylabel("grains", color="0.35")
        bx2.axhline(target, color="0.6", linewidth=1.0, linestyle=(0, (4, 3)))
        bx2.spines["right"].set_visible(True)
        nl, = bx2.plot([], [], color="0.6", linewidth=1.5)
        bx.set_zorder(bx2.get_zorder() + 1); bx.patch.set_visible(False)
        now = bx.axvline(0, color="k", linewidth=0.8)

        def draw(k):
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
                ph = "Sweep: drive on, slope rising."
            elif s < ts + fade + quiet:
                ph = "Reset: drive off.\nGrains were counted."
            else:
                ph = "Gather: drive on."
            i = min(int(round(t[k] / dt)), len(feed) - 1)
            txt_cycle.set_text(f"Cycle {c} of {n_cycles}"); txt_phase.set_text(ph)
            txt_feed.set_text(f"Feeder {'ON' if command[i] else 'OFF'}  (rate {feed[i]:.2f})")
            hopper.set_facecolor("0.35" if feed[i] > 0 else "0.9")
            bar.set_width(0.9 * feed[i] / fmax)
            ret_line = (f"\nreturned {np.searchsorted(ret_t, t[k], side='right')}" if ret_t is not None else "")
            txt_n.set_text(f"on the conveyor {int(live.sum())}{ret_line}\nfed {np.searchsorted(fed_t, t[k], side='right')}\n"
                           f"offloaded {np.searchsorted(exit_t, t[k], side='right')}")
            j = int(np.searchsorted(tfs, t[k] / period, side="right"))
            fl.set_data(tfs[:j], fs[:j]); nl.set_data(tfs[:j], cs[:j])
            now.set_xdata([t[k] / period] * 2)

        if args.still is not None:
            draw(args.still)
            fig.savefig(RESULTS / f"{name(d)}_frame.png", dpi=100)
            print("saved still")
            plt.close(fig)
            continue
        out = RESULTS / f"{name(d)}.mp4"
        w_, h_ = fig.canvas.get_width_height()
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w_}x{h_}",
                               "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf),
                               "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
        n = 0
        for k in range(0, len(t), args.stride):
            draw(k)
            fig.canvas.draw()
            ff.stdin.write(fig.canvas.buffer_rgba())
            n += 1
        ff.stdin.close(); ff.wait()
        plt.close(fig)
        print(f"saved {out} ({n} frames, {n / args.fps:.0f} s)")


# ============================================================================ delay embedding
def load_series(d):
    path = RESULTS / f"{name(d)}_series.npz"
    return np.load(path if path.exists() else RESULTS / f"{name(d)}.npz")


def first_zero_lag(x, max_lag):
    """Lag (in samples) at which the autocorrelation of x first falls to zero."""
    x = x - x.mean()
    for lag in range(1, max_lag):
        if np.dot(x[:-lag], x[lag:]) <= 0.0:
            return lag
    return max_lag


def embed(args):
    """3D delay embedding (N(t), N(t + tau), N(t + 2 tau)) of the grains on the conveyor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    n_d = len(args.d)
    fig = plt.figure(figsize=(11, 4.9))
    rows = []
    for q, (d, tag) in enumerate(zip(args.d, "ABCDEF")):
        z = load_series(d)
        dt, period = float(z["dt"]), float(z["period"])
        step = int(round(args.sample / dt))
        n = z["count"][::step].astype(float)
        t = np.arange(len(n)) * args.sample / period                   # in cycles
        k = max(1, int(round(args.smooth / args.sample)))
        if k > 1:                                                      # running mean over the fast grain-by-grain steps
            n = np.convolve(n, np.ones(k) / k, mode="same")
            n, t = n[k:-k], t[k:-k]                                    # drop the ends, where the mean runs off the data
        settled = t >= args.skip
        if args.tau is None:
            lag = first_zero_lag(n[settled], int(round(6 * period / args.sample)))
        else:
            lag = int(round(args.tau * period / args.sample))
        tau = lag * args.sample / period
        x, y, w = n[:-2 * lag], n[lag:-lag], n[2 * lag:]
        te = t[:-2 * lag]
        ax = fig.add_axes([-0.012 + q / n_d, 0.0, 1.0 / n_d, 0.88], projection="3d")
        ax.patch.set_alpha(0.0)
        early = te < args.skip
        if args.show_fill:
            ax.plot(x[early], y[early], w[early], color="0.75", linewidth=0.9)
        late = ~early
        seg = np.array_split(np.flatnonzero(late), 12)
        for j, idx in enumerate(seg):
            if len(idx) < 2:
                continue
            idx = np.r_[idx, min(idx[-1] + 1, len(x) - 1)]
            ax.plot(x[idx], y[idx], w[idx], color=str(0.55 * (1 - j / (len(seg) - 1))), linewidth=1.3)
        tgt = float(z["target"])
        ax.scatter([tgt], [tgt], [tgt], marker="+", s=140, color="k", linewidths=1.8, depthshade=False)
        lo, hi = 0.97 * n[settled].min(), 1.03 * n[settled].max()
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
        ax.set_xlabel("N(t)", labelpad=7, fontsize=12)
        ax.set_ylabel(r"N(t + $\tau$)", labelpad=7, fontsize=12)
        ax.set_zlabel(r"N(t + 2$\tau$)", labelpad=9, fontsize=12)
        ax.tick_params(labelsize=10, pad=2)
        ax.locator_params(nbins=4)
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect(None, zoom=0.80)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor("white"); axis.pane.set_edgecolor("0.6")
            axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
        fig.text((q + 0.5) / n_d, 0.925, f"{tag}: d = {d:g}, $\\tau$ = {tau:.2f} cycles", ha="center", fontsize=15, fontweight="bold")
        rows += [(d, a_, b_, c_, e_) for a_, b_, c_, e_ in zip(te, x, y, w)]
        m = n[settled]
        spec = np.abs(np.fft.rfft(m - m.mean()))
        f = np.fft.rfftfreq(len(m), args.sample / period)
        band = (f > 1 / 12.0)
        auto = first_zero_lag(m, int(round(6 * period / args.sample))) * args.sample / period
        print(f"d = {d:g}: tau = {tau:.2f} cycles (first zero of the autocorrelation: {auto:.2f}); after cycle {args.skip:g} the count "
              f"runs from {m.min():.0f} to {m.max():.0f}; strongest period below 12 cycles: {1 / f[band][np.argmax(spec[band])]:.1f} cycles")
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_adjustable_embedding.{ext}")
    np.savetxt(RESULTS / "adjustable_embedding.csv", np.array(rows), delimiter=",", comments="", fmt="%.6g",
               header="d,time_cycles,N_t,N_t_plus_tau,N_t_plus_2tau")
    print("saved", RESULTS / "fig_adjustable_embedding.png")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("what", choices=["run", "figure", "movie", "embed"])
    p.add_argument("--d", type=float, nargs="+", default=[8.0], help="distance from the feeder centre to the right edge")
    p.add_argument("--target", type=float, default=3000.0, help="grains to hold on the conveyor")
    p.add_argument("--cycles", type=int, default=40)
    p.add_argument("--feed-max", type=float, default=0.5, help="full-on feed rate, grains per time unit")
    p.add_argument("--feed-accel", type=float, default=3.05e-4, help="ramp valve: how fast the feed rate rises or falls")
    p.add_argument("--valve", choices=["smooth", "ramp"], default="smooth")
    p.add_argument("--valve-tau", type=float, default=420.0, help="smooth valve: time constant tau_f")
    p.add_argument("--return-slot", type=float, default=0.3, help="x of the return slot at the left edge (0 = wall, no return)")
    p.add_argument("--tag", default="", help="added to the output names")
    p.add_argument("--feed-width", type=float, default=0.8)
    p.add_argument("--drop", type=float, default=0.12)
    p.add_argument("--rate", type=float, default=2e-4, help="sweep rate of the tension slope")
    p.add_argument("--a", type=float, default=0.131)
    p.add_argument("--fade", type=float, default=30.0)
    p.add_argument("--quiet", type=float, default=60.0)
    p.add_argument("--gather", type=float, default=210.0)
    p.add_argument("--amplitude", type=float, default=0.029)
    p.add_argument("--gamma", type=float, default=4.0)
    p.add_argument("--roughness", type=float, default=0.3)
    p.add_argument("--mu-r", type=float, default=0.05)
    p.add_argument("--restitution", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--lane", type=float, nargs=2, default=[1.75, 5.25])
    p.add_argument("--pool", type=int, default=8000, help="array size; must stay above the number of grains in play")
    p.add_argument("--frame-every", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--crf", type=int, default=28)
    p.add_argument("--still", type=int, default=None)
    p.add_argument("--tau", type=float, default=None, help="embed: delay in cycles (default: first zero of the autocorrelation)")
    p.add_argument("--skip", type=float, default=8.0, help="embed: cycles treated as the first fill and left out")
    p.add_argument("--show-fill", action="store_true", help="embed: also draw the first fill, in light grey")
    p.add_argument("--sample", type=float, default=10.0, help="embed: time between the samples used")
    p.add_argument("--smooth", type=float, default=100.0, help="embed: running mean over this time (0 = none)")
    args = p.parse_args(argv)
    global TAG
    TAG = args.tag
    {"run": run, "figure": figure, "movie": movie, "embed": embed}[args.what](args)


if __name__ == "__main__":
    main()
