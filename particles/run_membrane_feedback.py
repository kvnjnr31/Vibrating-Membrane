"""Membrane conveyor under feedback: hold a set number of grains on the
conveyor by adjusting the feed rate and the sweep rate once per cycle.

    python particles/run_membrane_feedback.py

Same membrane, lane, feeder, offload slot and grain rules as
run_membrane_conveyor.py.  At the end of every cycle the controller reads
N, the number of grains on the conveyor, and sets, for the next cycle,

    feed rate   F = I + KP * (N_target - N),   I <- I + KI * (N_target - N)
    sweep rate  r = r0 * (1 + KR * (N - N_target) / N_target)

both kept inside fixed limits.  The feed rate has a memory term (I), so it
settles wherever input equals output at N_target.  The sweep rate has none,
so it returns to r0: it speeds the conveyor up while there are too many
grains and slows it down while there are too few.

Writes results/membrane_feedback.npz and .csv (one row per cycle).
"""
import argparse
import csv
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.sweep_experiment import make_model                              # noqa: E402
from forcing import run_make_table                                          # noqa: E402
from particles.experiment import RESULTS                                    # noqa: E402
from particles.roll_grains import RollGrains, driven_membrane               # noqa: E402


class Cycle:
    """One ratchet cycle starting at t_start with its own sweep rate."""

    def __init__(self, a_max, rate, fade, quiet, gather, t_start=0.0):
        self.a_max, self.fade, self.quiet, self.gather = a_max, fade, quiet, gather
        self.start(t_start, rate)

    def start(self, t_start, rate):
        self.t_start, self.rate = t_start, rate
        self.ts = 2 * self.a_max / rate
        self.period = self.ts + 2 * self.fade + self.quiet + self.gather

    def a(self, t):
        s = t - self.t_start
        if s <= 0.0 or s >= self.period:
            return -self.a_max
        if s < self.ts:
            return -self.a_max + 2 * self.a_max * s / self.ts
        q = (s - self.ts - self.fade - 0.5 * self.quiet) / (0.4 * self.quiet)
        return self.a_max - 2 * self.a_max * min(max(q, 0.0), 1.0)

    def env(self, t):
        s = t - self.t_start
        if s < self.ts or s >= self.period:
            return 1.0
        if s < self.ts + self.fade:
            return 1.0 - (s - self.ts) / self.fade
        if s < self.ts + self.fade + self.quiet:
            return 0.0
        return min((s - self.ts - self.fade - self.quiet) / self.fade, 1.0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", type=float, default=2000.0, help="grains to hold on the conveyor")
    p.add_argument("--cycles", type=int, default=40)
    p.add_argument("--rate0", type=float, default=2e-4, help="nominal sweep rate r0")
    p.add_argument("--feed0", type=float, default=0.25, help="feed rate in the first cycle, grains per time unit")
    p.add_argument("--kp", type=float, default=6e-5, help="feed gain on the error, per grain")
    p.add_argument("--ki", type=float, default=5.5e-5, help="feed memory gain, per grain and cycle")
    p.add_argument("--kr", type=float, default=1.5, help="sweep-rate gain on the relative error")
    p.add_argument("--feed-limits", type=float, nargs=2, default=[0.0, 0.6])
    p.add_argument("--rate-limits", type=float, nargs=2, default=[0.8e-4, 4e-4])
    p.add_argument("--open-loop", action="store_true", help="keep the feed rate and the sweep rate fixed")
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
    p.add_argument("--back", type=float, default=1.2)
    p.add_argument("--offload", type=float, default=9.3)
    p.add_argument("--feed-box", type=float, nargs=2, default=[1.8, 2.6])
    p.add_argument("--drop", type=float, default=0.12)
    p.add_argument("--pool", type=int, default=8000, help="array size; must stay above the number of grains in play")
    p.add_argument("--frame-every", type=float, default=40.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)
    mode = (5, 1)
    model = make_model("membrane", args.beta)[0]
    cyc = Cycle(args.a, args.rate0, args.fade, args.quiet, args.gather)
    sim, g, w, t0 = driven_membrane(model, mode, cyc.a, cyc.env, -args.a, args.amplitude, args.gamma, args.dt,
                                    table=run_make_table.table_path(mode))
    G = RollGrains(args.pool, model, g, w, restitution=args.restitution, roughness=args.roughness, mu_r=args.mu_r, seed=args.seed)
    G.xlo, G.ylo, G.yhi = args.back, args.lane[0], args.lane[1]
    G.alive[:] = False
    rng = np.random.default_rng(100 + args.seed)
    state = np.zeros(args.pool, dtype=np.int8)
    feed, integ, credit = args.feed0, args.feed0, 0.0
    frame_n = int(round(args.frame_every / args.dt))
    rows, frames = [], dict(t=[], a=[], n=[], x=[], y=[], state=[])
    fed_c = out_c = blocked = 0
    c, i = 0, 0
    clock = time.time()
    while c < args.cycles:
        t = sim.t - t0
        if t - cyc.t_start >= cyc.period:                      # end of a cycle: measure and decide
            n_on = int((state == 1).sum())
            rows.append(dict(cycle=c + 1, t_start=cyc.t_start, period=cyc.period, sweep_rate=cyc.rate, feed_set=feed,
                             input_rate=fed_c / cyc.period, output_rate=out_c / cyc.period, fed=fed_c, offloaded=out_c,
                             grains_end=n_on))
            print(f"cycle {c + 1}: period {cyc.period:.0f}, sweep rate {cyc.rate:.2e}, feed {feed:.3f}, in {fed_c}, out {out_c}, "
                  f"on the conveyor {n_on}, {time.time() - clock:.0f} s", flush=True)
            err = args.target - n_on
            rate = cyc.rate
            if not args.open_loop:
                integ = min(max(integ + args.ki * err, args.feed_limits[0]), args.feed_limits[1])
                feed = min(max(integ + args.kp * err, args.feed_limits[0]), args.feed_limits[1])
                rate = min(max(args.rate0 * (1 - args.kr * err / args.target), args.rate_limits[0]), args.rate_limits[1])
            cyc.start(cyc.t_start + cyc.period, rate)
            fed_c = out_c = 0
            c += 1
            if c == args.cycles:
                break
        credit += feed * args.dt
        while credit >= 1.0:
            credit -= 1.0
            free = np.flatnonzero(state == 0)
            if not free.size:
                blocked += 1
                continue
            k = free[0]
            x = args.feed_box[0] + (args.feed_box[1] - args.feed_box[0]) * rng.random()
            y = args.lane[0] + (0.15 + 0.7 * rng.random()) * (args.lane[1] - args.lane[0])
            G.place(k, x, y, args.drop, sim)
            state[k] = 1
            fed_c += 1
        if i % frame_n == 0:
            frames["t"].append(t); frames["a"].append(cyc.a(t)); frames["n"].append(int((state == 1).sum()))
            frames["x"].append(G.x.astype(np.float32)); frames["y"].append(G.y.astype(np.float32)); frames["state"].append(state.copy())
        sim.step()
        G.step(sim)
        out = G.alive & (G.x > args.offload)
        if out.any():
            G.alive[out] = False
            state[out] = 2
            G.vz[out] = np.minimum(G.vz[out], 0.0)
            out_c += int(out.sum())
        fall = state == 2
        if fall.any():
            G.vz[fall] -= g * args.dt
            G.x[fall] += G.vx[fall] * args.dt
            G.z[fall] += G.vz[fall] * args.dt
            state[fall & (G.z < -0.2)] = 0
        i += 1
    if blocked:
        print(f"warning: the feeder was blocked {blocked} times because the array was full")
    keys = list(rows[0])
    with open(RESULTS / f"membrane_feedback{args.tag}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for r in rows:
            wr.writerow([f"{r[k]:.6g}" if isinstance(r[k], float) else r[k] for k in keys])
    np.savez_compressed(RESULTS / f"membrane_feedback{args.tag}.npz", **{k: np.array([r[k] for r in rows]) for k in keys},
                        **{"frame_" + k: np.array(v) for k, v in frames.items()}, target=args.target, rate0=args.rate0,
                        gains=np.array([args.kp, args.ki, args.kr]), lane=np.array(args.lane), back=args.back, offload=args.offload,
                        blocked=blocked)
    print("saved", RESULTS / f"membrane_feedback{args.tag}.csv")


if __name__ == "__main__":
    main()
