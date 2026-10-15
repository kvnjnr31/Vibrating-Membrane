"""Ratchet conveyor on the 2D membrane: grains are fed at one end, carried
across by repeated tension sweeps and taken off at the other end.

    python particles/run_membrane_conveyor.py                # run and save
    python particles/run_membrane_movie.py                   # then the movie

Membrane mode (5,1), driven from the look-up table with the peak
acceleration held at 4 g.  Grains follow the rough rolling rules of
particles/roll_grains.py (roughness 0.3, rolling resistance 0.05).

Each cycle: the slope rises from -a to +a with the drive on, the drive fades
out, the slope is reset with the drive off, the drive fades in and the
grains gather.  Grains move toward larger x, against the nodal lines.

Layout (all in the membrane's own units, Lx = 10, Ly = 7):
* a lane between two side rails at y = --lane, which keep grains out of the
  quiet strips along the membrane edges, and a back rail at x = --back;
* a feeder that drops one grain every --feed-every time units from a height
  --drop above the membrane, inside the box --feed-box;
* an offload slot at x = --offload: a grain that passes it leaves the
  membrane and falls away.

Writes results/membrane_conveyor.npz (frames for the movie and statistics).
"""
import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.sweep_experiment import make_model                              # noqa: E402
from forcing import run_make_table                                          # noqa: E402
from particles.experiment import RESULTS                                    # noqa: E402
from particles.roll_grains import RollGrains, driven_membrane               # noqa: E402
from particles.run_ratchet import cycle_functions                           # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, nargs=2, default=[5, 1])
    p.add_argument("--a", type=float, default=0.131)
    p.add_argument("--rate", type=float, default=2e-4)
    p.add_argument("--cycles", type=int, default=12)
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
    p.add_argument("--feed-box", type=float, nargs=2, default=[1.8, 2.6], help="x range of the feeder")
    p.add_argument("--feed-every", type=float, default=5.0)
    p.add_argument("--drop", type=float, default=0.12, help="height of the feeder above the membrane")
    p.add_argument("--pool", type=int, default=3000, help="largest number of grains on the membrane")
    p.add_argument("--prefill", type=int, default=0, help="grains scattered over the lane at the start")
    p.add_argument("--frame-every", type=float, default=8.0, help="time between saved frames")
    p.add_argument("--slow-window", type=float, nargs=2, default=None, help="also save closely spaced frames between these two times, for slow motion")
    p.add_argument("--slow-every", type=float, default=0.4, help="time between the slow-motion frames")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reverse", action="store_true", help="sweep the slope from +a down to -a, so the nodal lines move toward the offload slot")
    p.add_argument("--control", action="store_true", help="control: slow sweeps both ways with the drive always on")
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)
    mode = tuple(args.mode)
    model = make_model("membrane", args.beta)[0]
    sgn = -1.0 if args.reverse else 1.0
    a_cyc, env_cyc, period = cycle_functions("control" if args.control else "ratchet", -sgn * args.a, sgn * args.a, args.rate, 0.0, args.fade, args.quiet, args.gather)
    a_of_t = lambda t: a_cyc(max(t, 1e-9))                     # noqa: E731
    env_of_t = lambda t: env_cyc(max(t, 1e-9))                 # noqa: E731
    sim, g, w, t0 = driven_membrane(model, mode, a_of_t, env_of_t, -sgn * args.a, args.amplitude, args.gamma,
                                    args.dt, table=run_make_table.table_path(mode))
    G = RollGrains(args.pool, model, g, w, restitution=args.restitution, roughness=args.roughness,
                   mu_r=args.mu_r, seed=args.seed)
    G.xlo, G.ylo, G.yhi = args.back, args.lane[0], args.lane[1]
    G.alive[:] = False
    rng = np.random.default_rng(100 + args.seed)
    if args.prefill:
        idx = np.arange(args.prefill)
        G.x[idx] = args.back + (args.offload - args.back) * rng.random(args.prefill)
        G.y[idx] = args.lane[0] + (args.lane[1] - args.lane[0]) * rng.random(args.prefill)
        G.alive[idx] = True
    state = np.zeros(args.pool, dtype=np.int8)                 # 0 = in the pool, 1 = on the membrane, 2 = falling away
    state[G.alive] = 1
    born = np.full(args.pool, np.nan)                          # time each grain was fed
    jc = len(model.y) // 2                                     # centreline row for the side view
    steps = int(round(args.cycles * period / args.dt))
    feed_n, frame_n = int(round(args.feed_every / args.dt)), int(round(args.frame_every / args.dt))
    per_n = int(round(2 * np.pi / w / args.dt))
    env = np.zeros(len(model.x))
    recent = []
    F = dict(t=[], a=[], on=[], x=[], y=[], z=[], state=[], line=[], env=[])
    Fs = {k: [] for k in F}
    slow_n = int(round(args.slow_every / args.dt))
    fed, exits, transit = [], [], []
    clock = time.time()
    for i in range(steps + 1):
        t = sim.t - t0
        if i % feed_n == 0:
            free = np.flatnonzero(state == 0)
            if free.size:
                k = free[0]
                x = args.feed_box[0] + (args.feed_box[1] - args.feed_box[0]) * rng.random()
                y = args.lane[0] + (0.15 + 0.7 * rng.random()) * (args.lane[1] - args.lane[0])
                G.place(k, x, y, args.drop, sim)
                state[k], born[k] = 1, t
                fed.append(t)
        recent.append(np.abs(sim.u[:, jc]))
        if len(recent) > per_n:
            recent.pop(0)
        if args.slow_window and args.slow_window[0] <= t < args.slow_window[1] and i % slow_n == 0:
            Fs["t"].append(t); Fs["a"].append(a_of_t(t)); Fs["on"].append(env_of_t(t))
            Fs["x"].append(G.x.astype(np.float32)); Fs["y"].append(G.y.astype(np.float32)); Fs["z"].append(G.z.astype(np.float32))
            Fs["state"].append(state.copy()); Fs["line"].append(sim.u[:, jc].astype(np.float32))
            Fs["env"].append(np.max(recent, axis=0).astype(np.float32))
        if i % frame_n == 0:
            F["t"].append(t); F["a"].append(a_of_t(t)); F["on"].append(env_of_t(t))
            F["x"].append(G.x.astype(np.float32)); F["y"].append(G.y.astype(np.float32)); F["z"].append(G.z.astype(np.float32))
            F["state"].append(state.copy()); F["line"].append(sim.u[:, jc].astype(np.float32))
            F["env"].append(np.max(recent, axis=0).astype(np.float32))
        sim.step()
        G.step(sim)
        out = G.alive & (G.x > args.offload)
        if out.any():
            G.alive[out] = False
            state[out] = 2
            G.vz[out] = np.minimum(G.vz[out], 0.0)
            exits.extend([t] * int(out.sum()))
            transit.extend((t - born[out]).tolist())
        fall = state == 2
        if fall.any():                                         # grains that left the membrane fall away
            G.vz[fall] -= g * args.dt
            G.x[fall] += G.vx[fall] * args.dt
            G.z[fall] += G.vz[fall] * args.dt
            gone = fall & (G.z < -0.2)
            state[gone] = 0
        if i % int(round(period / args.dt)) == 0 and i:
            c = i // int(round(period / args.dt))
            on_m = state == 1
            print(f"cycle {c}: {on_m.sum()} grains on the membrane, mean x {G.x[on_m].mean():.2f}, fed {len(fed)}, "
                  f"exited {len(exits)}, {time.time() - clock:.0f} s", flush=True)
    exits, fed, transit = np.array(exits), np.array(fed), np.array(transit)
    per_cycle = np.histogram(exits, bins=np.arange(args.cycles + 1) * period)[0]
    print(f"cycle length {period:.0f}; exits in each cycle: {per_cycle}; "
          f"transit time: median {np.median(transit) if transit.size else float('nan'):.0f}")
    xs = np.array([model.still_points(a, mode)[0] for a in F["a"]])
    slow = {f"slow_{k}": np.array(v) for k, v in Fs.items()} if Fs["t"] else {}
    if slow:
        slow["slow_nodes"] = np.array([model.still_points(a, mode)[0] for a in Fs["a"]]).astype(np.float32)
    np.savez_compressed(RESULTS / f"membrane_conveyor{args.tag}.npz", **{k: np.array(v) for k, v in F.items()},
                        nodes=xs.astype(np.float32), **slow, xgrid=model.x, period=period, g=g, omega=w, exits=exits, fed=fed,
                        transit=transit, exits_per_cycle=per_cycle, lane=np.array(args.lane), back=args.back,
                        offload=args.offload, feed_box=np.array(args.feed_box), drop=args.drop, Lx=model.Lx, Ly=model.Ly,
                        amplitude=args.amplitude, a_max=args.a, rate=args.rate,
                        times=np.array([2 * args.a / args.rate, args.fade, args.quiet, args.gather]))
    print("saved", RESULTS / f"membrane_conveyor{args.tag}.npz")


if __name__ == "__main__":
    main()
