"""Ratchet conveyor with a feeder and an offload, measured by hops between nodes.

    python particles/run_conveyor_hops.py --mode 7 --a 0.093 --amplitude 0.021

The ratchet of run_ratchet.py is run with the slope rising, one run per
sweep rate.  A grain that passes the offload line (half a node spacing
before the far end) is taken off and put back at rest in the feed zone
(between the first antinode and the first interior node).

At the start of every cycle each grain is assigned to its nearest still
point (index 0 = near end, 1 .. n-1 = interior nodes, n = far end).  The
hop of a grain in one cycle is the change of that index.  A grain taken off
during the cycle is counted as having reached the offload line.  The first
cycles are discarded, and only grains that start at one of the first
interior nodes are measured, so that few of them reach the offload.

Writes results/conveyor_hops_mode<n>.csv and .npz.
"""
import argparse
import copy
import csv
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.string_model import TensionString                              # noqa: E402
from forcing.simulation import Simulation                                    # noqa: E402
from forcing.drive import TableDrive                                         # noqa: E402
from forcing.compensation import compensation_gain                           # noqa: E402
from particles.surface import StringSurface                                  # noqa: E402
from particles.experiment import settle, RESULTS                             # noqa: E402
from particles.run_ratchet import cycle_functions                            # noqa: E402
from particles.fast_grains import FastGrains                                 # noqa: E402

HOPS = np.arange(-2, 6)   # hop sizes kept in the histogram (ends collect the rest)


def one(job):
    mode, a, amp, rate, seed, discard, measured, n_grains, beta, points, t_fade, t_quiet, t_gather = job
    model = TensionString(N=points, beta=beta)
    s = settle(model, StringSurface(model), mode, -a, drive="table+gain", gamma=4.0, amplitude=amp,
               n_grains=n_grains, t_settle=1000.0, dt=0.02, seed=seed, verbose=False,
               grain_kwargs=dict(restitution=0.4, roughness=0.3))
    af, env, period = cycle_functions("ratchet", -a, a, rate, s.sim.t, t_fade, t_quiet, t_gather)
    gain = compensation_gain(model, mode, af, s.sim.shape, "acceleration", rate_correction=False)
    w_res = model.resonance_table(mode)
    drive = TableDrive(lambda t: w_res(af(t)), gain_of_t=lambda t: gain(t) * env(t))
    sim = Simulation(model, drive, af, mode, dt=s.sim.dt).continue_from(s.sim)
    surface = s.surface
    surface.update(sim.u, sim.v)
    g = FastGrains(copy.deepcopy(s.grains))
    rng = np.random.default_rng(1000 + seed)

    L, sp = model.L, model.L / mode
    still = np.concatenate(([0.0], model.mode_nodes(-a, mode), [L]))        # where grains sit at a cycle start
    feed, line = (0.5 * sp, 1.0 * sp), L - 0.5 * sp
    last_start = max(1, mode - 3)                                           # measure grains starting at nodes 1 .. last_start
    taken = np.zeros(g.n, dtype=bool)

    def recycle():
        out = g.x > line
        k = int(out.sum())
        if k:
            g.x[out] = feed[0] + (feed[1] - feed[0]) * rng.random(k)
            g.v[out] = 0.0
            g.z[out] = np.interp(g.x[out], surface.x, surface.u)
            g.vz[out] = np.interp(g.x[out], surface.x, surface.v)
            taken[out] = True
        return k

    def index():
        return np.argmin(np.abs(g.x[:, None] - still[None, :]), axis=1)

    recycle()
    steps = int(round(period / sim.dt))
    hops, shifts, starts, off, dist = [], [], [], [], []
    for c in range(discard + measured):
        taken[:] = False
        k0, n_off = index(), 0
        for i in range(steps):
            sim.step()
            surface.update(sim.u, sim.v)
            g.step(sim.dt, surface)
            if i % 10 == 0:
                n_off += recycle()
        if c < discard:
            continue
        k1 = index()
        sel = (k0 >= 1) & (k0 <= last_start)
        dist.append(float(np.mean(np.abs(g.x - still[k1])) / sp))
        hop = np.where(taken, mode - 0.5 - k0, k1 - k0)[sel]
        shift = np.where(taken, line - still[k0], still[k1] - still[k0])[sel]
        hops.append(hop)
        shifts.append(shift)
        starts.append(k0[sel])
        off.append((n_off, float(taken[sel].mean()) if sel.any() else 0.0))
    per_cycle = np.array([x.mean() for x in shifts])
    hop, shift, start = np.concatenate(hops), np.concatenate(shifts), np.concatenate(starts)
    hist = np.array([np.mean(np.clip(np.round(hop), HOPS[0], HOPS[-1]) == h) for h in HOPS])
    by_node = np.array([hop[start == k].mean() if np.any(start == k) else np.nan for k in range(1, last_start + 1)])
    travel = float(np.mean(np.abs(model.mode_nodes(a, mode) - model.mode_nodes(-a, mode))) / sp)
    res = dict(mode=mode, rate=rate, period=period, cycles=measured, spacing=sp, sweep_time=2 * a / rate,
               a=a, amplitude=amp, points=points, node_travel=travel, t_fade=t_fade, t_quiet=t_quiet, t_gather=t_gather,
               dist_to_node=float(np.mean(dist)), raw_hop=hop.astype(np.float32), raw_shift=shift.astype(np.float32),
               raw_start=start.astype(np.int16), raw_cycle=np.concatenate([np.full(x.size, i, dtype=np.int16) for i, x in enumerate(hops)]),
               hop=float(hop.mean()), hop_se=float(hop.std() / np.sqrt(hop.size)),
               hop_se_cycles=float(np.std([x.mean() for x in hops], ddof=1) / np.sqrt(measured)),
               shift=float(shift.mean()), shift_se=float(shift.std() / np.sqrt(shift.size)),
               stay=float(hist[HOPS == 0][0]), back=float(hist[HOPS < 0].sum()),
               grains_per_cycle=float(hop.size / measured), taken_share=float(np.mean([o[1] for o in off])),
               offloaded_per_cycle=float(np.mean([o[0] for o in off])), hist=hist, by_node=by_node, per_cycle=per_cycle)
    print(f"mode {mode} rate {rate:.2e}: period {period:.0f}, {measured} cycles, hop {res['hop']:.3f} +- {res['hop_se']:.3f} nodes, "
          f"shift {res['shift']:.3f}, speed {res['shift'] / period:.2e}, stay {res['stay']:.2f}, gather {t_gather:.0f} quiet {t_quiet:.0f}, dist {res['dist_to_node']:.3f}, taken off {res['taken_share']:.2f}, "
          f"{res['grains_per_cycle']:.0f} grains per cycle", flush=True)
    return res


def slope_for_travel(mode, travel, points, beta):
    """Slope a for which the nodes move ``travel`` node spacings between -a and +a."""
    from scipy.optimize import brentq
    model = TensionString(N=points, beta=beta)
    sp = model.L / mode

    def f(a):
        return np.mean(np.abs(model.mode_nodes(a, mode) - model.mode_nodes(-a, mode))) / sp - travel
    return float(brentq(f, 1e-4, 0.19))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, default=5)
    p.add_argument("--a", type=float, default=None, help="the slope is swept from -a to +a (default: from --travel)")
    p.add_argument("--travel", type=float, default=0.75, help="node travel in one sweep, in node spacings, used when --a is not given")
    p.add_argument("--amplitude", type=float, default=None, help="peak displacement (default: 0.0145 x node spacing)")
    p.add_argument("--rates", type=float, nargs="+", default=None, help="explicit sweep rates (overrides the range)")
    p.add_argument("--fade", type=float, default=30.0)
    p.add_argument("--quiet", type=float, nargs="+", default=[150.0], help="drive-off time; several values are scanned")
    p.add_argument("--gather", type=float, nargs="+", default=[300.0], help="gather time; several values are scanned")
    p.add_argument("--tag", default="", help="suffix for the output file names")
    p.add_argument("--rate-range", type=float, nargs=2, default=[3e-5, 3e-3])
    p.add_argument("--n-rates", type=int, default=18)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grains", type=int, default=2000)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--points", type=int, default=None, help="grid points (default: 20 per half wave, at least 101)")
    p.add_argument("--discard", type=int, default=2, help="cycles thrown away at the start")
    p.add_argument("--min-cycles", type=int, default=6, help="measured cycles, at least")
    p.add_argument("--max-cycles", type=int, default=12)
    p.add_argument("--t-target", type=float, default=7000.0, help="measure for about this long when cycles are short")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args(argv)
    if args.points is None:
        args.points = max(101, 20 * args.mode + 1)
    if args.amplitude is None:
        args.amplitude = 0.0145 * 10.0 / args.mode
    if args.a is None:
        args.a = slope_for_travel(args.mode, args.travel, args.points, args.beta)
    print(f"mode {args.mode}: slope range +-{args.a:.4f}, amplitude {args.amplitude:.4f}, {args.points} grid points", flush=True)
    rates = args.rates if args.rates else np.geomspace(*args.rate_range, args.n_rates)
    jobs = []
    for r in rates:
        for tq in args.quiet:
            for tg in args.gather:
                cyc = int(min(max(round(args.t_target / (2 * args.a / r + 2 * args.fade + tq + tg)), args.min_cycles), args.max_cycles))
                jobs.append((args.mode, args.a, args.amplitude, float(r), args.seed, args.discard, cyc, args.grains,
                             args.beta, args.points, args.fade, tq, tg))
    jobs.sort(key=lambda j: -(2 * j[1] / j[3] + j[11] + j[12]) * j[6])          # longest first
    with Pool(args.workers) as pool:
        out = pool.map(one, jobs, chunksize=1)
    out.sort(key=lambda o: (o["rate"], o["t_quiet"], o["t_gather"]))
    path = RESULTS / f"conveyor_hops_mode{args.mode}{args.tag}"
    keys = ["mode", "rate", "a", "amplitude", "points", "node_travel", "t_fade", "t_quiet", "t_gather", "sweep_time", "period", "cycles", "spacing", "dist_to_node", "hop", "hop_se", "hop_se_cycles", "shift",
            "shift_se", "stay", "back", "grains_per_cycle", "taken_share", "offloaded_per_cycle"]
    with open(f"{path}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for o in out:
            w.writerow([f"{o[k]:.5g}" if isinstance(o[k], float) else o[k] for k in keys])
    np.savez_compressed(f"{path}.npz", rates=np.array([o["rate"] for o in out]), hop_sizes=HOPS,
                        hist=np.array([o["hist"] for o in out]), by_node=np.array([o["by_node"] for o in out]),
                        **{f"per_cycle_{i}": o["per_cycle"] for i, o in enumerate(out)},
                        **{f"{k}_{i}": o[k] for i, o in enumerate(out) for k in ("raw_hop", "raw_shift", "raw_start", "raw_cycle")})
    print(f"saved {path}.csv")


if __name__ == "__main__":
    main()
