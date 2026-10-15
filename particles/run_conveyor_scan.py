"""Ratchet conveyor with a feeder and an offload: transport against sweep rate.

    python particles/run_conveyor_scan.py --mode 7 --a 0.093 --amplitude 0.021

The ratchet of run_ratchet.py is run with the slope rising.  A grain that
passes the offload line is taken off and put back at rest in the feed zone,
so the number of grains stays fixed and they do not pile up at the far end.
An unwrapped position (position plus the distance skipped at each offload)
is kept for every grain.  The transport is read from grains that start a
cycle in the middle zone.

One run per sweep rate.  Writes results/conveyor_scan_mode<n>.csv and .npz.
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

OVERHEAD = 510.0          # fade out, quiet time, fade in and gather
FEED = (1.0, 2.0)         # grains are put back here
OFFLOAD = 8.0             # grains past this line are taken off
ZONE = (2.5, 6.5)         # grains starting a cycle here are measured


def one(job):
    mode, a, amp, rate, seed, cycles, n_grains, beta, points = job
    model = TensionString(N=points, beta=beta)
    s = settle(model, StringSurface(model), mode, -a, drive="table+gain", gamma=4.0, amplitude=amp,
               n_grains=n_grains, t_settle=1000.0, dt=0.02, seed=seed, verbose=False,
               grain_kwargs=dict(restitution=0.4, roughness=0.3))
    af, env, period = cycle_functions("ratchet", -a, a, rate, s.sim.t)
    gain = compensation_gain(model, mode, af, s.sim.shape, "acceleration", rate_correction=False)
    w_res = model.resonance_table(mode)
    drive = TableDrive(lambda t: w_res(af(t)), gain_of_t=lambda t: gain(t) * env(t))
    sim = Simulation(model, drive, af, mode, dt=s.sim.dt).continue_from(s.sim)
    g = copy.deepcopy(s.grains)
    surface = s.surface
    surface.update(sim.u, sim.v)
    rng = np.random.default_rng(1000 + seed)

    def recycle():
        out = g.X[:, 0] > OFFLOAD
        k = int(out.sum())
        if k:
            new = FEED[0] + (FEED[1] - FEED[0]) * rng.random(k)
            skipped[out] += g.X[out, 0] - new
            g.X[out, 0] = new
            g.V[out] = 0.0
            h, w, _ = surface.sample(g.X[out])
            g.z[out], g.vz[out] = h, w
        return k

    skipped = np.zeros(g.n)
    recycle()                                   # clear the far end before the first cycle
    skipped[:] = 0.0
    steps = int(round(period / sim.dt))
    X0, shift, off = [], [], []
    for c in range(cycles):
        x_start, u_start, n_off = g.X[:, 0].copy(), g.X[:, 0] + skipped, 0
        for i in range(steps):
            sim.step()
            surface.update(sim.u, sim.v)
            g.step(sim.dt, surface)
            if i % 10 == 0:
                n_off += recycle()
        X0.append(x_start)
        shift.append(g.X[:, 0] + skipped - u_start)
        off.append(n_off)
    X0, shift = np.array(X0), np.array(shift)
    mid = (X0 > ZONE[0]) & (X0 < ZONE[1])
    per_cycle = np.array([shift[c][mid[c]].mean() for c in range(cycles)])
    spacing = float(np.mean(np.diff(model.mode_nodes(0.0, mode))))
    travel = float(np.mean(np.abs(model.mode_nodes(a, mode) - model.mode_nodes(-a, mode))[1:-1]))
    res = dict(mode=mode, rate=rate, period=period, cycles=cycles, spacing=spacing, node_travel=travel,
               sweep_time=2 * a / rate, shift=float(shift[mid].mean()), shift_first=float(per_cycle[0]),
               shift_last=float(per_cycle[-1]), shift_all=float(shift.mean()), n_mid=float(mid.sum(axis=1).mean()),
               offloaded_per_cycle=float(np.mean(off)), X0=X0, shifts=shift)
    print(f"mode {mode} rate {rate:.2e}: period {period:.0f}, {cycles} cycles, middle shift per cycle "
          f"{np.round(per_cycle, 2)}, offloaded {off}", flush=True)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, default=5)
    p.add_argument("--a", type=float, default=0.13, help="the slope is swept from -a to +a")
    p.add_argument("--amplitude", type=float, default=0.029)
    p.add_argument("--rate-range", type=float, nargs=2, default=[3e-5, 3e-3])
    p.add_argument("--n-rates", type=int, default=18)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grains", type=int, default=500)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--points", type=int, default=201)
    p.add_argument("--t-target", type=float, default=7000.0, help="aim for about this much time per run")
    p.add_argument("--min-cycles", type=int, default=2)
    p.add_argument("--max-cycles", type=int, default=12)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args(argv)
    rates = np.geomspace(*args.rate_range, args.n_rates)
    jobs = []
    for r in rates[::-1]:
        cyc = int(min(max(round(args.t_target / (2 * args.a / r + OVERHEAD)), args.min_cycles), args.max_cycles))
        jobs.append((args.mode, args.a, args.amplitude, float(r), args.seed, cyc, args.grains, args.beta, args.points))
    with Pool(args.workers) as pool:
        out = pool.map(one, jobs, chunksize=1)
    out.sort(key=lambda o: o["rate"])
    path = RESULTS / f"conveyor_scan_mode{args.mode}"
    keys = ["mode", "rate", "sweep_time", "period", "cycles", "spacing", "node_travel", "shift", "shift_first",
            "shift_last", "shift_all", "n_mid", "offloaded_per_cycle"]
    with open(f"{path}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for o in out:
            w.writerow([f"{o[k]:.5g}" if isinstance(o[k], float) else o[k] for k in keys])
    np.savez_compressed(f"{path}.npz", **{f"X0_{i}": o["X0"].astype(np.float32) for i, o in enumerate(out)},
                        **{f"shift_{i}": o["shifts"].astype(np.float32) for i, o in enumerate(out)},
                        rates=np.array([o["rate"] for o in out]))
    print(f"saved {path}.csv")


if __name__ == "__main__":
    main()
