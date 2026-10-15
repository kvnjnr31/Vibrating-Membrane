"""Ratchet transport speed against sweep rate, for one string mode.

    python particles/run_ratchet_scan.py --mode 5 --a 0.13 --amplitude 0.029

For each sweep rate and seed the ratchet of run_ratchet.py is run for a few
cycles with the slope rising.  The grain positions at the start of every
cycle are saved, so the transport speed can be read off afterwards.

Writes results/ratchet_scan_mode<n>.npz and .csv.
"""
import argparse
import csv
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.string_model import TensionString                              # noqa: E402
from particles.surface import StringSurface                                  # noqa: E402
from particles.experiment import settle, RESULTS                             # noqa: E402
from particles.run_ratchet import run                                        # noqa: E402

OVERHEAD = 510.0          # fade out, quiet time, fade in and gather


def n_cycles(a, rate, t_target, lo, hi):
    period = 2 * a / rate + OVERHEAD
    return int(min(max(round(t_target / period), lo), hi))


def one(job):
    mode, a, amp, rate, seed, cycles, grains, beta, points = job
    model = TensionString(N=points, beta=beta)
    s = settle(model, StringSurface(model), mode, -a, drive="table+gain", gamma=4.0, amplitude=amp,
               n_grains=grains, t_settle=1000.0, dt=0.02, seed=seed,
               grain_kwargs=dict(restitution=0.4, roughness=0.3))
    res = run("ratchet", s, a, rate, cycles)
    per = res["period"]
    marks = [int(np.searchsorted(res["t"], k * per)) for k in range(cycles + 1)]
    marks[-1] = len(res["t"]) - 1
    X = res["X"][marks]
    print(f"mode {mode} rate {rate:.1e} seed {seed}: period {per:.0f}, mean {np.round(X.mean(axis=1), 2)}", flush=True)
    return dict(rate=rate, seed=seed, period=per, X=X, spacing=float(np.mean(np.diff(model.mode_nodes(0.0, mode)))))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, default=5)
    p.add_argument("--a", type=float, default=0.13, help="the slope is swept from -a to +a")
    p.add_argument("--amplitude", type=float, default=0.029)
    p.add_argument("--rates", type=float, nargs="+", default=[3e-5, 8e-5, 2e-4, 5e-4, 1e-3, 3e-3])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--grains", type=int, default=500)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--points", type=int, default=201)
    p.add_argument("--t-target", type=float, default=7000.0, help="aim for about this much time per run")
    p.add_argument("--min-cycles", type=int, default=2)
    p.add_argument("--max-cycles", type=int, default=10)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args(argv)
    jobs = [(args.mode, args.a, args.amplitude, r, s, n_cycles(args.a, r, args.t_target, args.min_cycles, args.max_cycles),
             args.grains, args.beta, args.points) for r in sorted(args.rates, reverse=True) for s in args.seeds]
    with Pool(args.workers) as pool:
        out = pool.map(one, jobs, chunksize=1)
    path = RESULTS / f"ratchet_scan_mode{args.mode}"
    np.savez_compressed(f"{path}.npz", **{f"r{o['rate']:.1e}_s{o['seed']}": o["X"] for o in out},
                        meta=np.array([(o["rate"], o["seed"], o["period"], o["spacing"]) for o in out]))
    with open(f"{path}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "rate", "seed", "period", "spacing", "cycles", "mean_at_each_cycle_start"])
        for o in out:
            w.writerow([args.mode, o["rate"], o["seed"], round(o["period"], 1), round(o["spacing"], 3), len(o["X"]) - 1,
                        " ".join(f"{m:.3f}" for m in o["X"].mean(axis=1))])
    print(f"saved {path}.csv")


if __name__ == "__main__":
    main()
