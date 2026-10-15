"""Bifurcation diagram of the reduced conveyor model against the feeder distance d.

    python particles/run_model_bifurcation.py --d0 1 --d1 9.5 --step 0.1 --cycles 300

For each d the model (no noise, fitted v and D) runs for `cycles` cycles. The counts at the
last `keep` sweep ends, the repeat length of the on/off pattern and the feed statistics are
saved to results/model_bifurcation.npz, and the diagram to results/fig_model_bifurcation.{png,pdf}.
"""
import argparse
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from particles.experiment import RESULTS                                    # noqa: E402
from particles.run_model_long import run, period_of                         # noqa: E402

ARGS = None


def one(d):
    r = run(d, ARGS.cycles, cells=ARGS.cells, dt=ARGS.dt, record=100.0)
    s, n = r["dec_s"], r["dec_N"]
    keep = ARGS.keep
    p = period_of(s, tail=keep, max_p=keep // 2)
    half = len(r["off"]) // 2
    return dict(d=d, N=n[-keep:], s=s[-keep:], period=p if p else 0, on=s[-keep:].mean(),
                off=r["off"][half:].mean(), ret=r["ret"][half:].mean(), gap=np.min(np.abs(n[-keep:] - r["target"])))


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", type=float, default=1.0)
    ap.add_argument("--d1", type=float, default=9.5)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--keep", type=int, default=120)
    ap.add_argument("--cells", type=int, default=200)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--procs", type=int, default=2)
    ARGS = ap.parse_args()
    ds = np.round(np.arange(ARGS.d0, ARGS.d1 + 1e-9, ARGS.step), 4)
    with Pool(ARGS.procs) as pool:
        rows = pool.map(one, ds)
    for r in rows:
        pat = "".join("1" if x > 0.5 else "0" for x in r["s"][-r["period"]:]) if r["period"] else "-"
        print(f"d = {r['d']:4.1f}: period {r['period']:3d}  pattern {pat:24s} on {r['on']:.2f}  "
              f"offloaded {r['off']:6.1f}  returned {r['ret']:6.1f}  closest to target {r['gap']:6.1f}", flush=True)
    np.savez(RESULTS / f"model_bifurcation{ARGS.tag}.npz", d=ds, N=np.array([r["N"] for r in rows]),
             s=np.array([r["s"] for r in rows]), period=np.array([r["period"] for r in rows]),
             on=np.array([r["on"] for r in rows]), off=np.array([r["off"] for r in rows]),
             ret=np.array([r["ret"] for r in rows]), gap=np.array([r["gap"] for r in rows]))
    print("saved", RESULTS / f"model_bifurcation{ARGS.tag}.npz")


if __name__ == "__main__":
    main()
