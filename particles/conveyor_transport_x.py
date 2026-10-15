"""Drift v(x) and spreading D(x) along the belt, from grain displacements over one full cycle.

    python particles/conveyor_transport_x.py --d 8 6 4 --tag _ret
"""
import argparse, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from particles.experiment import RESULTS  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--d", type=float, nargs="+", default=[8.0, 6.0, 4.0])
p.add_argument("--tag", default="_ret")
p.add_argument("--skip", type=int, default=6, help="cycles left out at the start")
args = p.parse_args()
edges = np.arange(0.0, 10.01, 1.0)
rows = []
for d in args.d:
    z = np.load(RESULTS / f"adjustable_conveyor{args.tag}_d{d:g}.npz")
    t, x, st = z["t"], z["x"], z["state"]
    T, ts = float(z["period"]), float(z["times"][0])
    starts = [np.argmin(abs(t - (k * T + ts))) for k in range(args.skip, int(t[-1] // T))]
    dx_all, x0_all, ncount = [], [], []
    for a, b in zip(starts[:-1], starts[1:]):
        on = np.all(st[a:b + 1] == 1, axis=0)             # on the belt for the whole cycle (no exit, no slot reuse)
        dx_all.append(x[b, on] - x[a, on]); x0_all.append(x[a, on]); ncount.append(on.sum())
    dxa, x0 = np.concatenate(dx_all), np.concatenate(x0_all)
    print(f"d = {d:g}: {len(dxa)} grain-cycles; whole belt v = {dxa.mean() / T:.2e}, D = {dxa.var() / (2 * T):.2e}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x0 >= lo) & (x0 < hi)
        if s.sum() > 50:
            v, D = dxa[s].mean() / T, dxa[s].var() / (2 * T)
            rows.append((d, 0.5 * (lo + hi), s.sum(), v, D))
            print(f"   x {lo:4.1f}-{hi:4.1f}: n {s.sum():6d}  v {v:.2e}  D {D:.2e}")
np.savetxt(RESULTS / f"conveyor_transport_x{args.tag}.csv", np.array(rows), delimiter=",",
           header="d,x,grain_cycles,v,D", comments="", fmt=["%g", "%g", "%d", "%.4e", "%.4e"])
