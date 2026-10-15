"""How well does each node hold its grains at a fixed tension slope?

    python particles/run_escape.py --mode 11 --slope 0.1 --amplitude 0.014

Settles the grains, tags those sitting at each node, and reports the share
still at the same node later, next to the height of the neighbouring
antinodes and the width of the quiet stretch (local acceleration below g).
Writes results/escape_mode<n>_slope<a>.csv.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.string_model import TensionString                  # noqa: E402
from particles.surface import StringSurface                     # noqa: E402
from particles.experiment import settle, RESULTS                # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, default=11)
    p.add_argument("--slope", type=float, default=0.1)
    p.add_argument("--amplitude", type=float, default=0.014)
    p.add_argument("--gamma", type=float, default=4.0)
    p.add_argument("--grains", type=int, default=800)
    p.add_argument("--t-tag", type=float, default=1000.0)
    p.add_argument("--t-watch", type=float, default=2000.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    model = TensionString(N=201, beta=0.05)
    s = settle(model, StringSurface(model), args.mode, args.slope, drive="table+gain",
               gamma=args.gamma, amplitude=args.amplitude, n_grains=args.grains,
               t_settle=args.t_tag + args.t_watch, dt=0.02, seed=args.seed,
               grain_kwargs=dict(restitution=0.4, roughness=0.3))
    t = np.array(s.rec["t"])
    X = np.array(s.rec["X"])[:, :, 0]
    nodes = np.asarray(model.mode_nodes(args.slope, args.mode))
    phi = np.abs(model.mode_shape(args.slope, args.mode))
    rel = phi / phi.max()
    i0 = int(np.searchsorted(t, s.t_spin + args.t_tag))
    marks = [int(np.searchsorted(t, s.t_spin + args.t_tag + f * args.t_watch)) for f in (0.25, 0.5, 1.0)]
    marks[-1] = min(marks[-1], len(t) - 1)
    edges = np.concatenate(([0.0], nodes, [model.L]))
    rows = []
    for k, xn in enumerate(nodes):
        left = rel[(model.x > edges[k]) & (model.x < xn)].max()
        right = rel[(model.x > xn) & (model.x < edges[k + 2])].max()
        near = np.abs(model.x - xn) < 0.6 * min(xn - edges[k], edges[k + 2] - xn)
        quiet = 0.5 * np.sum(near & (rel * args.gamma < 1.0)) * model.dx
        tag = np.abs(X[i0] - xn) < 0.15
        kept = [float(np.mean(np.abs(X[i][tag] - xn) < 0.2)) if tag.any() else np.nan for i in marks]
        rows.append([xn, 0.5 * (left + right), quiet, int(tag.sum()), *kept])
    rows = np.array(rows)
    path = RESULTS / f"escape_mode{args.mode}_slope{args.slope:+.2f}.csv"
    head = "node_x,neighbour_antinode_height,quiet_half_width,tagged,kept_quarter,kept_half,kept_full"
    np.savetxt(path, rows, delimiter=",", header=head, comments="", fmt="%.4g")
    print(f"slope {args.slope:+.2f}, watched for {args.t_watch:g} time units")
    print(" node x  antinode  quiet  tagged  kept(1/4, 1/2, full)")
    for r in rows:
        print(f" {r[0]:5.2f}   {r[1]:5.2f}    {r[2]:5.3f}  {int(r[3]):4d}    {r[4]:.2f} {r[5]:.2f} {r[6]:.2f}")
    c = np.corrcoef(rows[:, 1], rows[:, 6])[0, 1]
    print(f"correlation between antinode height and share kept: {c:+.2f}")
    print("saved", path)


if __name__ == "__main__":
    main()
