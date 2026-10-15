"""Write the look-up table for one membrane mode before any simulation.

    python forcing/run_make_table.py                 # mode (3,1)
    python forcing/run_make_table.py --mode 3 2 --check

Columns: slope a, eigenvalue, resonant frequency, steady-state peak
response per unit forcing (for the feed-forward gain) and the x positions
of the interior nodal lines.  --check compares the eigenvalues with the
closed-form (Kummer function) solution at a few slopes.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.sweep_experiment import make_model          # noqa: E402
from forcing.compensation import response                # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def table_path(mode):
    return RESULTS / f"membrane_table_mode{mode[0]}{mode[1]}.csv"


def make_table(model, mode, samples=401, a_max=None):
    a_max = 0.97 * model.a_limit if a_max is None else a_max
    a = np.linspace(-a_max, a_max, samples)
    shape = model.uniform_shape()
    rows = []
    for x in a:
        lam = model.eigenvalue(x, mode)
        nodes = model.mode_nodes(x, mode)
        xs = np.atleast_1d(nodes[0] if isinstance(nodes, tuple) else nodes)
        rows.append([x, lam, np.sqrt(lam - 0.5 * model.beta ** 2),
                     response(model, x, mode, shape, "peak"), *xs[: mode[0] - 1]])
    return np.array(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, nargs=2, default=[3, 1])
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--samples", type=int, default=401)
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    model, _, _, _ = make_model("membrane", args.beta)
    mode = tuple(args.mode)
    tab = make_table(model, mode, args.samples)
    cols = ["a", "eigenvalue", "omega_res", "peak_response"] + [f"node_x{i + 1}" for i in range(tab.shape[1] - 4)]
    RESULTS.mkdir(exist_ok=True)
    path = table_path(mode)
    with open(path, "w") as fh:
        fh.write(f"# mode ({mode[0]},{mode[1]}) grid {model.Nx}x{model.Ny} beta {model.beta}\n")
        fh.write(",".join(cols) + "\n")
        for r in tab:
            fh.write(",".join(f"{v:.10g}" for v in r) + "\n")
    print(f"saved {path}  ({len(tab)} slopes from {tab[0, 0]:+.3f} to {tab[-1, 0]:+.3f})")
    print(f"resonance: {tab[:, 2].min():.4f} to {tab[:, 2].max():.4f}")
    if args.check:
        from forcing.kummer import eigenvalue
        from forcing.membrane_model import TensionMembrane
        fine = TensionMembrane(Lx=model.Lx, Ly=model.Ly, Nx=641, Ny=449, beta=model.beta)
        print("    a     grid(run)   grid(fine)   Kummer     run-Kummer in linewidths")
        for x in (-0.18, -0.1, -0.05, 0.05, 0.1, 0.18):
            lg, lf = model.eigenvalue(x, mode), fine.eigenvalue(x, mode)
            lk = eigenvalue(x, mode, model.Lx, model.Ly, guess=lf)
            print(f"  {x:+.2f}  {lg:.6f}   {lf:.6f}   {lk:.6f}   "
                  f"{(np.sqrt(lg) - np.sqrt(lk)) / model.beta:+.4f}")


if __name__ == "__main__":
    main()
