"""Do the Bessel solutions of the thesis describe the simulated string?

1. Compares the analytic eigenvalues, mode shapes and node positions
   (bessel.py) with the modes of the finite-difference operator that the
   simulation integrates.
2. Runs the tension sweep with the baseline drive reading its resonance from
   the Bessel eigenvalues instead of the discrete ones.

Writes results/string_bessel_check.csv.

    python forcing/run_bessel_check.py
    python forcing/run_bessel_check.py --points 201 --mode 4
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing import bessel, schedules                                   # noqa: E402
from forcing.drive import TableDrive                                    # noqa: E402
from forcing.simulation import Simulation, make_drive, steady_state_energy  # noqa: E402
from forcing.string_model import TensionString                          # noqa: E402
from forcing.sweep_experiment import RESULTS                            # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, default=3)
    p.add_argument("--points", type=int, default=301)
    p.add_argument("--beta", type=float, default=0.015)
    p.add_argument("--a0", type=float, default=-0.18)
    p.add_argument("--a1", type=float, default=0.0)
    p.add_argument("--rate", type=float, default=1e-4)
    p.add_argument("--no-sweep", action="store_true")
    args = p.parse_args(argv)

    model = TensionString(N=args.points, beta=args.beta)
    n, beta = args.mode, args.beta
    print(f"String, mode {n}, {args.points} grid points, beta = {beta}")
    print(f"{'a':>6s} {'lam (Bessel)':>13s} {'lam (grid)':>12s} {'frequency gap':>14s} "
          f"{'in linewidths':>14s} {'node gap':>10s} {'shape gap':>10s}")
    rows = []
    for a in np.round(np.linspace(-0.18, 0.18, 13), 3):
        lb, ld = bessel.eigenvalue(a, n), model.eigenvalue(a, n)
        dw = np.sqrt(ld - 0.5 * beta ** 2) - np.sqrt(lb - 0.5 * beta ** 2)
        node_gap = float(np.abs(bessel.nodes(a, n) - model.mode_nodes(a, n)).max()) if n > 1 else 0.0
        pb, pd = bessel.mode_shape(model.x, a, n), model.mode_shape(a, n)
        shape_gap = float(np.abs(pb - pd).max() / np.abs(pb).max())
        rows.append(dict(a=a, lam_bessel=lb, lam_grid=ld, frequency_gap=dw,
                         frequency_gap_linewidths=dw / beta, node_gap=node_gap, shape_gap=shape_gap))
        print(f"{a:+6.2f} {lb:13.6f} {ld:12.6f} {dw:14.2e} {dw / beta:14.4f} {node_gap:10.1e} {shape_gap:10.1e}")
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "string_bessel_check.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("saved", RESULTS / "string_bessel_check.csv")

    print("\nGrid convergence at a = -0.18: (grid frequency - Bessel frequency) / beta, modes 1..6")
    lb = bessel.eigenvalues(-0.18, 6)
    for N in (101, 201, 301, 601):
        ld = TensionString(N=N, beta=beta).eigen(-0.18, 6)[0]
        print(f"  {N:4d} points: " + "  ".join(f"{v:+.4f}" for v in (np.sqrt(ld) - np.sqrt(lb)) / beta))

    if args.no_sweep:
        return rows
    print(f"\nSweep a {args.a0} -> {args.a1} at {args.rate:g}, baseline drive:")
    t_spin, t_hold = 200.0, 300.0
    sched = schedules.linear_ramp(args.a0, args.a1, t_spin, args.rate)
    t_end = t_spin + abs(args.a1 - args.a0) / args.rate + t_hold
    w_bessel = bessel.resonance_table(n, beta)
    w_grid = model.resonance_table(n)
    target = steady_state_energy(model, args.a1, w_grid(args.a1))
    drives = {"resonance from the grid modes": make_drive("table", model, n, sched),
              "resonance from the Bessel solution": TableDrive(lambda t: w_bessel(sched(t)))}
    for label, drive in drives.items():
        sim = Simulation(model, drive, sched, n).start_in_steady_state(drive.w)
        sim.run(t_end)
        s = sim.summary(t_from=t_spin)
        print(f"  {label:36s} final energy {s['final_energy']:8.1f} (prediction {target:.1f})   "
              f"lowest purity {s['min_purity']:.4f}   largest node error {s['max_node_error']:.4f}")
    return rows


if __name__ == "__main__":
    main()
