"""Tension sweep: does the mode survive and do the nodes move?

Starts in the steady state at slope a0, sweeps the slope to a1 and holds.
Compares the phase-locked loop with the table-lookup baseline.
Writes results/<model>_sweep.{png,npz}.

    python forcing/run_sweep.py
    python forcing/run_sweep.py --model membrane
    python forcing/run_sweep.py --rate 1e-3 --bandwidth 0.05
    python forcing/run_sweep.py --legacy          # fixed frequency and direct substitution
    python forcing/run_sweep.py --legacy --velocity-feedback   # ... plus plain velocity feedback
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.sweep_experiment import run_sweep, save_results, plot_sweep, make_model  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="string", choices=["string", "membrane"])
    p.add_argument("--a0", type=float, default=None)
    p.add_argument("--a1", type=float, default=0.0)
    p.add_argument("--rate", type=float, default=1e-4, help="|da/dt| during the sweep")
    p.add_argument("--beta", type=float, default=0.015)
    p.add_argument("--bandwidth", type=float, default=0.1, help="loop bandwidth of the PLL")
    p.add_argument("--t-spin", type=float, default=200.0, help="time before the sweep")
    p.add_argument("--t-hold", type=float, default=300.0, help="time held at a1 afterwards")
    p.add_argument("--schedule", default="linear",
                   choices=["linear", "smooth", "constant-frequency-rate"])
    p.add_argument("--drives", nargs="+", default=None)
    p.add_argument("--legacy", action="store_true",
                   help="run the failed approaches (fixed, naive) against the baseline")
    p.add_argument("--velocity-feedback", action="store_true",
                   help="with --legacy: add plain velocity feedback, which jumps to another mode")
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)

    legacy = ["fixed", "naive"] + (["self"] if args.velocity_feedback else []) + ["table"]
    drives = args.drives or (legacy if args.legacy else ["pll", "table"])
    model, mode, dt, a0 = make_model(args.model, args.beta)
    a0 = a0 if args.a0 is None else args.a0
    print(f"{args.model}, mode {mode}: a {a0} -> {args.a1} at {args.rate:g} per unit time")
    if args.model == "membrane":
        for a in (a0, 0.5 * (a0 + args.a1), args.a1):
            other, gap = model.nearest_neighbour(a, mode)
            flag = "   <-- within 3 linewidths" if gap < 3 * args.beta else ""
            print(f"  a = {a:+.3f}: nearest other mode {other} is {gap:.4f} away in frequency{flag}")
    res = run_sweep(model, mode, a0, args.a1, args.rate, args.t_spin, args.t_hold, drives, dt,
                    schedule_kind=args.schedule, steady_start=True, verbose=False,
                    drive_kwargs={"pll": dict(bandwidth=args.bandwidth)})
    w_res = model.resonance_table(mode)
    print(f"{'drive':7s} {'final energy':>13s} {'lowest purity':>14s} {'node error':>11s} "
          f"{'largest detuning / beta':>24s}")
    for kind, sim in res["sims"].items():
        s = sim.summary(t_from=args.t_spin)
        sel = sim.series("t") >= args.t_spin
        dw = (sim.series("w") - np.array([w_res(a) for a in sim.series("a")]))[sel]
        print(f"{kind:7s} {s['final_energy']:13.1f} {s['min_purity']:14.4f} {s['max_node_error']:11.4f} "
              f"{np.abs(dw).max() / model.beta:24.3f}")
    print(f"(prediction for the final energy: {res['e_ref'][-1]:.1f})")
    name = f"{args.model}_sweep" + ("_legacy" if args.legacy else "") \
        + ("_velocity_feedback" if args.legacy and args.velocity_feedback else "")
    print("saved", save_results(res, name))
    panels = ("detuning", "energy", "purity", "nodes") if args.legacy else ("energy", "detuning", "purity", "nodes")
    what = "approaches that fail" if args.legacy else "phase-locked loop and baseline"
    print("saved", plot_sweep(res, f"{args.model.capitalize()}, mode {mode}: {what}, "
                                   f"slope {a0} to {args.a1}", name, args.show, panels=panels))
    return res


if __name__ == "__main__":
    main()
