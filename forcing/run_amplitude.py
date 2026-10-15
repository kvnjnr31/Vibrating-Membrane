"""Hold the vibration amplitude constant while the tension is swept.

Compares, over the same sweep:
    pll          phase-locked loop alone (the amplitude drifts)
    pll+gain     with the feed-forward gain from the amplitude table
    pll+agc      with its own amplitude control loop (no table at all)
    table+gain   baseline: table lookup with the feed-forward gain

Writes results/<model>_amplitude.{png,npz}.

    python forcing/run_amplitude.py
    python forcing/run_amplitude.py --model membrane
    python forcing/run_amplitude.py --quantity acceleration --rate 1e-3
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.sweep_experiment import run_sweep, save_results, plot_sweep, make_model  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="string", choices=["string", "membrane"])
    p.add_argument("--a0", type=float, default=None)
    p.add_argument("--a1", type=float, default=0.0)
    p.add_argument("--rate", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.015)
    p.add_argument("--quantity", default="peak", choices=["peak", "acceleration"],
                   help="what is held constant: peak displacement or peak acceleration")
    p.add_argument("--no-rate-correction", action="store_true",
                   help="use only the slow-sweep (quasi-static) feed-forward gain")
    p.add_argument("--t-spin", type=float, default=200.0)
    p.add_argument("--t-hold", type=float, default=300.0)
    p.add_argument("--drives", nargs="+", default=["pll", "pll+gain", "pll+agc", "table+gain"])
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)

    model, mode, dt, a0 = make_model(args.model, args.beta)
    a0 = a0 if args.a0 is None else args.a0
    print(f"{args.model}, mode {mode}: a {a0} -> {args.a1} at {args.rate:g} per unit time, "
          f"holding the peak {args.quantity if args.quantity != 'peak' else 'displacement'}")
    ff = dict(quantity=args.quantity, rate_correction=not args.no_rate_correction)
    res = run_sweep(model, mode, a0, args.a1, args.rate, args.t_spin, args.t_hold, args.drives, dt,
                    steady_start=True, verbose=False,
                    drive_kwargs={"pll+gain": ff, "table+gain": ff,
                                  "pll+agc": dict(quantity=args.quantity,
                                                  agc_start=0.5 * args.t_spin)})
    print(f"{'drive':11s} {'held quantity (min .. max / start)':36s} {'final gain':>10s} "
          f"{'lowest purity':>14s} {'node error':>11s}")
    for kind, sim in res["sims"].items():
        t = sim.series("t")
        q = sim.series("peak") * (sim.series("w") ** 2 if args.quantity == "acceleration" else 1.0)
        sel = t >= args.t_spin
        rel = q[sel] / q[sel][0]
        s = sim.summary(t_from=args.t_spin)
        print(f"{kind:11s} {rel.min():.3f} .. {rel.max():.3f}{'':21s} {sim.series('gain')[-1]:10.3f} "
              f"{s['min_purity']:14.4f} {s['max_node_error']:11.4f}")
    name = f"{args.model}_amplitude"
    print("saved", save_results(res, name))
    print("saved", plot_sweep(res, f"{args.model.capitalize()}, mode {mode}: holding the amplitude, "
                                   f"slope {a0} to {args.a1}", name, args.show,
                              panels=(args.quantity, "gain", "purity", "nodes")))
    return res


if __name__ == "__main__":
    main()
