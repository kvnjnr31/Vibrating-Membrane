"""Forcing runs behind the main-text figures and tables, for one mode.

    python forcing/run_main_runs.py loop  --mode 5                 # string
    python forcing/run_main_runs.py loop  --mode 5 1 --membrane
    python forcing/run_main_runs.py checks --mode 5
    python forcing/run_main_runs.py rates  --mode 5

loop    phase-locked loop, amplitude control and baseline through one sweep
checks  what the loop needs: bandwidth, calibration, starting frequency
rates   worst purity and node error against the sweep rate

Each writes results/main_<what>_<model>_mode<n>.{npz,json,csv}.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing import schedules                                              # noqa: E402
from forcing.string_model import TensionString                             # noqa: E402
from forcing.simulation import Simulation, make_drive, steady_state_energy  # noqa: E402
from forcing.sweep_experiment import run_sweep, save_results, make_model, RESULTS  # noqa: E402
from forcing.run_rate_scan import one_run                                  # noqa: E402


def setup(args):
    if args.membrane:
        model, _, dt, _ = make_model("membrane", args.beta)
        return model, tuple(args.mode), dt, -0.15, 0.15, 1.5e-4, "membrane"
    return TensionString(N=301, beta=args.beta), args.mode[0], 0.02, -0.18, 0.18, 1.8e-4, "string"


def tag(kind, mode):
    return f"{kind}_mode{''.join(str(m) for m in np.atleast_1d(mode))}"


def describe(sim, model, mode, t_spin, e_ref_of_a):
    t, a = sim.series("t"), sim.series("a")
    sel = t >= t_spin
    w_res = model.resonance_table(mode)
    det = (sim.series("w") - np.array([w_res(x) for x in a])) / model.beta
    pk = sim.series("peak")
    ratio = sim.series("energy") / e_ref_of_a(a)
    s = sim.summary(t_from=t_spin)
    return dict(final_energy=s["final_energy"], min_purity=s["min_purity"],
                max_node_error=s["max_node_error"], max_detuning=float(np.abs(det[sel]).max()),
                peak_min=float(pk[sel].min() / pk[sel][0]), peak_max=float(pk[sel].max() / pk[sel][0]),
                peak_end=float(pk[-1] / pk[sel][0]), min_energy_ratio=float(ratio[sel].min()),
                final_energy_ratio=float(ratio[-1]))


def energy_curve(model, mode, a0, a1):
    w_res = model.resonance_table(mode)
    a_ref = np.linspace(min(a0, a1), max(a0, a1), 41)
    e_ref = np.array([steady_state_energy(model, x, w_res(x)) for x in a_ref])
    return lambda a: np.interp(a, a_ref, e_ref)


def loop(args):
    model, mode, dt, a0, a1, rate, kind = setup(args)
    t_spin, t_hold = 200.0, 400.0
    ff = dict(quantity="peak")
    kw = {"pll+gain": ff, "table+gain": ff, "pll+agc": dict(quantity="peak", agc_start=0.5 * t_spin)}
    res = run_sweep(model, mode, a0, a1, rate, t_spin, t_hold,
                    ["pll", "pll+agc", "pll+gain", "table+gain", "table"], dt, steady_start=True,
                    drive_kwargs=kw)
    slow = run_sweep(model, mode, a0, a1, rate, t_spin, t_hold, ["table+gain"], dt, steady_start=True,
                     drive_kwargs={"table+gain": dict(quantity="peak", rate_correction=False)})
    res["sims"]["table+slowgain"] = slow["sims"]["table+gain"]
    name = "main_loop_" + tag(kind, mode)
    save_results(res, name)
    e_of_a = energy_curve(model, mode, a0, a1)
    nodes = np.atleast_1d(model.mode_nodes(a0, mode)[0] if kind == "membrane" else model.mode_nodes(a0, mode))
    nodes1 = np.atleast_1d(model.mode_nodes(a1, mode)[0] if kind == "membrane" else model.mode_nodes(a1, mode))
    w_res = model.resonance_table(mode)
    out = dict(model=kind, mode=[int(m) for m in np.atleast_1d(mode)], a0=a0, a1=a1, rate=rate, t_spin=t_spin,
               t_hold=t_hold, beta=model.beta, predicted_final_energy=float(e_of_a(a1)),
               node_travel=[float(v) for v in np.abs(nodes1 - nodes)],
               resonance=[float(w_res(a0)), float(w_res(0.0)), float(w_res(a1))],
               drives={k: describe(s, model, mode, t_spin, e_of_a) for k, s in res["sims"].items()})
    json.dump(out, open(RESULTS / f"{name}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def checks(args):
    model, mode, dt, a0, a1, rate, kind = setup(args)
    e_of_a = energy_curve(model, mode, a0, a1)
    out = dict(bandwidth={}, calibration={}, pull_in={})
    for bw in (0.02, 0.03, 0.07, 0.15):
        res = run_sweep(model, mode, a0, a1, rate, 200.0, 400.0, ["pll"], dt, steady_start=True,
                        verbose=False, drive_kwargs={"pll": dict(bandwidth=bw)})
        out["bandwidth"][str(bw)] = describe(res["sims"]["pll"], model, mode, 200.0, e_of_a)
        print("bandwidth", bw, out["bandwidth"][str(bw)], flush=True)
    w_res = model.resonance_table(mode)
    w0 = w_res(a0)
    sched = schedules.constant(a0)
    for cal in (True, False):
        sim = Simulation(model, make_drive("pll", model, mode, sched, calibrate=cal), sched, mode, dt=dt)
        sim.start_in_steady_state().run(1500.0)
        out["calibration"][str(cal)] = dict(
            detuning=float((sim.series("w")[-1] - w0) / model.beta),
            energy_ratio=float(sim.series("energy")[-1] / e_of_a(a0)))
        print("calibrate", cal, out["calibration"][str(cal)], flush=True)
    for bw in (0.03, 0.1):
        for off in (0.01, 0.03, 0.10):
            w_start = w0 * (1.0 + off)
            sim = Simulation(model, make_drive("pll", model, mode, sched, w0=w_start, bandwidth=bw),
                             sched, mode, dt=dt)
            sim.start_in_steady_state(w=w_start).run(2000.0)
            det = float((sim.series("w")[-1] - w0) / model.beta)
            out["pull_in"][f"bw{bw}_off{off}"] = dict(final_detuning=det, start_linewidths=w0 * off / model.beta,
                                                      energy_ratio=float(sim.series("energy")[-1] / e_of_a(a0)))
            print("pull-in", bw, off, out["pull_in"][f"bw{bw}_off{off}"], flush=True)
    json.dump(out, open(RESULTS / f"main_checks_{tag(kind, mode)}.json", "w"), indent=1)


def rates(args):
    rows = []
    _, mode, dt, a0, a1, _, kind = setup(args)
    for beta in (0.015, 0.05):
        model = TensionString(N=301, beta=beta)
        for drive in ("pll", "table"):
            for rate in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
                r = one_run(model, mode, dt, a0, a1, rate, drive, 100.0, 400.0)
                r.update(beta=beta, drive=drive, rate=rate)
                rows.append(r)
                print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    keys = ["beta", "drive", "rate", "sweep_time", "min_purity", "max_node_error",
            "max_energy_deviation", "final_energy_deviation", "peak_min", "peak_max"]
    with open(RESULTS / f"main_rates_{tag(kind, mode)}.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("what", choices=["loop", "checks", "rates"])
    p.add_argument("--mode", type=int, nargs="+", default=[5])
    p.add_argument("--membrane", action="store_true")
    p.add_argument("--beta", type=float, default=0.015)
    args = p.parse_args(argv)
    {"loop": loop, "checks": checks, "rates": rates}[args.what](args)


if __name__ == "__main__":
    main()
