"""How fast can the tension be swept before the mode degrades?

For each damping value, drive and sweep rate: start in the steady state,
sweep the slope a0 -> a1 at that rate, hold, and record the worst values
seen from the start of the sweep to the end of the hold:

    lowest purity, largest node error, largest energy deviation from the
    steady state at the current tension, and the same after the hold.

Writes results/<model>_rate_scan.{png,csv}.

    python forcing/run_rate_scan.py
    python forcing/run_rate_scan.py --model membrane --betas 0.015
    python forcing/run_rate_scan.py --drives pll table naive --rates 1e-5 1e-4 1e-3
"""
import argparse
import csv
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing import schedules                                          # noqa: E402
from forcing.simulation import Simulation, make_drive, steady_state_energy  # noqa: E402
from forcing.sweep_experiment import make_model, RESULTS               # noqa: E402

FIELDS = ["model", "beta", "drive", "rate", "sweep_time", "ring_down_times", "min_purity",
          "max_node_error", "max_energy_deviation", "final_energy_deviation", "peak_min",
          "peak_max"]


def one_run(model, mode, dt, a0, a1, rate, kind, t_spin, t_hold):
    sched = schedules.linear_ramp(a0, a1, t_spin, rate)
    duration = abs(a1 - a0) / rate
    sim = Simulation(model, make_drive(kind, model, mode, sched), sched, mode, dt=dt)
    sim.start_in_steady_state().run(t_spin + duration + t_hold)
    t, a = sim.series("t"), sim.series("a")
    sel = t >= t_spin
    w_res = model.resonance_table(mode)
    a_ref = np.linspace(min(a0, a1), max(a0, a1), 25)
    e_ref = np.array([steady_state_energy(model, x, w_res(x)) for x in a_ref])
    ratio = sim.series("energy") / np.interp(a, a_ref, e_ref)
    s = sim.summary(t_from=t_spin)
    pk = sim.series("peak")
    return dict(sweep_time=duration, ring_down_times=duration * model.beta / 2.0,
                min_purity=s["min_purity"], max_node_error=s["max_node_error"],
                max_energy_deviation=float(np.abs(ratio[sel] - 1.0).max()),
                final_energy_deviation=float(abs(ratio[-1] - 1.0)),
                peak_min=float(pk[sel].min() / pk[sel][0]), peak_max=float(pk[sel].max() / pk[sel][0]))


def plot(rows, model_name, mode, a0, a1, name, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from forcing import plotstyle as ps
    ps.apply()

    betas = sorted({r["beta"] for r in rows})
    drives = list(dict.fromkeys(r["drive"] for r in rows))
    styles = ["-", (0, (4, 3)), (0, (1, 2))]
    markers = ["o", "s", "^"]
    panels = [("min_purity", "Lowest mode purity during and after the sweep", False),
              ("max_node_error", "Largest node position error", True),
              ("max_energy_deviation", "Largest energy deviation from the steady state "
                                       "(fraction)", True)]
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.8), sharex=True,
                             gridspec_kw=dict(hspace=0.3, left=0.11, right=0.97, top=0.86, bottom=0.07))
    for ax, (key, title, logy) in zip(axes, panels):
        for kind in drives:
            for ib, beta in enumerate(betas):
                sel = sorted((r for r in rows if r["drive"] == kind and r["beta"] == beta),
                             key=lambda r: r["rate"])
                ax.plot([r["rate"] for r in sel], [max(r[key], 1e-6) for r in sel],
                        color=ps.DRIVE_COLOR[kind], linestyle=styles[ib % 3],
                        marker=markers[ib % 3], markersize=6, markeredgecolor=ps.SURFACE,
                        markeredgewidth=1.0)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.05)
        ax.set_title(title)
    axes[-1].set_xlabel("sweep rate |da/dt|")
    handles = [Line2D([], [], color=ps.DRIVE_COLOR[k], label=ps.DRIVE_LABEL[k]) for k in drives]
    handles += [Line2D([], [], color=ps.INK2, linestyle=styles[i % 3], marker=markers[i % 3],
                       markersize=6, markeredgecolor=ps.SURFACE, label=f"beta = {b:g}")
                for i, b in enumerate(betas)]
    fig.legend(handles=handles, loc="upper center", ncol=max(len(drives), 2), fontsize=9,
               bbox_to_anchor=(0.54, 0.955))
    fig.suptitle(f"{model_name.capitalize()}, mode {mode}: sweep-rate scan, slope {a0} to {a1}",
                 x=0.11, y=0.985, ha="left", fontsize=12, fontweight="bold")
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.png"
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="string", choices=["string", "membrane"])
    p.add_argument("--a0", type=float, default=None)
    p.add_argument("--a1", type=float, default=0.0)
    p.add_argument("--betas", type=float, nargs="+", default=[0.015, 0.05])
    p.add_argument("--drives", nargs="+", default=["pll", "table"])
    p.add_argument("--rates", type=float, nargs="+",
                   default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    p.add_argument("--t-spin", type=float, default=100.0)
    p.add_argument("--t-hold", type=float, default=400.0)
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)

    rows = []
    for beta in args.betas:
        model, mode, dt, a0 = make_model(args.model, beta)
        a0 = a0 if args.a0 is None else args.a0
        for kind in args.drives:
            for rate in args.rates:
                t0 = time.time()
                r = one_run(model, mode, dt, a0, args.a1, rate, kind, args.t_spin, args.t_hold)
                r.update(model=args.model, beta=beta, drive=kind, rate=rate)
                rows.append(r)
                print(f"beta {beta:<6g} {kind:8s} rate {rate:<7g} sweep lasts {r['sweep_time']:7.1f} "
                      f"({r['ring_down_times']:6.2f} ring-down times)  purity {r['min_purity']:.4f}  "
                      f"node error {r['max_node_error']:.4f}  energy dev {r['max_energy_deviation']:.3f} "
                      f"(after hold {r['final_energy_deviation']:.3f})  [{time.time() - t0:.0f} s]",
                      flush=True)
    name = f"{args.model}_rate_scan"
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{name}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print("saved", RESULTS / f"{name}.csv")
    print("saved", plot(rows, args.model, mode, a0, args.a1, name, args.show))
    return rows


if __name__ == "__main__":
    main()
