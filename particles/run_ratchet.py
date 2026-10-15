"""Ratchet conveyor on the string: carry grains with a slow tension sweep,
then switch the drive off, reset the tension and switch on again.

    python particles/run_ratchet.py                 # ratchet and a control
    python particles/run_ratchet.py --mode 5 --cycles 6 --rate 3e-4

While the drive is off the grains rest where the nodes left them.  When it
comes back at the starting tension, an antinode has moved past them, so
they slide on to the next node along.  The control sweeps slowly back and
forth with the drive on, which should give no net motion.

Writes results/fig_ratchet_mode<n>.{png,pdf} and a small .npz.
"""
import argparse
import copy
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.string_model import TensionString                              # noqa: E402
from forcing.simulation import Simulation                                    # noqa: E402
from forcing.drive import TableDrive                                         # noqa: E402
from forcing.compensation import compensation_gain                           # noqa: E402
from forcing.run_paper_figures import style                                  # noqa: E402
from particles.surface import StringSurface                                  # noqa: E402
from particles.experiment import settle, _advance, RESULTS                   # noqa: E402


def cycle_functions(kind, a0, a1, rate, t0, t_fade=30.0, t_quiet=150.0, t_gather=300.0):
    """a(t), drive envelope(t) and the cycle length, starting at t0."""
    ts = abs(a1 - a0) / rate
    if kind == "control":
        period = 2 * ts

        def a(t):
            s = (t - t0) % period if t > t0 else 0.0
            return a0 + (a1 - a0) * (s / ts if s < ts else 2.0 - s / ts)
        return a, (lambda t: 1.0), period
    period = ts + 2 * t_fade + t_quiet + t_gather

    def a(t):
        s = (t - t0) % period if t > t0 else 0.0
        if s < ts:
            return a0 + (a1 - a0) * s / ts
        q = (s - ts - t_fade - 0.5 * t_quiet) / (0.4 * t_quiet)     # reset late in the quiet time
        return a1 + (a0 - a1) * min(max(q, 0.0), 1.0)

    def env(t):
        s = (t - t0) % period if t > t0 else 0.0
        if s < ts:
            return 1.0
        if s < ts + t_fade:
            return 1.0 - (s - ts) / t_fade
        if s < ts + t_fade + t_quiet:
            return 0.0
        return min((s - ts - t_fade - t_quiet) / t_fade, 1.0)
    return a, env, period


def step_functions(slopes, t0, drive_off, t_dwell=250.0, t_move=100.0):
    """Stepping ratchet: hold each slope in turn, jumping quickly between
    them.  With ``drive_off`` the drive is faded out around each jump."""
    seg = t_dwell + t_move
    period = seg * len(slopes)

    def locate(t):
        s = (t - t0) % period if t > t0 else 0.0
        j = int(s // seg)
        return j, s - j * seg

    def a(t):
        j, s = locate(t)
        q = (s - t_dwell - 0.3 * t_move) / (0.4 * t_move)
        return slopes[j] + (slopes[(j + 1) % len(slopes)] - slopes[j]) * min(max(q, 0.0), 1.0)

    def env(t):
        if not drive_off:
            return 1.0
        j, s = locate(t)
        q = (s - t_dwell) / t_move
        if q <= 0.0:
            return 1.0
        return max(1.0 - q / 0.25, 0.0) if q < 0.5 else min((q - 0.75) / 0.25, 1.0) if q > 0.75 else 0.0
    return a, env, period


def run(kind, settled, a1, rate, cycles, slopes=None, dwell=250.0):
    s = settled
    model, mode = s.model, s.mode
    if kind.startswith("steps"):
        a, env, period = step_functions(slopes, s.sim.t, drive_off=kind.endswith("off"), t_dwell=dwell)
    else:
        a, env, period = cycle_functions(kind, s.a0, a1, rate, s.sim.t)
    gain = compensation_gain(model, mode, a, s.sim.shape, "acceleration", rate_correction=False)
    w_res = model.resonance_table(mode)
    drive = TableDrive(lambda t: w_res(a(t)), gain_of_t=lambda t: gain(t) * env(t))
    sim = Simulation(model, drive, a, mode, dt=s.sim.dt).continue_from(s.sim)
    grains = copy.deepcopy(s.grains)
    s.surface.update(sim.u, sim.v)
    rec = dict(t=[], a=[], X=[], airborne=[], w=[], peak=[])
    t_start = sim.t
    _advance(sim, s.surface, grains, t_start + cycles * period, s.record_dt, rec)
    t, av = np.array(rec["t"]), np.array(rec["a"])
    X = np.array(rec["X"])[:, :, 0]
    nodes = np.array([model.mode_nodes(x, mode) for x in av])
    on = np.array([env(x) for x in t])
    return dict(kind=kind, t=t - t_start, a=av, X=X, nodes=nodes, on=on, period=period, model=model)


def figure(ratchet, control, mode, path, titles=("A: Grain Density, Ratchet (drive off during each reset)", "B: Grain Density, Control (slow sweep both ways)"), labels=("Ratchet", "Control")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    fig, axes = plt.subplots(3, 1, figsize=(11, 12.5), gridspec_kw=dict(
        height_ratios=[1.5, 1.5, 1], hspace=0.3, left=0.09, right=0.97, top=0.96, bottom=0.06))
    L = ratchet["model"].L
    bins = np.linspace(0, L, 101)
    for ax, res, title in ((axes[0], ratchet, titles[0]), (axes[1], control, titles[1])):
        dens = np.array([np.histogram(row, bins)[0] for row in res["X"]]).T / res["X"].shape[1]
        ax.pcolormesh(res["t"], bins, np.pad(dens, ((0, 1), (0, 0))), cmap="Greys", shading="auto",
                      vmin=0, vmax=np.percentile(dens, 99.5), rasterized=True)
        nd = np.where(res["on"][:, None] > 0.5, res["nodes"], np.nan)
        for k in range(nd.shape[1]):
            ax.plot(res["t"], nd[:, k], color="k", linestyle=(0, (4, 3)), linewidth=1.3,
                    label="Node (drive on)" if k == 0 else None)
        ax.set_ylim(0, L)
        ax.set_ylabel("position x")
        ax.set_title(title)
        ax.legend(loc="upper right", facecolor="white", frameon=True, edgecolor="0.8")
    ax = axes[2]
    for res, ls, col, lab in ((ratchet, "-", "k", labels[0]), (control, (0, (4, 3)), "0.5", labels[1])):
        ax.plot(res["t"], res["X"].mean(axis=1), color=col, linestyle=ls, linewidth=2.2, label=lab)
    ax.set_ylim(0, L)
    ax.set_xlabel("time since the first sweep started")
    ax.set_ylabel("mean grain position")
    ax.set_title("C: Mean Position of All Grains")
    ax.legend(loc="upper right")
    for a_ in axes:
        a_.set_xlim(0, min(ratchet["t"][-1], control["t"][-1]))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, default=5)
    p.add_argument("--a0", type=float, default=-0.18)
    p.add_argument("--a1", type=float, default=0.18)
    p.add_argument("--rate", type=float, default=3e-4)
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--grains", type=int, default=400)
    p.add_argument("--gamma", type=float, default=4.0)
    p.add_argument("--t-settle", type=float, default=1000.0)
    p.add_argument("--steps", type=float, nargs="+", default=None,
                   help="stepping ratchet: slopes visited in turn, e.g. --steps -0.06 0 0.06")
    p.add_argument("--points", type=int, default=201)
    p.add_argument("--tag", default="", help="suffix for the output file names")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control-cycles", type=int, default=None, help="cycles of the control (0 = no control)")
    p.add_argument("--amplitude", type=float, default=0.05, help="peak displacement of the wave")
    p.add_argument("--dwell", type=float, default=250.0, help="time held at each step")
    args = p.parse_args(argv)
    if args.steps:
        args.a0 = args.steps[0]
    model = TensionString(N=args.points, beta=args.beta)
    s = settle(model, StringSurface(model), args.mode, args.a0, drive="table+gain", gamma=args.gamma,
               amplitude=args.amplitude, n_grains=args.grains, t_settle=args.t_settle, dt=0.02, seed=args.seed,
               grain_kwargs=dict(restitution=0.4, roughness=0.3))
    nd0, nd1 = model.mode_nodes(args.a0, args.mode), model.mode_nodes(args.a1, args.mode)
    print("nodes at a0:", np.round(nd0, 2), " at a1:", np.round(nd1, 2), flush=True)
    out = {}
    kinds = ("steps_off", "steps_on") if args.steps else ("ratchet", "control")
    if args.steps:
        for x in args.steps:
            print(f"  nodes at a = {x:+.3f}:", np.round(model.mode_nodes(x, args.mode), 2))
    for kind in kinds:
        n_cyc = args.cycles
        if kind == "control" and args.control_cycles is not None:
            n_cyc = args.control_cycles
            if n_cyc == 0:
                continue
        res = run(kind, s, args.a1, args.rate, n_cyc, args.steps, args.dwell)
        per = res["period"]
        marks = [int(np.searchsorted(res["t"], k * per)) for k in range(n_cyc + 1)]
        marks[-1] = len(res["t"]) - 1
        mean = [float(res["X"][i].mean()) for i in marks]
        disp = res["X"][marks[-1]] - res["X"][marks[0]]
        print(f"{kind}: cycle {per:.0f}; mean grain position at each cycle start "
              f"{np.round(mean, 2)}; median displacement {np.median(disp):+.2f}, "
              f"share moved by more than one node spacing in the transport direction "
              f"{np.mean(np.abs(disp) > np.min(np.diff(nd0))):.2f}", flush=True)
        out[kind] = res
    if args.steps:
        path = RESULTS / f"fig_ratchet_steps_mode{args.mode}"
        figure(out["steps_off"], out["steps_on"], args.mode, path,
               ("A: Grain Density, Stepping Ratchet (drive off during each jump)",
                "B: Grain Density, Stepping Ratchet (drive left on)"), ("Drive off at jumps", "Drive left on"))
    else:
        path = RESULTS / f"fig_ratchet_mode{args.mode}{args.tag}"
        if "control" in out:
            figure(out["ratchet"], out["control"], args.mode, path)
    for v in out.values():
        v["period"] = np.array([v["period"]])
    np.savez_compressed(f"{path}.npz", **{f"{k}_{f}": v[f][::5] for k, v in out.items()
                                           for f in ("t", "a", "X", "nodes", "on", "period")})
    print(f"saved {path}.png")


if __name__ == "__main__":
    main()
