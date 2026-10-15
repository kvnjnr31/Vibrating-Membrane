"""Shared driver for the tension-sweep experiments (string and membrane)."""
import pathlib
import time

import numpy as np

from . import schedules
from .simulation import Simulation, make_drive, steady_state_energy

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def run_sweep(model, mode, a0, a1, rate, t_spin, t_hold, drives, dt, F0=1.0,
              shape=None, schedule_kind="linear", verbose=True, steady_start=False,
              drive_kwargs=None):
    """Settle on resonance at a0, sweep to a1, hold.  One run per drive.

    steady_start : begin in the computed steady state, so t_spin only needs
                   to be long enough to show a flat baseline
    drive_kwargs : {drive name: extra arguments for make_drive}
    """
    drive_kwargs = drive_kwargs or {}
    duration = abs(a1 - a0) / rate
    if schedule_kind == "linear":
        sched = schedules.linear_ramp(a0, a1, t_spin, rate)
    elif schedule_kind == "smooth":
        sched = schedules.smooth_ramp(a0, a1, t_spin, duration)
    elif schedule_kind == "constant-frequency-rate":
        sched = schedules.constant_frequency_rate(model, mode, a0, a1, t_spin, duration)
    else:
        raise ValueError(schedule_kind)
    t_end = t_spin + duration + t_hold

    sims = {}
    for kind in drives:
        t0 = time.time()
        drive = make_drive(kind, model, mode, sched, shape=shape, **drive_kwargs.get(kind, {}))
        sim = Simulation(model, drive, sched, mode, F0=F0, shape=shape, dt=dt)
        if steady_start:
            sim.start_in_steady_state()
        sim.run(t_end)
        sims[kind] = sim
        if verbose:
            s = sim.summary(t_from=t_spin)
            print(f"{kind:8s} final energy {s['final_energy']:10.1f}   min purity {s['min_purity']:.4f}   "
                  f"max node error {s['max_node_error']:.4f}   "
                  f"nodes lost in {100 * s['fraction_cycles_nodes_lost']:.0f}% of cycles   "
                  f"({time.time() - t0:.0f} s)")

    # steady-state energy a perfectly tracked sweep would have
    w_res = model.resonance_table(mode)
    a_ref = np.linspace(a0, a1, 25)
    e_ref = np.array([steady_state_energy(model, a, w_res(a), F0, shape) for a in a_ref])
    return dict(sims=sims, schedule=sched, t_spin=t_spin, t_end=t_end, a_ref=a_ref, e_ref=e_ref,
                a0=a0, a1=a1, mode=mode, model=model)


def save_results(res, name):
    RESULTS.mkdir(exist_ok=True)
    out = dict(a_ref=res["a_ref"], e_ref=res["e_ref"])
    for kind, sim in res["sims"].items():
        meas, exp = sim.node_tracks()
        out.update({f"{kind}_t": sim.series("t"), f"{kind}_a": sim.series("a"),
                    f"{kind}_energy": sim.series("energy"), f"{kind}_purity": sim.series("purity"),
                    f"{kind}_peak": sim.series("peak"), f"{kind}_gain": sim.series("gain"),
                    f"{kind}_peak_acceleration": sim.series("peak") * sim.series("w") ** 2,
                    f"{kind}_w": sim.series("w"),
                    f"{kind}_nodes": meas, f"{kind}_expected_nodes": exp})
    path = RESULTS / f"{name}.npz"
    np.savez(path, **out)
    return path


PANEL_TITLES = {
    "energy": "Total energy",
    "purity": "Mode purity (share of the oscillation in the target mode)",
    "nodes": "Node positions (gaps: the pattern no longer has the target number of nodes)",
    "peak": "Peak displacement, relative to its value before the sweep",
    "acceleration": "Peak acceleration, relative to its value before the sweep",
    "gain": "Force gain applied by the drive",
    "detuning": "Instantaneous drive frequency minus the resonance, in linewidths (beta)",
}


def plot_sweep(res, title, name, show=False, panels=("energy", "purity", "nodes")):
    """One panel per quantity, all sharing the time axis."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import plotstyle as ps
    ps.apply()

    sims, model, mode = res["sims"], res["model"], res["mode"]
    n = len(panels)
    top = 1.0 - (0.95 + 0.22 * ((len(sims) + 2) // 2)) / (2.8 * n)
    fig, axes = plt.subplots(n, 1, figsize=(8.6, 2.8 * n), sharex=True, squeeze=False,
                             gridspec_kw=dict(hspace=0.3, left=0.1, right=0.97, top=top,
                                              bottom=0.2 / n + 0.02))
    axes = axes[:, 0]
    any_sim = next(iter(sims.values()))
    t, a_t = any_sim.series("t"), any_sim.series("a")
    dash = dict(color=ps.REFERENCE, linestyle=(0, (4, 3)), linewidth=1.4, zorder=5)
    w_res = model.resonance_table(mode)
    i_spin = max(int(np.searchsorted(t, res["t_spin"])) - 1, 0)
    ref_label = "Prediction for the target mode at the current tension"
    have_ref = False

    for ax, panel in zip(axes, panels):
        ax.set_title(PANEL_TITLES[panel])
        if panel == "energy":
            order = np.argsort(res["a_ref"])
            ax.plot(t, np.interp(a_t, res["a_ref"][order], res["e_ref"][order]),
                    label=None if have_ref else ref_label, **dash)
            have_ref = True
            for kind, sim in sims.items():
                ax.plot(sim.series("t"), sim.series("energy"), color=ps.DRIVE_COLOR[kind], linestyle=ps.DRIVE_LINESTYLE.get(kind, "-"))
            ax.set_ylim(bottom=0)
        elif panel == "purity":
            for kind, sim in sims.items():
                ax.plot(sim.series("t"), sim.series("purity"), color=ps.DRIVE_COLOR[kind], linestyle=ps.DRIVE_LINESTYLE.get(kind, "-"))
            ax.set_ylim(0, 1.05)
        elif panel == "nodes":
            exp = any_sim.node_tracks()[1]
            for k in range(exp.shape[1]):
                ax.plot(t, exp[:, k], label=None if have_ref or k else ref_label, **dash)
            have_ref = True
            for kind, sim in sims.items():
                meas = sim.node_tracks()[0]
                for k in range(meas.shape[1]):
                    ax.plot(sim.series("t"), meas[:, k], color=ps.DRIVE_COLOR[kind], linestyle=ps.DRIVE_LINESTYLE.get(kind, "-"), linewidth=1.6)
            ax.set_ylabel("x")
        elif panel in ("peak", "acceleration"):
            for kind, sim in sims.items():
                pk = sim.series("peak") * (sim.series("w") ** 2 if panel == "acceleration" else 1.0)
                ax.plot(sim.series("t"), pk / pk[min(i_spin, len(pk) - 1)], color=ps.DRIVE_COLOR[kind], linestyle=ps.DRIVE_LINESTYLE.get(kind, "-"))
            ax.set_ylim(bottom=0)
        elif panel == "gain":
            for kind, sim in sims.items():
                ax.plot(sim.series("t"), sim.series("gain"), color=ps.DRIVE_COLOR[kind], linestyle=ps.DRIVE_LINESTYLE.get(kind, "-"))
            ax.set_ylim(bottom=0)
        elif panel == "detuning":
            worst = 0.0
            for kind, sim in sims.items():
                dw = (sim.series("w") - np.array([w_res(a) for a in sim.series("a")])) / model.beta
                ax.plot(sim.series("t"), dw, color=ps.DRIVE_COLOR[kind],
                        linestyle=ps.DRIVE_LINESTYLE.get(kind, "-"))
                worst = max(worst, float(np.nanmax(np.abs(dw))))
            lim = 1.5 if worst <= 1.5 else min(1.1 * worst, 25.0)
            ax.set_ylim(-lim, lim)
        ax.axvline(res["t_spin"], color=ps.AXIS, linewidth=1.0)
    axes[-1].set_xlabel("time")
    axes[0].annotate("sweep starts", (res["t_spin"], 0.0), xycoords=("data", "axes fraction"),
                     ha="right", xytext=(-5, 6), textcoords="offset points", fontsize=8.5,
                     color=ps.MUTED)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ps.DRIVE_COLOR[k], linestyle=ps.DRIVE_LINESTYLE.get(k, "-"),
                      label=ps.DRIVE_LABEL[k]) for k in sims]
    if have_ref:
        handles.insert(0, Line2D([], [], label=ref_label, **dash))
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.53, 1.0 - 0.36 / (2.8 * n)))
    fig.suptitle(title, x=0.1, y=1.0 - 0.07 / n, ha="left", fontsize=12, fontweight="bold")
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.png"
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def make_model(kind, beta, points=None, size=None):
    """String or membrane with the default grids used by the experiments."""
    from .string_model import TensionString
    from .membrane_model import TensionMembrane
    if kind == "string":
        return TensionString(N=points or 301, beta=beta), 3, 0.02, -0.18
    Lx, Ly = size or (10.0, 7.0)
    Nx, Ny = points or (81, 57)
    return TensionMembrane(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, beta=beta), (3, 1), 0.04, -0.15
