"""Continuum model of the adjustable conveyor, and its comparison with the
membrane simulation.

    python particles/conveyor_model.py compare --d 8 6 4 --tag _ret      # model against the runs
    python particles/conveyor_model.py fit --d 8 6 4 --tag _ret          # fit v and D, then compare

The model (feeder at x_f = L - d, belt 0 < x < L):

    valve      dF/dt = G,   dG/dt = (F_max u - F - 2 tau_f G) / tau_f^2
    belt       dn/dt = -v dn/dx + D d2n/dx2 + F(t) s(x)       s: the feeder box, unit area
               n(0) = 0 (return slot),  n(L) = 0 (offload slot)
    ends       B = D dn/dx at x = 0 (returned),  Q = -D dn/dx at x = L (offloaded)
    hopper     dR/dt = B - F H(R),  fresh supply S = F (1 - H(R))
    switch     at the end of each sweep: u = 1 if N = int n dx < N_target, else 0

The belt equation is solved with finite volumes (upwind drift, central
spreading, exact zero-density ends).  The switching times, the valve and the
feeder box are taken from the simulation run being compared.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import style                                  # noqa: E402
from particles.experiment import RESULTS                                    # noqa: E402

L = 10.0


def solve(v, D, box, f_max, tau_f, target, period, t_sweep, t_end, cells=200, dt=1.0, n0=0.0, record=10.0):
    """Integrate the model.  Returns a dict of time series sampled every `record`."""
    dx = L / cells
    xc = (np.arange(cells) + 0.5) * dx
    src = ((xc + 0.5 * dx > box[0]) & (xc - 0.5 * dx < box[1])).astype(float)
    overlap = np.clip(np.minimum(xc + 0.5 * dx, box[1]) - np.maximum(xc - 0.5 * dx, box[0]), 0.0, None)
    src = overlap / overlap.sum() / dx                         # unit area, spread over the feeder box
    dt = min(dt, 0.4 * dx * dx / D, 0.5 * dx / max(v, 1e-12))
    n = np.full(cells, n0 / L)
    F = G = R = 0.0
    u = 1.0
    out = dict(t=[], F=[], N=[], R=[], B=[], Q=[], S=[], u=[])
    cum = dict(B=0.0, Q=0.0, S=0.0)
    steps = int(round(t_end / dt))
    rec_n = max(1, int(round(record / dt)))
    next_switch = t_sweep
    for i in range(steps):
        t = i * dt
        if t >= next_switch:
            u = 1.0 if n.sum() * dx < target else 0.0
            next_switch += period
        # valve
        G += dt * (f_max * u - F - 2 * tau_f * G) / tau_f ** 2
        F = min(max(F + dt * G, 0.0), f_max)
        # belt: fluxes at cell faces, J = v n_upwind - D dn/dx; zero density beyond both ends
        ng = np.concatenate(([-n[0]], n, [-n[-1]]))            # ghost cells give n = 0 on the end faces
        upw = np.where(v >= 0, ng[:-1], ng[1:])
        upw[0] = 0.0 if v >= 0 else n[0]
        upw[-1] = n[-1] if v >= 0 else 0.0
        J = v * upw - D * (ng[1:] - ng[:-1]) / dx
        b, q = -J[0], J[-1]                                    # out through the left end, out through the right end
        n = n + dt * (-(J[1:] - J[:-1]) / dx + F * src)
        n = np.maximum(n, 0.0)
        hopper_on = R > 1e-9
        R = max(R + dt * (b - (F if hopper_on else 0.0)), 0.0)
        s_rate = 0.0 if hopper_on else F
        cum["B"] += b * dt; cum["Q"] += q * dt; cum["S"] += s_rate * dt
        if i % rec_n == 0:
            out["t"].append(t); out["F"].append(F); out["N"].append(n.sum() * dx); out["R"].append(R)
            out["B"].append(cum["B"]); out["Q"].append(cum["Q"]); out["S"].append(cum["S"]); out["u"].append(u)
    return {k: np.array(v_) for k, v_ in out.items()}


def load_run(d, tag):
    z = np.load(RESULTS / f"adjustable_conveyor{tag}_d{d:g}_series.npz")
    dt = float(z["dt"])
    t = np.arange(len(z["feed_rate"])) * dt
    return z, t


def run_model(z, v, D, cells=200, dt=1.0):
    t_end = len(z["feed_rate"]) * float(z["dt"])
    return solve(v, D, z["box"], float(z["feed_max"]), float(z["valve_tau"]), float(z["target"]), float(z["period"]),
                 float(z["times"][0]), t_end, cells=cells, dt=dt)


def misfit(z, t, m):
    """RMS difference of the grain count, relative to the target, over the whole run."""
    sim = z["count"][::500].astype(float)
    ts = t[::500]
    mod = np.interp(ts, m["t"], m["N"])
    return np.sqrt(np.mean((sim - mod) ** 2)) / float(z["target"])


def score(z, t, m):
    """Fit target: grain-count misfit plus the errors in total offloaded and total returned, as shares of grains fed."""
    fed = float(len(z["fed"]))
    return (misfit(z, t, m) + abs(m["Q"][-1] - len(z["exits"])) / fed + abs(m["B"][-1] - len(z["returns"])) / fed)


def summary(z, t, m, label):
    period = float(z["period"])
    T = t[-1]
    half = t > T / 2
    sim = dict(feed=z["feed_rate"][half].mean(), N=z["count"][half].mean(),
               off=len(z["exits"]), ret=len(z["returns"]), fed=len(z["fed"]), fresh=int(len(z["fed"]) - z["fed_returned"].sum()))
    mh = m["t"] > T / 2
    mod = dict(feed=m["F"][mh].mean(), N=m["N"][mh].mean(), off=m["Q"][-1], ret=m["B"][-1], fed=np.trapezoid(m["F"], m["t"]), fresh=m["S"][-1])
    ons = lambda u: np.flatnonzero(np.diff(u) > 0.5)           # noqa: E731
    print(f"{label}: second-half feed rate sim {sim['feed']:.3f} model {mod['feed']:.3f}; grains sim {sim['N']:.0f} model {mod['N']:.0f}; "
          f"offloaded sim {sim['off']} model {mod['off']:.0f}; returned sim {sim['ret']} model {mod['ret']:.0f}; "
          f"fed sim {sim['fed']} model {mod['fed']:.0f}; feeder switched on sim {int((np.diff(z['command'].astype(int)) > 0).sum())} "
          f"model {len(ons(m['u']))} times; misfit {misfit(z, t, m):.3f}")
    return sim, mod


def figure(args, params, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    nd = len(args.d)
    height = 3.4 * nd + 0.7
    fig, axes = plt.subplots(nd, 2, figsize=(11, height), sharex=True,
                             gridspec_kw=dict(left=0.08, right=0.985, top=1 - 0.55 / height, bottom=0.8 / height, hspace=0.42, wspace=0.24))
    axes = np.atleast_2d(axes)
    for r, d in enumerate(args.d):
        z, t = load_run(d, args.tag)
        period = float(z["period"])
        v, D = params[d]
        m = run_model(z, v, D)
        a0, a1 = axes[r]
        a0.plot(t[::50] / period, z["feed_rate"][::50], color="k", linewidth=1.5, label="Simulation")
        a0.plot(m["t"] / period, m["F"], color="0.5", linewidth=1.6, linestyle=(0, (4, 3)), label="Model")
        a0.set_ylim(0, 1.15 * float(z["feed_max"])); a0.set_ylabel("feed rate")
        a0.set_title(f"{'ABCDEF'[2 * r]}: Feed Rate, d = {d:g}", fontsize=15)
        a1.plot(t[::500] / period, z["count"][::500], color="k", linewidth=1.5, label="Simulation")
        a1.plot(m["t"] / period, m["N"], color="0.5", linewidth=1.6, linestyle=(0, (4, 3)), label="Model")
        a1.axhline(float(z["target"]), color="0.75", linewidth=1.0, linestyle=":")
        a1.set_ylim(0, 1.3 * max(z["count"].max(), m["N"].max())); a1.set_ylabel("grains on the conveyor")
        a1.set_title(f"{'ABCDEF'[2 * r + 1]}: Grains on the Conveyor, d = {d:g}", fontsize=15)
        for a in (a0, a1):
            a.set_xlim(0, round(t[-1] / period))
        if r == 0:
            a1.legend(loc="lower right", fontsize=12, ncol=2)
    for a in axes[-1]:
        a.set_xlabel("time (cycles)")
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"{name}.{ext}")
    print("saved", RESULTS / f"{name}.png")


def compare(args):
    params = {d: (args.v, args.D) for d in args.d}
    for d in args.d:
        z, t = load_run(d, args.tag)
        summary(z, t, run_model(z, args.v, args.D), f"d = {d:g}, v = {args.v:.2e}, D = {args.D:.2e}")
    figure(args, params, f"fig_conveyor_model{args.tag}")


def fit(args):
    """One (v, D) for all runs, by grid search on the score (grain count, offloaded and returned totals)."""
    vs = np.geomspace(*args.v_range, args.grid)
    Ds = np.geomspace(*args.D_range, args.grid)
    runs = [load_run(d, args.tag) for d in args.d]
    err = np.zeros((len(vs), len(Ds)))
    for i, v in enumerate(vs):
        for j, D in enumerate(Ds):
            err[i, j] = np.mean([score(z, t, run_model(z, v, D, cells=100, dt=2.0)) for z, t in runs])
    i, j = np.unravel_index(np.argmin(err), err.shape)
    v, D = vs[i], Ds[j]
    print(f"best fit: v = {v:.2e}, D = {D:.2e}, mean score {err[i, j]:.3f} (Peclet number vL/D = {v * L / D:.1f})")
    print(f"score with the hop-rate estimate v = {args.v:.2e}, D = {args.D:.2e}: "
          f"{np.mean([score(z, t, run_model(z, args.v, args.D, cells=100, dt=2.0)) for z, t in runs]):.3f}")
    np.savez(RESULTS / f"conveyor_model_fit{args.tag}.npz", v=vs, D=Ds, misfit=err, best=np.array([v, D]))
    for d, (z, t) in zip(args.d, runs):
        summary(z, t, run_model(z, v, D), f"d = {d:g}, fitted")
    figure(args, {d: (v, D) for d in args.d}, f"fig_conveyor_model_fit{args.tag}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("what", choices=["compare", "fit"])
    p.add_argument("--d", type=float, nargs="+", default=[8.0, 6.0, 4.0])
    p.add_argument("--tag", default="_ret")
    p.add_argument("--v", type=float, default=3.2e-4, help="drift (default: from the hop-rate estimate)")
    p.add_argument("--D", type=float, default=4.8e-4, help="spreading (default: from the hop-rate estimate)")
    p.add_argument("--v-range", type=float, nargs=2, default=[1.5e-4, 1.2e-3])
    p.add_argument("--D-range", type=float, nargs=2, default=[1e-4, 4e-3])
    p.add_argument("--grid", type=int, default=13)
    args = p.parse_args(argv)
    {"compare": compare, "fit": fit}[args.what](args)


if __name__ == "__main__":
    main()
