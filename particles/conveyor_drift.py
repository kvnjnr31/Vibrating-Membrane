"""Where does the model drift away from the adjustable conveyor?

    python particles/conveyor_drift.py --d 8 6 4 --tag _ret

Three tests, all with the fitted v and D:
  1. closed loop  - the model makes its own on/off decisions (as in conveyor_model.py)
  2. open loop    - the model is given the simulation's on/off record, so only the belt physics is tested
  3. noise        - the closed-loop model with a small random error added to the count at each decision,
                    repeated many times, to see how fast copies of the same model separate
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import style                                  # noqa: E402
from particles.experiment import RESULTS                                    # noqa: E402
from particles.conveyor_model import L, load_run                             # noqa: E402

V_FIT, D_FIT = 3.0e-4, 6.32e-4


def solve(z, v, D, u_sim=None, noise=0.0, rng=None, pulsed=False, cells=100, dt=2.0):
    """Model run.  u_sim: the simulation's command, used instead of the model's decisions.
    noise: standard deviation (grains) of an error added to the count at each decision.
    pulsed: drift and spreading act only during the sweep (scaled up so the cycle average is the same)."""
    period, t_sweep = float(z["period"]), float(z["times"][0])
    f_max, tau_f, target = float(z["feed_max"]), float(z["valve_tau"]), float(z["target"])
    box = z["box"]
    t_end = len(z["feed_rate"]) * float(z["dt"])
    dx = L / cells
    xc = (np.arange(cells) + 0.5) * dx
    overlap = np.clip(np.minimum(xc + 0.5 * dx, box[1]) - np.maximum(xc - 0.5 * dx, box[0]), 0.0, None)
    src = overlap / overlap.sum() / dx
    k = period / t_sweep if pulsed else 1.0
    dt = min(dt, 0.4 * dx * dx / (k * D), 0.5 * dx / (k * v))
    n = np.zeros(cells)
    F = G = R = 0.0
    u = 1.0
    dec_N, dec_u = [], []
    t_rec, N_rec = [], []
    B = Q = 0.0
    next_switch = t_sweep
    sim_dt = float(z["dt"])
    for i in range(int(round(t_end / dt))):
        t = i * dt
        if t >= next_switch:
            N = n.sum() * dx
            dec_N.append(N)
            if u_sim is not None:
                u = float(u_sim[min(int(t / sim_dt) + 1, len(u_sim) - 1)])
            else:
                Nm = N + (rng.normal(0.0, noise) if noise > 0 else 0.0)
                u = 1.0 if Nm < target else 0.0
            dec_u.append(u)
            next_switch += period
        G += dt * (f_max * u - F - 2 * tau_f * G) / tau_f ** 2
        F = min(max(F + dt * G, 0.0), f_max)
        on = (not pulsed) or (t % period) < t_sweep
        if on:
            ng = np.concatenate(([-n[0]], n, [-n[-1]]))
            upw = ng[:-1].copy(); upw[0] = 0.0; upw[-1] = n[-1]
            J = k * v * upw - k * D * (ng[1:] - ng[:-1]) / dx
            b, q = -J[0], J[-1]
            n = n - dt * (J[1:] - J[:-1]) / dx
        else:
            b = q = 0.0
        n = np.maximum(n + dt * F * src, 0.0)
        R = max(R + dt * (b - (F if R > 1e-9 else 0.0)), 0.0)
        B += b * dt; Q += q * dt
        if i % max(1, int(round(10 / dt))) == 0:
            t_rec.append(t); N_rec.append(n.sum() * dx)
    return dict(t=np.array(t_rec), N=np.array(N_rec), dec_N=np.array(dec_N), dec_u=np.array(dec_u), B=B, Q=Q)


def rms(z, t, m):
    sim = z["count"][::500].astype(float)
    return np.sqrt(np.mean((sim - np.interp(t[::500], m["t"], m["N"])) ** 2)) / float(z["target"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=float, nargs="+", default=[8.0, 6.0, 4.0])
    p.add_argument("--tag", default="_ret")
    p.add_argument("--copies", type=int, default=40)
    p.add_argument("--noise", type=float, default=None, help="decision noise (grains); default: measured from the open-loop residual")
    args = p.parse_args()
    out = {}
    for d in args.d:
        z, t = load_run(d, args.tag)
        target = float(z["target"])
        meas = z["measured"]                                   # (time, count, command) at each decision
        u_cmd = z["command"].astype(float)
        closed = solve(z, V_FIT, D_FIT)
        opened = solve(z, V_FIT, D_FIT, u_sim=u_cmd)
        pulsed = solve(z, V_FIT, D_FIT, u_sim=u_cmd, pulsed=True)
        n_dec = min(len(meas), len(closed["dec_u"]))
        sim_u = meas[:n_dec, 2]
        first = next((i for i in range(n_dec) if sim_u[i] != closed["dec_u"][i]), None)
        # residual of the open-loop model at the decision times, after the first fill
        res = meas[8:n_dec, 1] - opened["dec_N"][8:n_dec]
        margin = np.abs(meas[:n_dec, 1] - target)
        print(f"\nd = {d:g}")
        print(f"  closed-loop misfit {rms(z, t, closed):.3f},  open-loop misfit {rms(z, t, opened):.3f},"
              f"  open loop with transport only in the sweep {rms(z, t, pulsed):.3f}")
        print(f"  open-loop count error at decisions (after cycle 8): mean {res.mean():+.0f}, sd {res.std():.0f} grains")
        print(f"  first decision where model and simulation differ: cycle {None if first is None else first + 1}")
        print(f"  decisions with the count within 100 grains of the target: {(margin < 100).sum()} of {n_dec}")
        out[d] = dict(z=z, t=t, closed=closed, opened=opened, res=res, first=first)
    noise = args.noise if args.noise is not None else float(np.mean([out[d]["res"].std() for d in args.d]))
    print(f"\nnoise test: decision error sd {noise:.0f} grains, {args.copies} copies of the model per d")
    rng = np.random.default_rng(1)
    for d in args.d:
        z, t = out[d]["z"], out[d]["t"]
        copies = [solve(z, V_FIT, D_FIT, noise=noise, rng=rng) for _ in range(args.copies)]
        base = out[d]["closed"]
        Ns = np.array([np.interp(base["t"], c["t"], c["N"]) for c in copies])
        out[d]["copies"] = Ns
        spread = [np.sqrt(np.mean((Ns[i] - base["N"]) ** 2)) / float(z["target"]) for i in range(len(Ns))]
        diverge = []
        for c in copies:
            k = next((i for i in range(min(len(c["dec_u"]), len(base["dec_u"]))) if c["dec_u"][i] != base["dec_u"][i]), None)
            diverge.append(np.nan if k is None else k + 1)
        diverge = np.array(diverge)
        print(f"  d = {d:g}: copies vs noise-free model misfit {np.mean(spread):.3f} (range {np.min(spread):.3f}-{np.max(spread):.3f}); "
              f"first different decision median cycle {np.nanmedian(diverge):.0f}, never differ {np.isnan(diverge).sum()}/{len(diverge)}")
    np.savez(RESULTS / f"conveyor_drift{args.tag}.npz",
             **{f"d{d:g}_{k}": out[d][k]["N"] for d in args.d for k in ("closed", "opened")},
             **{f"d{d:g}_t": out[d]["closed"]["t"] for d in args.d},
             **{f"d{d:g}_copies": out[d]["copies"] for d in args.d}, noise=noise)


if __name__ == "__main__":
    main()
