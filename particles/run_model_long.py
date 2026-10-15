"""Long-term behaviour of the reduced conveyor model.

    python particles/run_model_long.py --d 8 --cycles 400

Runs the drift-diffusion model of the self-regulating conveyor (fitted v, D) for many
cycles, without noise and with a random error added to each count, and writes
results/fig_model_long_d<d>.{png,pdf} and results/model_long_d<d>.npz.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import style                                  # noqa: E402
from particles.experiment import RESULTS                                    # noqa: E402

L = 10.0
V_FIT, D_FIT = 3.0e-4, 6.32e-4
SIM = dict(period=1640.0, t_sweep=1310.0, f_max=0.5, tau_f=420.0, target=3000.0, box_width=0.8)


def run(d, cycles, v=V_FIT, D=D_FIT, noise=0.0, seed=0, cells=400, dt=1.0, record=10.0):
    """Model run; returns time series (every `record`) and the count and setting at each decision.
    The long-term pattern needs a fine grid: 100 cells give a 20-cycle pattern, 200 or more give the converged 9-cycle one."""
    p = SIM
    T, ts, fmax, tau, target = p["period"], p["t_sweep"], p["f_max"], p["tau_f"], p["target"]
    rng = np.random.default_rng(seed)
    dx = L / cells
    xc = (np.arange(cells) + 0.5) * dx
    xf = L - d
    box = (xf - 0.5 * p["box_width"], xf + 0.5 * p["box_width"])
    ov = np.clip(np.minimum(xc + 0.5 * dx, box[1]) - np.maximum(xc - 0.5 * dx, box[0]), 0.0, None)
    src = ov / ov.sum() / dx
    dt = min(dt, 0.4 * dx * dx / D, 0.5 * dx / v)
    rho = np.zeros(cells)
    f = V = M = 0.0
    sigma = 1.0
    t_rec, N_rec, f_rec, M_rec = [], [], [], []
    dec_N, dec_s = [], []
    q_off = q_ret = 0.0
    off_cyc, ret_cyc = [], []
    k_rec = max(1, int(round(record / dt)))
    next_switch = ts
    next_cycle = T
    steps = int(round(cycles * T / dt))
    for i in range(steps):
        t = i * dt
        if t >= next_switch:
            N = rho.sum() * dx
            dec_N.append(N)
            sigma = 1.0 if N + (rng.normal(0.0, noise) if noise > 0 else 0.0) < target else 0.0
            dec_s.append(sigma)
            next_switch += T
        if t >= next_cycle:
            off_cyc.append(q_off); ret_cyc.append(q_ret)
            next_cycle += T
        V += dt * (fmax * sigma - f - 2 * tau * V) / tau ** 2
        f = min(max(f + dt * V, 0.0), fmax)
        g = np.concatenate(([-rho[0]], rho, [-rho[-1]]))
        up = g[:-1].copy(); up[0] = 0.0; up[-1] = rho[-1]
        J = v * up - D * (g[1:] - g[:-1]) / dx
        b, q = -J[0], J[-1]
        rho = np.maximum(rho + dt * (-(J[1:] - J[:-1]) / dx + f * src), 0.0)
        M = max(M + dt * (b - (f if M > 1e-9 else 0.0)), 0.0)
        q_off += q * dt; q_ret += b * dt
        if i % k_rec == 0:
            t_rec.append(t); N_rec.append(rho.sum() * dx); f_rec.append(f); M_rec.append(M)
    return dict(t=np.array(t_rec), N=np.array(N_rec), f=np.array(f_rec), M=np.array(M_rec),
                dec_N=np.array(dec_N), dec_s=np.array(dec_s), off=np.diff(off_cyc, prepend=0.0),
                ret=np.diff(ret_cyc, prepend=0.0), T=T, target=target)


def period_of(seq, tail=120, max_p=60):
    """Smallest p for which the last `tail` decisions repeat with period p (None if none)."""
    s = np.asarray(seq[-tail:])
    for p in range(1, max_p + 1):
        if np.all(s[p:] == s[:-p]):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=float, default=8.0)
    ap.add_argument("--cycles", type=int, default=400)
    ap.add_argument("--noise", type=float, default=33.0, help="sd of the count error in the noisy run (grains)")
    ap.add_argument("--zoom", type=int, default=30, help="cycles shown in the close-up")
    args = ap.parse_args()
    det = run(args.d, args.cycles)
    noisy = run(args.d, args.cycles, noise=args.noise, seed=1)
    T, target = det["T"], det["target"]
    for lab, r in (("no noise", det), (f"noise {args.noise:g} grains", noisy)):
        s = r["dec_s"]
        p = period_of(s)
        half = len(r["off"]) // 2
        pat = "".join("1" if x > 0.5 else "0" for x in s[-p:]) if p else "-"
        print(f"{lab}: decisions repeat with period {p} cycles (pattern {pat}); on share {s[len(s)//2:].mean():.3f}; "
              f"count at decisions, second half: mean {r['dec_N'][len(s)//2:].mean():.0f}, "
              f"range {r['dec_N'][len(s)//2:].min():.0f}-{r['dec_N'][len(s)//2:].max():.0f}; "
              f"offloaded per cycle {r['off'][half:].mean():.1f}, returned per cycle {r['ret'][half:].mean():.1f}; "
              f"closest count to the target, second half {np.min(np.abs(r['dec_N'][len(s)//2:] - target)):.1f}")
    np.savez(RESULTS / f"model_long_d{args.d:g}.npz", **{f"det_{k}": v for k, v in det.items() if isinstance(v, np.ndarray)},
             **{f"noisy_{k}": v for k, v in noisy.items() if isinstance(v, np.ndarray)})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    fig = plt.figure(figsize=(14, 13.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5], left=0.07, right=0.915, top=0.96, bottom=0.06, hspace=0.42, wspace=0.42)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(noisy["t"] / T, noisy["N"], color="0.65", linewidth=0.8, label=f"With count noise ({args.noise:g} grains)")
    ax.plot(det["t"] / T, det["N"], color="k", linewidth=0.9, label="No noise")
    ax.axhline(target, color="0.4", linewidth=1.0, linestyle=":")
    ax.set_xlim(0, args.cycles); ax.set_ylim(0, 4200)
    ax.set_xlabel("time (cycles)"); ax.set_ylabel("grains on the conveyor")
    ax.set_title(f"(A) Grains on the Conveyor over {args.cycles} Cycles, d = {args.d:g}", fontsize=15, loc="left")
    ax.legend(loc="lower right", fontsize=12, ncol=2)
    t0 = args.cycles - args.zoom
    for col, (r, lab, letter) in enumerate(((det, "No Noise", "B"), (noisy, "With Count Noise", "C"))):
        a = fig.add_subplot(gs[1, col])
        m = r["t"] / T >= t0
        a.plot(r["t"][m] / T, r["f"][m], color="k", linewidth=1.4, label="feed rate")
        a.set_ylim(0, 0.58); a.set_ylabel("feed rate"); a.set_xlim(t0, args.cycles)
        a2 = a.twinx()
        a2.plot(r["t"][m] / T, r["N"][m], color="0.55", linewidth=1.4, linestyle=(0, (4, 3)))
        a2.axhline(target, color="0.75", linewidth=1.0, linestyle=":")
        a2.set_ylim(2000, 4200); a2.set_ylabel("grains", color="0.4"); a2.spines["right"].set_visible(True)
        a.set_xlabel("time (cycles)")
        a.set_title(f"({letter}) Last {args.zoom} Cycles, {lab}", fontsize=15, loc="left")
    for col, (r, lab, letter) in enumerate(((det, "No Noise", "D"), (noisy, "With Count Noise", "E"))):
        a = fig.add_subplot(gs[2, col])
        n = r["dec_N"]
        k0 = 10                                             # leave out the first fill
        a.plot(n[k0:-1], n[k0 + 1:], color="0.75", linewidth=0.6, zorder=1)
        on = r["dec_s"][k0:-1] > 0.5
        a.scatter(n[k0:-1][on], n[k0 + 1:][on], s=22, color="k", zorder=2, label="feeder set on")
        a.scatter(n[k0:-1][~on], n[k0 + 1:][~on], s=26, facecolors="white", edgecolors="k", linewidths=1.1, zorder=2,
                  label="feeder set off")
        a.axvline(target, color="0.5", linewidth=1.0, linestyle=":")
        a.axhline(target, color="0.5", linewidth=1.0, linestyle=":")
        lo, hi = 2300, 3700
        a.plot([lo, hi], [lo, hi], color="0.85", linewidth=1.0)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal", adjustable="box")
        a.set_xlabel("count at sweep end $k$"); a.set_ylabel("count at sweep end $k+1$")
        a.set_title(f"({letter}) Count Map, {lab}", fontsize=15, loc="left")
        if letter == "D":
            a.legend(loc="upper left", fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_model_long_d{args.d:g}.{ext}", dpi=200)
    print("saved", RESULTS / f"fig_model_long_d{args.d:g}.png")


if __name__ == "__main__":
    main()
