"""Simulation against itself (two random seeds) and against the model; drift speed along the belt.

    python particles/conveyor_seed_figure.py
"""
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import style  # noqa: E402
from particles.experiment import RESULTS  # noqa: E402
from particles.conveyor_model import load_run  # noqa: E402
from particles.conveyor_drift import solve, V_FIT, D_FIT  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def rms(a, b, target):
    return np.sqrt(np.mean((a - b) ** 2)) / target


style()
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(left=0.06, right=0.99, top=0.88, bottom=0.16, wspace=0.28))
for ax, d, lab in zip(axes[:2], (8, 6), "AB"):
    z0, t0 = load_run(d, "_ret"); z1, t1 = load_run(d, "_ret_s1")
    T, target = float(z0["period"]), float(z0["target"])
    ts = np.arange(0, min(t0[-1], t1[-1]), 10.0)
    n0 = np.interp(ts, t0[::500], z0["count"][::500]); n1 = np.interp(ts, t1[::500], z1["count"][::500])
    m0 = solve(z0, V_FIT, D_FIT); m1 = solve(z1, V_FIT, D_FIT)
    nm = np.interp(ts, m0["t"], m0["N"])
    print(f"d = {d}: seed 0 vs seed 1 misfit {rms(n0, n1, target):.3f}; model vs seed 0 {rms(nm, n0, target):.3f}, "
          f"model vs seed 1 {rms(nm, n1, target):.3f}")
    for z, s in ((z0, 0), (z1, 1)):
        print(f"   seed {s}: offloaded {len(z['exits'])}, returned {len(z['returns'])}, fed {len(z['fed'])}, "
              f"switched on {int((np.diff(z['command'].astype(int)) > 0).sum())} times, second-half feed {z['feed_rate'][len(z['feed_rate'])//2:].mean():.3f}")
    print(f"   model: offloaded {m0['Q']:.0f}, returned {m0['B']:.0f}")
    ax.plot(ts / T, n0, color="k", lw=1.5, label="Simulation, seed 0")
    ax.plot(ts / T, n1, color="0.55", lw=1.5, label="Simulation, seed 1")
    ax.plot(ts / T, nm, color="k", lw=1.4, ls=(0, (4, 3)), label="Model")
    ax.axhline(target, color="0.75", lw=1.0, ls=":")
    ax.set_xlim(0, 40); ax.set_ylim(0, 4600)
    ax.set_xlabel("time (cycles)"); ax.set_ylabel("grains on the conveyor")
    ax.set_title(f"{lab}: Grains on the Conveyor, d = {d}", fontsize=15)
    if lab == "A":
        ax.legend(loc="lower right", fontsize=11)
rows = np.loadtxt(RESULTS / "conveyor_transport_x_ret.csv", delimiter=",", skiprows=1)
ax = axes[2]
for d, c, ls, mk in ((8, "k", "-", "o"), (6, "0.4", "--", "s"), (4, "0.65", ":", "^")):
    r = rows[(rows[:, 0] == d) & (rows[:, 1] > 2) & (rows[:, 1] < 8)]
    ax.plot(r[:, 1], r[:, 3] * 1e4, color=c, ls=ls, marker=mk, lw=1.5, label=f"d = {d}")
ax.axhline(V_FIT * 1e4, color="0.75", lw=1.2, ls="-.", label="model v")
ax.axhline(0, color="0.85", lw=0.8)
ax.set_xlim(2, 8); ax.set_xlabel("position on the belt, x"); ax.set_ylabel(r"drift speed ($10^{-4}$)")
ax.set_title("C: Drift Speed Along the Belt", fontsize=15)
ax.legend(loc="upper right", fontsize=11)
for ext in ("png", "pdf"):
    fig.savefig(RESULTS / f"fig_conveyor_drift.{ext}")
print("saved")
