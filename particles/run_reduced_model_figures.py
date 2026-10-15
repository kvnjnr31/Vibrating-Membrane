"""Figures for the section on the reduced model of the adjustable conveyor.

    python particles/run_reduced_model_figures.py

Needs the return-slot runs (adjustable_conveyor_ret_d{2,4,6,8}, adjustable_conveyor_ret_s1_d{8,6}).
Writes
    fig_reduced_model_setup   A: the belt, feeder, slots and hopper   B: share of grains offloaded against d
    fig_reduced_model_runs    feed rate and grain count, model against simulation, d = 4, 6, 8
    fig_reduced_model_noise   A, B: two random seeds and the model   C: drift speed along the belt
    reduced_model_summary.csv the numbers quoted in the text
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import style, pub_style                                  # noqa: E402
from particles.experiment import RESULTS                                    # noqa: E402
from particles.conveyor_model import L, load_run, run_model, misfit         # noqa: E402

import matplotlib                                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402
from matplotlib.patches import Rectangle, FancyArrowPatch                   # noqa: E402

V_FIT, D_FIT = 3.0e-4, 6.32e-4          # fitted to the three runs (conveyor_model.py fit)
V_MID, D_MID = 3.29e-4, 6.54e-4         # measured from grain displacements in the middle of the belt, 3 < x < 7
DS = (4.0, 6.0, 8.0)                    # runs figure, top to bottom
DS_SHARE = (2.0, 4.0, 6.0, 8.0)         # offloaded-share panel
TAG, TAG_S1 = "_ret", "_ret_s1"


def p_off(xf, v, D):
    pe = v * L / D
    return (1 - np.exp(-pe * xf / L)) / (1 - np.exp(-pe))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"{name}.{ext}", dpi=200)
    print("saved", RESULTS / f"{name}.png")


def arrow(ax, a, b, **kw):
    kw.setdefault("arrowstyle", "-|>")
    kw.setdefault("mutation_scale", 16)
    kw.setdefault("color", "k")
    kw.setdefault("linewidth", 1.4)
    ax.add_patch(FancyArrowPatch(a, b, **kw))


def figure_setup():
    pub_style()
    fig = plt.figure(figsize=(21, 8.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], left=0.005, right=0.985, top=0.92, bottom=0.12, wspace=0.13)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    ax = axes[0]
    ax.set_xlim(-1.6, 11.6); ax.set_ylim(-3.0, 3.6); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), L, 0.5, facecolor="0.92", edgecolor="k", linewidth=1.6))
    for xn in (2, 4, 6, 8):
        ax.plot([xn, xn], [0, 0.5], color="0.45", linewidth=1.4, linestyle=":")
        ax.text(xn, -0.3, "node", ha="center", va="top", fontsize=15, color="0.35")
    ax.plot([0, 0], [-0.2, 1.3], color="k", linewidth=4.0)
    ax.text(-0.25, 1.0, "wall", ha="right", va="center", fontsize=17)
    ax.add_patch(Rectangle((0, 0), 0.3, 0.5, facecolor="0.6", edgecolor="k", hatch="///", linewidth=1.2))
    ax.add_patch(Rectangle((L, 0), 0.25, 0.5, facecolor="0.6", edgecolor="k", hatch="///", linewidth=1.2))
    xf = 5.0                                 # between two nodes: the feeder can sit anywhere
    ax.add_patch(Rectangle((xf - 0.85, 1.4), 1.7, 0.7, facecolor="white", edgecolor="k", linewidth=1.6))
    ax.text(xf, 1.75, "feeder", ha="center", va="center", fontsize=17)
    arrow(ax, (xf, 1.38), (xf, 0.56), linewidth=1.8, mutation_scale=20)
    ax.text(xf + 0.18, 0.97, r"$f(t)$", ha="left", va="center", fontsize=19)
    arrow(ax, (xf + 1.0, 0.25), (xf + 2.6, 0.25), linewidth=1.6, mutation_scale=20)
    ax.text(xf + 1.8, 0.65, r"drift $v$", ha="center", va="bottom", fontsize=17)
    arrow(ax, (2.1, 0.25), (3.7, 0.25), arrowstyle="<|-|>", linewidth=1.6, mutation_scale=20)
    ax.text(2.9, 0.65, r"spread $D$", ha="center", va="bottom", fontsize=17)
    arrow(ax, (L + 0.15, 0.25), (L + 1.1, 0.25), linewidth=1.8, mutation_scale=20)
    ax.text(L + 0.6, 0.65, r"$Q_{\rm off}$", ha="center", va="bottom", fontsize=19)
    ax.text(L + 0.15, -0.3, "offload\nslot", ha="center", va="top", fontsize=16)
    ax.text(0.15, -0.3, "return\nslot", ha="center", va="top", fontsize=16)
    ax.add_patch(Rectangle((1.5, 2.55), 2.3, 0.75, facecolor="white", edgecolor="k", linewidth=1.6))
    ax.text(2.65, 2.92, r"hopper $M$", ha="center", va="center", fontsize=17)
    arrow(ax, (0.15, 0.55), (1.45, 2.8), connectionstyle="arc3,rad=-0.35", linewidth=1.8, mutation_scale=20)
    ax.text(-0.15, 2.15, r"$Q_{\rm ret}$", ha="center", va="center", fontsize=19)
    arrow(ax, (3.85, 2.9), (xf - 0.2, 2.15), connectionstyle="arc3,rad=-0.2", linewidth=1.8, mutation_scale=20)
    ax.plot([xf, xf], [-1.75, -1.2], color="k", linewidth=1.3)
    ax.plot([L, L], [-1.75, -1.2], color="k", linewidth=1.3)
    arrow(ax, (xf, -1.47), (L, -1.47), arrowstyle="<|-|>", linewidth=1.3, mutation_scale=16)
    ax.text(0.5 * (xf + L), -1.62, r"$d$", ha="center", va="top", fontsize=19)
    ax.plot([0, 0], [-2.55, -2.05], color="k", linewidth=1.3)
    arrow(ax, (0, -2.3), (xf, -2.3), arrowstyle="<|-|>", linewidth=1.3, mutation_scale=16)
    ax.text(0.5 * xf, -2.45, r"$x_f = L - d$", ha="center", va="top", fontsize=18)
    ax.set_title("(A) Reduced Model of the Conveyor")

    ax = axes[1]
    d = np.linspace(0.05, L - 0.05, 300)
    ax.plot(d, p_off(L - d, V_FIT, D_FIT), color="k", linewidth=2.6, linestyle=(0, (6, 3)), label="Model")
    rows = []
    for tag, plot in ((TAG, True), (TAG_S1, False)):         # seed 1 goes to the csv only
        pts = []
        for dd in DS_SHARE:
            f = RESULTS / f"adjustable_conveyor{tag}_d{dd:g}_series.npz"
            if not f.exists():
                continue
            z = np.load(f)
            o, r = len(z["exits"]), len(z["returns"])
            pts.append((dd, o / (o + r)))
            rows.append((tag, dd, o, r, o / (o + r), p_off(L - dd, V_FIT, D_FIT), p_off(L - dd, V_MID, D_MID)))
        pts = np.array(pts)
        if plot:
            ax.plot(pts[:, 0], pts[:, 1], linestyle="none", marker="o", markersize=15, color="k", label="Simulation")
    ax.set_xlim(0, L); ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"feeder distance from the offload, $d$"); ax.set_ylabel("share offloaded")
    ax.set_title("(B) Offloaded Share")
    ax.legend(loc="lower left", handlelength=2.4)
    save(fig, "Fig9")
    style()
    return rows


def figure_runs():
    pub_style()
    nd = len(DS)
    fig, axes = plt.subplots(nd, 2, figsize=(17, 18), sharex=True,
                             gridspec_kw=dict(left=0.09, right=0.975, top=0.965, bottom=0.06, hspace=0.24, wspace=0.36))
    rows = []
    for r, d in enumerate(DS):
        z, t = load_run(d, TAG)
        T = float(z["period"])
        m = run_model(z, V_FIT, D_FIT)
        a0, a1 = axes[r]
        a0.plot(t[::50] / T, z["feed_rate"][::50], color="k", linewidth=2.2, label="Simulation")
        a0.plot(m["t"] / T, m["F"], color="0.5", linewidth=2.4, linestyle=(0, (5, 3)), label="Model")
        a0.set_ylim(0, 1.15 * float(z["feed_max"])); a0.set_ylabel("feed rate")
        a0.set_title(f"({'ABCDEF'[2 * r]}) Feed Rate, $d$ = {d:g}")
        a1.plot(t[::500] / T, z["count"][::500], color="k", linewidth=2.2, label="Simulation")
        a1.plot(m["t"] / T, m["N"], color="0.5", linewidth=2.4, linestyle=(0, (5, 3)), label="Model")
        a1.axhline(float(z["target"]), color="0.6", linewidth=1.2, linestyle=":")
        a1.set_ylim(0, 4600); a1.set_ylabel("grains on conveyor", labelpad=10)
        a1.set_yticks([0, 1000, 2000, 3000, 4000])
        a1.set_title(f"({'ABCDEF'[2 * r + 1]}) Grains on the Conveyor, $d$ = {d:g}")
        for a in (a0, a1):
            a.set_xlim(0, round(t[-1] / T))
            a.set_xticks([0, 10, 20, 30, 40])
        if r == 0:
            a1.legend(loc="lower right", handlelength=2.4)
        half = t > t[-1] / 2
        mh = m["t"] > t[-1] / 2
        rows.append(dict(d=d, feed_sim=z["feed_rate"][half].mean(), feed_mod=m["F"][mh].mean(),
                         N_sim=z["count"][half].mean(), N_mod=m["N"][mh].mean(),
                         off_sim=len(z["exits"]), off_mod=m["Q"][-1], ret_sim=len(z["returns"]), ret_mod=m["B"][-1],
                         on_sim=int((np.diff(z["command"].astype(int)) > 0).sum()), on_mod=int((np.diff(m["u"]) > 0.5).sum()),
                         misfit=misfit(z, t, m)))
    for a in axes[-1]:
        a.set_xlabel("time (cycles)")
    save(fig, "Fig10")
    style()
    return rows


def figure_noise():
    def rms(a, b, target):
        return np.sqrt(np.mean((a - b) ** 2)) / target

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(left=0.06, right=0.99, top=0.88, bottom=0.16, wspace=0.28))
    rows = []
    for ax, d, lab in zip(axes[:2], (6.0, 8.0), "AB"):
        z0, t0 = load_run(d, TAG)
        z1, t1 = load_run(d, TAG_S1)
        T, target = float(z0["period"]), float(z0["target"])
        ts = np.arange(0, min(t0[-1], t1[-1]), 10.0)
        n0 = np.interp(ts, t0[::500], z0["count"][::500])
        n1 = np.interp(ts, t1[::500], z1["count"][::500])
        nm = np.interp(ts, *(lambda m: (m["t"], m["N"]))(run_model(z0, V_FIT, D_FIT)))
        rows.append(dict(d=d, seed_vs_seed=rms(n0, n1, target), model_vs_seed0=rms(nm, n0, target), model_vs_seed1=rms(nm, n1, target),
                         off_seed0=len(z0["exits"]), off_seed1=len(z1["exits"]), ret_seed0=len(z0["returns"]), ret_seed1=len(z1["returns"])))
        ax.plot(ts / T, n0, color="k", lw=1.5, label="Simulation, seed 0")
        ax.plot(ts / T, n1, color="0.55", lw=1.5, label="Simulation, seed 1")
        ax.plot(ts / T, nm, color="k", lw=1.4, ls=(0, (4, 3)), label="Model")
        ax.axhline(target, color="0.75", lw=1.0, ls=":")
        ax.set_xlim(0, 40); ax.set_ylim(0, 4600)
        ax.set_xlabel("time (cycles)"); ax.set_ylabel("grains on the conveyor")
        ax.set_title(f"({lab}) Grains on the Conveyor, d = {d:g}", fontsize=15)
        if lab == "A":
            ax.legend(loc="lower right", fontsize=11)
    prof = np.loadtxt(RESULTS / f"conveyor_transport_x{TAG}.csv", delimiter=",", skiprows=1)
    ax = axes[2]
    for d, c, ls, mk in ((8, "k", "-", "o"), (6, "0.4", "--", "s"), (4, "0.65", ":", "^")):
        r = prof[(prof[:, 0] == d) & (prof[:, 1] > 2) & (prof[:, 1] < 8)]
        ax.plot(r[:, 1], r[:, 3] * 1e4, color=c, ls=ls, marker=mk, lw=1.5, label=f"d = {d}")
    ax.axhline(V_FIT * 1e4, color="0.6", lw=1.2, ls="-.", label="model $v$")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_xlim(2, 8); ax.set_xlabel("position at the end of the sweep, $x$"); ax.set_ylabel(r"drift speed ($10^{-4}$)")
    ax.set_title("(C) Drift Speed Along the Belt", fontsize=15)
    ax.legend(loc="upper right", fontsize=11)
    save(fig, "fig_reduced_model_noise")
    return rows


def main():
    style()
    if "--runs-only" in sys.argv:
        figure_runs()
        return
    share = figure_setup()
    if "--setup-only" in sys.argv:
        return
    runs = figure_runs()
    noise = figure_noise()
    with open(RESULTS / "reduced_model_summary.csv", "w") as f:
        f.write("offloaded share\nrun,d,offloaded,returned,share,model_fitted,model_measured\n")
        for r in share:
            f.write(",".join(str(round(x, 4)) if isinstance(x, float) else str(x) for x in r) + "\n")
        f.write("\nruns (feed and count over the second half)\n" + ",".join(runs[0]) + "\n")
        for r in runs:
            f.write(",".join(f"{x:.4g}" for x in r.values()) + "\n")
        f.write("\nseeds\n" + ",".join(noise[0]) + "\n")
        for r in noise:
            f.write(",".join(f"{x:.4g}" for x in r.values()) + "\n")
    print(open(RESULTS / "reduced_model_summary.csv").read())


if __name__ == "__main__":
    main()
