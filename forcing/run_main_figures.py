"""Main-text figures for the phase-locked loop, the amplitude control and the
sweep-rate limit, from the data written by run_main_runs.py.

    python forcing/run_main_figures.py --mode 5

Writes results/fig_loop_mode<n>.{png,pdf} and results/fig_rate_mode<n>.{png,pdf}.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.string_model import TensionString                  # noqa: E402
from forcing.run_paper_figures import style                     # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
DASH = (0, (4, 3))
DOT = (0, (1, 1.2))
DASHDOT = (0, (6, 2, 1.5, 2))


def figure_loop(mode):
    from forcing.run_paper_figures import pub_style
    from matplotlib.lines import Line2D
    plt = pub_style()
    d = np.load(RESULTS / f"main_loop_string_mode{mode}.npz")
    info = json.load(open(RESULTS / f"main_loop_string_mode{mode}.json"))
    model = TensionString(N=301, beta=info["beta"])
    w_res = model.resonance_table(mode)
    a0, a1 = info["a0"], info["a1"]
    t0, t1 = info["t_spin"], info["t_spin"] + abs(a1 - a0) / info["rate"]
    fig, axes = plt.subplots(2, 2, figsize=(17.0, 12.6), sharex=True, gridspec_kw=dict(
        hspace=0.28, wspace=0.3, left=0.085, right=0.985, top=0.885, bottom=0.075))
    axes = axes.ravel()
    grey = dict(color="0.72", linewidth=6.0)
    dash = dict(color="k", linestyle=DASH, linewidth=2.0, zorder=6)
    t = d["pll_t"]
    res = np.array([w_res(a) for a in d["pll_a"]])
    # (A) drive frequency of the loop and the resonance
    ax = axes[0]
    ax.plot(t, d["pll_w"], **grey)
    ax.plot(t, res, **dash)
    ax.set_ylabel(r"angular frequency $\omega$")
    ax.set_title("(A) Drive Frequency of the Loop", pad=62)
    ax.legend(handles=[Line2D([], [], label="Loop", **grey), Line2D([], [], label="Resonance", **dash)],
              loc="lower center", handlelength=2.0, ncol=2, columnspacing=1.2)
    lo, hi = res.min(), res.max()
    ax.set_ylim(lo - 0.35 * (hi - lo), hi + 0.12 * (hi - lo))
    a_ticks = np.linspace(a0, a1, 5)
    for a_top in axes[:2]:
        a_top.tick_params(top=False)
        top = a_top.secondary_xaxis("top")
        top.set_xticks(t0 + (a_ticks - a0) / (a1 - a0) * (t1 - t0))
        top.set_xticklabels([f"{v:+.2f}".replace("+0.00", "0") for v in a_ticks])
        top.set_xlabel("tension slope $a$", labelpad=6)
        top.tick_params(direction="in")
    # (B) detuning
    ax = axes[1]
    det = (d["pll_w"] - res) / info["beta"]
    ax.plot(t, det, color="k", linewidth=2.0)
    ax.axhline(0, color="0.75", linewidth=0.9, zorder=0)
    lim = 1.25 * np.abs(det)[t > t0].max()
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("detuning (linewidths)")
    ax.set_title("(B) Loop Frequency Minus the Resonance", pad=62)
    for x, lab, ha in ((t0, "sweep starts\n$t = %d$" % t0, "left"), (t1, "sweep ends\n$t = %d$" % t1, "right")):
        ax.text(x + (35 if ha == "left" else -35), -0.92 * lim, lab, ha=ha, va="bottom", fontsize=17, color="0.3")
    # (C) node error
    ax = axes[2]
    for kind, st in (("pll", grey), ("table", dash)):
        err = np.nanmax(np.abs(d[f"{kind}_nodes"] - d[f"{kind}_expected_nodes"]), axis=1)
        ax.plot(d[f"{kind}_t"], err, **st)
    ax.set_ylim(0, 0.024)
    ax.set_ylabel("node error")
    ax.set_title("(C) Largest Node Error")
    ax.legend(handles=[Line2D([], [], label="Loop", **grey), Line2D([], [], label="Lookup", **dash)],
              loc="upper center", handlelength=2.0, ncol=2, columnspacing=1.2)
    # (D) peak displacement
    ax = axes[3]
    i0 = int(np.searchsorted(t, t0)) - 1
    styles = (("pll", "Loop only", dict(color="0.4", linestyle=DASHDOT, linewidth=2.6)),
              ("pll+agc", "Loop + AL", dict(color="k", linestyle=DOT, linewidth=2.2)),
              ("pll+gain", "Loop + FF", grey),
              ("table+gain", "Lookup + FF", dash))
    for kind, lab, st in styles:
        pk = d[f"{kind}_peak"]
        ax.plot(d[f"{kind}_t"], pk / pk[min(i0, len(pk) - 1)], **st)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("peak displacement")
    for a_bot in axes[2:]:
        a_bot.set_xlabel("time $t$")
    ax.set_title("(D) Peak Displacement")
    ax.legend(handles=[Line2D([], [], label=lab, **st) for _, lab, st in styles], loc="lower left",
              ncol=2, handlelength=2.0, columnspacing=1.0, handletextpad=0.5)
    for a_ in axes:
        for tt in (t0, t1):
            a_.axvline(tt, color="0.8", linewidth=0.9, zorder=0)
        a_.set_xlim(t[0], t[-1])
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_loop_mode{mode}.{ext}")
    plt.close(fig)


def figure_rate(mode):
    from forcing.run_paper_figures import pub_style
    plt = pub_style()
    rows = np.genfromtxt(RESULTS / f"main_rates_string_mode{mode}.csv", delimiter=",", names=True,
                         dtype=None, encoding=None)
    info = json.load(open(RESULTS / f"main_loop_string_mode{mode}.json"))
    span = abs(info["a1"] - info["a0"])
    travel = float(np.max(info["node_travel"]))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15.0, 7.4), gridspec_kw=dict(
        wspace=0.32, left=0.085, right=0.985, top=0.92, bottom=0.14))
    # lookup: large grey squares drawn first; loop: smaller black circles on top, so both stay visible
    styles = {("table", 0.015): dict(marker="s", markersize=17, color="0.45", mfc="0.65", mec="0.3", mew=2.0, zorder=3,
                                     label=r"Lookup, $\beta = 0.015$"),
              ("table", 0.05): dict(marker="s", markersize=17, color="0.45", mfc="white", mec="0.3", mew=2.0, zorder=3,
                                    label=r"Lookup, $\beta = 0.05$"),
              ("pll", 0.015): dict(marker="o", markersize=10, color="k", mfc="k", mec="k", mew=2.0, zorder=5,
                                   label=r"Loop, $\beta = 0.015$"),
              ("pll", 0.05): dict(marker="o", markersize=11, color="k", mfc="white", mec="k", mew=2.2, zorder=5,
                                  label=r"Loop, $\beta = 0.05$")}
    slow = []
    for (drive, beta), st in styles.items():
        sel = (rows["drive"] == drive) & np.isclose(rows["beta"], beta)
        r = rows[sel]
        shift = 0.88 if beta < 0.03 else 1.13          # small sideways offset so the two dampings do not hide each other
        ax.loglog(r["rate"] * shift, r["max_node_error"], linestyle="none", **st)
        bx.loglog(r["rate"] * shift, np.maximum(1.0 - r["min_purity"], 1e-6), linestyle="none", **st)
        ok = r["rate"] <= 3e-3
        slow.extend(r["max_node_error"][ok] / (travel * r["rate"][ok] / span))
    delay = float(np.median(slow))
    rr = np.array([6e-5, 5e-2])
    ax.loglog(rr, delay * travel * rr / span, color="k", linestyle=DASH, linewidth=2.0, zorder=1,
              label=f"{delay:.0f}-unit delay")
    for a_ in (ax, bx):
        a_.set_xlabel(r"sweep rate $|da/dt|$")
        a_.set_xlim(6e-5, 5e-2)
        a_.minorticks_off()
        a_.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_ylabel("largest node error")
    ax.set_title("(A) Node Error")
    ax.legend(loc="lower right", fontsize=17, handletextpad=0.3, borderaxespad=0.3, labelspacing=0.45)
    bx.set_ylabel("1 $-$ lowest purity")
    bx.set_title("(B) Loss of Purity")
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_rate_mode{mode}.{ext}")
    plt.close(fig)
    print(f"node delay {delay:.2f} time units (largest node travel {travel:.2f} over a span of {span:.2f} in a)")


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, default=5)
    p.add_argument("--only", choices=["loop", "rate"], default=None)
    args = p.parse_args(argv)
    style()
    if args.only in (None, "loop"):
        figure_loop(args.mode)
    if args.only in (None, "rate"):
        figure_rate(args.mode)


if __name__ == "__main__":
    main()
