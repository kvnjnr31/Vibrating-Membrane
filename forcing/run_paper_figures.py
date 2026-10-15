"""Paper figures for the drive comparison, in the paper's plain style.

For one mode of the string, with the tension slope swept across its whole
range so the nodes travel as far as they can:

    fig_drives_mode<n>     A: node positions, B: mode purity, C: leading
                           frequency of the response, for the fixed drive,
                           direct substitution and the table lookup
    fig_snapshots_mode<n>  the table-lookup wave against the predicted mode
                           at the start, middle and end of the sweep, with
                           the sine-series spectrum of each snapshot

Each is written to results/ as .png and .pdf.

    python forcing/run_paper_figures.py              # mode 3
    python forcing/run_paper_figures.py --mode 7
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing import schedules                                            # noqa: E402
from forcing.simulation import Simulation, make_drive                    # noqa: E402
from forcing.string_model import TensionString                           # noqa: E402
from forcing.sweep_experiment import RESULTS                             # noqa: E402

COLOR = {"fixed": "0.4", "naive": "0.0", "table": "0.72"}
LSTYLE = {"fixed": (0, (7, 2.5, 1.8, 2.5)), "naive": (0, (1, 1.3)), "table": "-"}
LWIDTH = {"fixed": 2.6, "naive": 2.2, "table": 6.0}
LABEL = {"fixed": "Fixed frequency", "naive": "Direct substitution", "table": "Table lookup"}
SHORT = {"fixed": "Fixed freq.", "naive": "Direct sub.", "table": "Lookup"}
BLACK = "black"


def style():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 18, "legend.fontsize": 15,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "axes.edgecolor": BLACK, "axes.labelcolor": BLACK, "text.color": BLACK,
        "xtick.color": BLACK, "ytick.color": BLACK, "axes.linewidth": 1.0,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "lines.linewidth": 2.0, "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 200,
    })
    import matplotlib.pyplot as plt
    return plt


def pub_style():
    """Style of the author's published figures: Arial-type font, full box around each panel,
    inward ticks on all four sides, heavier axis lines, bold axis labels, plain centred panel titles."""
    plt = style()
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": 1.4,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 6, "ytick.major.size": 6, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlesize": 22,
        "axes.labelweight": "bold", "axes.labelsize": 22, "legend.frameon": False,
        "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 18, "font.size": 20,
    })
    return plt


def leading_frequency(t, S, window, hop, band):
    """Strongest peak of the temporal Fourier spectrum in sliding Hann windows.

    S has one column per pickup point; the power spectra of all points are
    added, so the result does not depend on where a single pickup sits
    relative to the nodes.  Returns window-centre times and angular
    frequencies; the peak is refined by zero padding and a parabola through
    the three highest bins."""
    dt = t[1] - t[0]
    nw = int(round(window / dt))
    nh = max(1, int(round(hop / dt)))
    nfft = 16 * 2 ** int(np.ceil(np.log2(nw)))
    w = 2.0 * np.pi * np.fft.rfftfreq(nfft, dt)
    keep = (w >= band[0]) & (w <= band[1])
    han = np.hanning(nw)[:, None]
    tc, wc = [], []
    for i in range(0, len(S) - nw + 1, nh):
        seg = S[i:i + nw]
        P = np.sum(np.abs(np.fft.rfft((seg - seg.mean(axis=0)) * han, nfft, axis=0)) ** 2, axis=1)
        P[~keep] = 0.0
        k = int(np.argmax(P))
        if 0 < k < len(P) - 1 and P[k] > 0:
            y0, y1, y2 = np.log(P[k - 1] + 1e-300), np.log(P[k]), np.log(P[k + 1] + 1e-300)
            k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        tc.append(t[i] + 0.5 * window)
        wc.append(k * (w[1] - w[0]))
    return np.array(tc), np.array(wc)


def sine_spectrum(model, u, kmax):
    """Amplitudes of the sine series of u(x), harmonics 1..kmax."""
    k = np.arange(1, kmax + 1)
    S = np.sin(np.outer(k, np.pi * model.x / model.L))
    return k, np.abs(2.0 / model.L * (S @ u) * model.dx)


def run(mode, beta, a0, a1, t_start, t_stop, t_hold, points):
    model = TensionString(N=points, beta=beta)
    rate = abs(a1 - a0) / (t_stop - t_start)
    sched = schedules.linear_ramp(a0, a1, t_start, rate)
    t_end = t_stop + t_hold
    w_res = model.resonance_table(mode)
    period = 2.0 * np.pi / w_res(a0)
    pick = np.arange(3, model.N - 3, 6)                 # pickup points along the string
    targets = [t_start, 0.5 * (t_start + t_stop), t_stop]
    out = dict(model=model, sched=sched, w_res=w_res, mode=mode, targets=targets, period=period,
               t_start=t_start, t_stop=t_stop, t_end=t_end, sims={}, probe={}, snaps={})
    for kind in ("fixed", "naive", "table"):
        sim = Simulation(model, make_drive(kind, model, mode, sched), sched, mode).start_in_steady_state()
        ts, ps, cand = [], [], {tt: [] for tt in targets}
        sim._step = 0

        def record(s, kind=kind, ts=ts, ps=ps, cand=cand):
            if s._step % 5 == 0:
                ts.append(s.t)
                ps.append(s.u[pick].copy())
            s._step += 1
            if kind == "table":
                for tt in targets:
                    if abs(s.t - tt) <= 0.5 * period:
                        cand[tt].append((s.t, s.a, s.u.copy()))
        sim.run(t_end, callback=record)
        out["sims"][kind] = sim
        out["probe"][kind] = (np.array(ts), np.array(ps))
        if kind == "table":
            for tt in targets:                            # the instant of largest displacement
                out["snaps"][tt] = max(cand[tt], key=lambda c: np.max(np.abs(c[2])))
        s = sim.summary(t_from=t_start)
        print(f"{kind:6s} final energy {s['final_energy']:9.1f}  lowest purity {s['min_purity']:.4f}  "
              f"largest node error {s['max_node_error']:.4f}", flush=True)
    return out


def save(fig, name):
    RESULTS.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"{name}.{ext}", pad_inches=0.08)
    print("saved", RESULTS / f"{name}.png")


def figure_drives(out, name):
    plt = pub_style()
    from matplotlib.lines import Line2D
    model, mode = out["model"], out["mode"]
    fig, (ax_n, ax_p, ax_f) = plt.subplots(3, 1, figsize=(9.0, 16.4), sharex=True,
                                           gridspec_kw=dict(hspace=0.25, left=0.15, right=0.965, top=0.93, bottom=0.05))
    dash = dict(color=BLACK, linestyle=(0, (4, 3)), linewidth=2.0, zorder=6)

    # A: node positions
    table = out["sims"]["table"]
    t = table.series("t")
    exp = table.node_tracks()[1]
    for k in range(exp.shape[1]):
        ax_n.plot(t, exp[:, k], **dash)
    for kind, sim in out["sims"].items():
        meas = sim.node_tracks()[0]
        for k in range(meas.shape[1]):
            ax_n.plot(sim.series("t"), meas[:, k], color=COLOR[kind], linestyle=LSTYLE[kind], linewidth=LWIDTH[kind])
    ax_n.set_ylim(0, model.L)
    ax_n.set_ylabel("node position $x$")
    ax_n.set_title("(A) Node Positions", pad=58)
    # slope a(t) on a top axis, and the start and end of the sweep
    ax_n.tick_params(top=False)
    top = ax_n.secondary_xaxis("top")
    a_ticks = np.linspace(out["sched"](out["t_start"]), out["sched"](out["t_stop"]), 5)
    t_ticks = out["t_start"] + (a_ticks - a_ticks[0]) / (a_ticks[-1] - a_ticks[0]) * (out["t_stop"] - out["t_start"])
    top.set_xticks(t_ticks)
    top.set_xticklabels([f"{v:+.2f}".replace("+0.00", "0") for v in a_ticks])
    top.set_xlabel("tension slope $a$", labelpad=6)
    top.tick_params(direction="in")
    sweep_labels = []
    for x, lab, ha in ((out["t_start"], "sweep starts\n$t = %d$" % out["t_start"], "right"),
                       (out["t_stop"], "sweep ends\n$t = %d$" % out["t_stop"], "right")):
        sweep_labels.append((x + (-25 if ha == "right" else 25), lab, ha))

    # B: mode purity
    for kind, sim in out["sims"].items():
        ax_p.plot(sim.series("t"), sim.series("purity"), color=COLOR[kind], linestyle=LSTYLE[kind], linewidth=LWIDTH[kind])
    ax_p.set_ylim(0, 1.05)
    ax_p.set_ylabel("purity")
    ax_p.set_title("(B) Mode Purity")

    # C: leading frequency of the response
    w0 = out["w_res"](out["sched"](0.0))
    window, hop = 8 * out["period"], 2 * out["period"]
    tt = np.linspace(0, out["t_end"], 600)
    wr = np.array([out["w_res"](out["sched"](x)) for x in tt])
    lo, hi = 0.6 * wr.min(), 1.6 * wr.max()
    for kind in out["sims"]:
        tc, wc = leading_frequency(*out["probe"][kind], window, hop, (lo, hi))
        ax_f.plot(tc, wc, color=COLOR[kind], linestyle=LSTYLE[kind], linewidth=LWIDTH[kind])
    ax_f.plot(tt, wr, **dash)
    ax_f.set_ylabel(r"angular frequency $\omega$")
    ax_f.set_xlabel("time $t$")
    ax_f.set_title("(C) Leading Frequency of the Response")
    for x, lab, ha in sweep_labels:
        ax_f.text(x, 0.96, lab, ha=ha, va="top", fontsize=17, color="0.3", transform=ax_f.get_xaxis_transform())
    ax_f.set_xlim(0, out["t_end"])

    for ax in (ax_n, ax_p, ax_f):
        for x in (out["t_start"], out["t_stop"]):
            ax.axvline(x, color="0.8", linewidth=0.9, zorder=0)
    handles = [Line2D([], [], color=COLOR[k], linestyle=LSTYLE[k], linewidth=LWIDTH[k], label=SHORT[k]) for k in out["sims"]]
    handles.append(Line2D([], [], label="Predicted", **dash))
    ax_p.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.02), handlelength=1.9,
                handletextpad=0.5, labelspacing=0.45, borderaxespad=0.3, fontsize=17)
    ax_f.set_ylim(top=ax_f.get_ylim()[1] + 0.05 * np.ptp(ax_f.get_ylim()))
    save(fig, name)
    plt.close(fig)


def figure_snapshots(out, name):
    plt = pub_style()
    from matplotlib.lines import Line2D
    model, mode = out["model"], out["mode"]
    sim = out["sims"]["table"]
    kmax = max(16, 3 * mode + 4)
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 12.4), gridspec_kw=dict(hspace=0.5, wspace=0.34, left=0.1, right=0.985, top=0.955,
                                                                           bottom=0.08, width_ratios=[1.25, 1]))
    dash = dict(color=BLACK, linestyle=(0, (4, 3)), linewidth=2.0, zorder=6)
    umax = 0.0
    rows = []
    for tt in out["targets"]:
        ts, a, u = out["snaps"][tt]
        phi = model.mode_shape(a, mode)
        # predicted steady state of the target mode at this tension
        lam = model.eigenvalue(a, mode)
        mu = model.inner(phi, sim.shape)
        amp = abs(mu) * sim.F0 / (model.beta * np.sqrt(lam - 0.25 * model.beta ** 2))
        pred = amp * phi * np.sign(model.inner(u, phi))
        rows.append((tt, a, u, pred))
        umax = max(umax, np.abs(u).max(), np.abs(pred).max())
    for (ax_w, ax_s), (tt, a, u, pred), tag in zip(axes, rows, "ABC"):
        ax_w.axhline(0.0, color="0.75", linewidth=0.9, zorder=0)
        ax_w.plot(model.x, u, color=COLOR["table"], linewidth=LWIDTH["table"])
        ax_w.plot(model.x, pred, **dash)
        ax_w.set_xlim(0, model.L)
        ax_w.set_ylim(-1.6 * umax, 1.2 * umax)
        ax_w.set_ylabel("displacement $u$")
        a_txt = "0" if abs(a) < 5e-3 else f"{a:+.2f}".replace("-", "\u2212")
        ax_w.set_title(f"({tag}) $t$ = {tt:.0f},  $a$ = {a_txt}")
        k, cu = sine_spectrum(model, u, kmax)
        _, cp = sine_spectrum(model, pred, kmax)
        floor = 1e-6 * max(cu.max(), cp.max())
        su, sp = cu > floor, cp > floor                 # harmonics below the floor are not drawn
        ax_s.vlines(k[su], floor, cu[su], color=COLOR["table"], linewidth=2.6)
        ax_s.plot(k[su], cu[su], "o", color=COLOR["table"], markersize=8)
        ax_s.plot(k[sp], cp[sp], "o", markerfacecolor="none", markeredgecolor=BLACK, markersize=12, markeredgewidth=1.6)
        ax_s.set_yscale("log")
        ax_s.set_ylim(floor, 30 * max(cu.max(), cp.max()))
        ax_s.set_xlim(0.3, kmax + 0.7)
        ax_s.set_xticks([1, 5, 10, 15])
        ax_s.set_ylabel("amplitude")
        ax_s.set_title("Sine-series spectrum")
        ax_s.minorticks_off()
    axes[-1, 0].set_xlabel("position $x$")
    axes[-1, 1].set_xlabel("sine harmonic $k$")
    axes[0, 0].legend(handles=[Line2D([], [], color=COLOR["table"], linewidth=LWIDTH["table"], label="Lookup"),
                               Line2D([], [], label="Predicted", **dash)],
                      loc="lower left", handlelength=2.0, ncol=2, columnspacing=1.2, borderaxespad=0.3)
    axes[1, 1].legend(handles=[Line2D([], [], marker="o", linestyle="", color=COLOR["table"], markersize=8, label="Lookup"),
                               Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=BLACK,
                                      markersize=12, markeredgewidth=1.6, label="Predicted")],
                      loc="upper right", handletextpad=0.3, borderaxespad=0.2, ncol=2, columnspacing=0.8)
    save(fig, name)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, default=3)
    p.add_argument("--beta", type=float, default=0.015)
    p.add_argument("--a0", type=float, default=-0.18)
    p.add_argument("--a1", type=float, default=0.18)
    p.add_argument("--t-start", type=float, default=1000.0)
    p.add_argument("--t-stop", type=float, default=3000.0)
    p.add_argument("--t-hold", type=float, default=400.0)
    p.add_argument("--points", type=int, default=301)
    args = p.parse_args(argv)
    print(f"String, mode {args.mode}: a {args.a0} -> {args.a1} from t = {args.t_start:g} to {args.t_stop:g}")
    out = run(args.mode, args.beta, args.a0, args.a1, args.t_start, args.t_stop, args.t_hold, args.points)
    nodes = out["sims"]["table"].node_tracks()[1]
    print("predicted node travel:", np.round(np.abs(nodes[-1] - nodes[0]), 2))
    figure_drives(out, f"fig_drives_mode{args.mode}")
    figure_snapshots(out, f"fig_snapshots_mode{args.mode}")
    return out


if __name__ == "__main__":
    main()
