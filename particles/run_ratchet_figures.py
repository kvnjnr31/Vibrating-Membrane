"""Paper figures and table for the ratchet conveyor, from saved runs.

    python particles/run_ratchet_figures.py

Reads results/fig_ratchet_mode11_{forward,reverse}*.npz (written by
run_ratchet.py with --tag) and writes

    results/fig_ratchet_conveyor.{png,pdf}    density maps and mean position
    results/fig_ratchet_mechanism.{png,pdf}   single-grain paths and where moves start
    results/ratchet_summary.csv               numbers for the table
"""
import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import style                    # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
L, GAMMA = 10.0, 4.0
DASH = (0, (4, 3))


def load(name, kind):
    d = np.load(RESULTS / f"{name}.npz")
    out = {k: d[f"{kind}_{k}"] for k in ("t", "a", "X", "nodes", "on")}
    if f"{kind}_period" in d:
        out["period"] = float(d[f"{kind}_period"][0])
    return out


def sweep_end(res):
    """Time at which the first sweep ends (the drive starts to fade)."""
    off = np.where(res["on"] < 1.0)[0]
    return float(res["t"][off[0]]) if len(off) else float(res["t"][-1])


def cycle_marks(t, period, cycles):
    return [min(int(np.searchsorted(t, k * period)), len(t) - 1) for k in range(cycles + 1)]


def grain_density(res):
    """Share of the grains in each 1% of the string, in percent: (bins, density[bin, time])."""
    bins = np.linspace(0, L, 101)
    return bins, 100.0 * np.array([np.histogram(r, bins)[0] for r in res["X"]]).T / res["X"].shape[1]


def density(ax, res, title, vmax=None):
    bins, dens = grain_density(res)
    mesh = ax.pcolormesh(res["t"], bins, np.pad(dens, ((0, 1), (0, 0))), cmap="Greys", shading="auto",
                         vmin=0, vmax=vmax if vmax is not None else np.percentile(dens, 99.5), rasterized=True)
    nd = np.where(res["on"][:, None] > 0.5, res["nodes"], np.nan)
    for k in range(nd.shape[1]):
        ax.plot(res["t"], nd[:, k], color="k", linestyle=DASH, linewidth=1.0)
    ax.set_ylim(0, L)
    ax.set_ylabel("position $x$")
    ax.set_title(title)
    ax.tick_params(direction="in", top=True, right=True)
    return mesh


def figure_conveyor(fwd, rev, ctl, suffix=""):
    from forcing.run_paper_figures import pub_style
    from matplotlib.lines import Line2D
    plt = pub_style()
    fig = plt.figure(figsize=(17.5, 13.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.045], left=0.07, right=0.93, top=0.95, bottom=0.08,
                          hspace=0.3, wspace=0.22)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[0, 2])
    vmax = max(np.percentile(grain_density(r)[1], 99.5) for r in (fwd, rev, ctl))
    mesh = density(ax_a, fwd, "(A) Ratchet, $a$ Rising", vmax)
    density(ax_b, rev, "(B) Ratchet, $a$ Falling", vmax)
    density(ax_c, ctl, "(C) Control, Drive Always On", vmax)
    cb = fig.colorbar(mesh, cax=cax, extend="max")
    cb.set_label("grains per 1% of string (%)", fontweight="bold")
    cb.ax.tick_params(direction="in")
    styles = ((fwd, "Ratchet, $a$ rising", dict(color="k", linewidth=2.6)),
              (rev, "Ratchet, $a$ falling", dict(color="k", linewidth=2.6, linestyle=(0, (1, 1.3)))),
              (ctl, "Control", dict(color="0.5", linewidth=2.6, linestyle=DASH)))
    for res, lab, st in styles:
        ax_d.plot(res["t"], res["X"].mean(axis=1), **st)
    ax_d.set_ylim(0, L)
    ax_d.set_ylabel("mean grain position")
    ax_d.set_title("(D) Mean Position of All Grains")
    ax_d.legend(handles=[Line2D([], [], label=lab, **st) for _, lab, st in styles], loc="lower left",
                handlelength=2.2, borderaxespad=0.3)
    t_end = min(fwd["t"][-1], rev["t"][-1], ctl["t"][-1])
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xlim(0, t_end)
        ax.set_xticks([x for x in (0, 4000, 8000, 12000, 16000) if x <= t_end])
    for ax in (ax_c, ax_d):
        ax.set_xlabel("time $t$")
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_ratchet_conveyor{suffix}.{ext}")
    plt.close(fig)


def offsets(res, t_end=None, lo=2.5, hi=7.5):
    """Offset of each grain from its nearest node, in node spacings, with the
    sign chosen so that positive is behind the node (the side it came from)."""
    t_end = sweep_end(res) if t_end is None else t_end
    m = res["t"] < t_end
    X, N = res["X"][m], res["nodes"][m]
    X = X[:, (X[0] > lo) & (X[0] < hi)]
    c = N.shape[1] // 2
    direction = -np.sign(N[-1, c] - N[0, c])
    off = np.empty_like(X)
    for i in range(len(X)):
        j = np.abs(X[i][:, None] - N[i][None, :]).argmin(1)
        spacing = np.gradient(N[i])[j]
        off[i] = direction * (X[i] - N[i][j]) / spacing
    dx = np.diff(X, axis=0)
    rest = np.abs(dx) < 0.02
    return off, dx * direction, rest


def events(res, t_end=None, lo=2.5, hi=7.5, spacing=None):
    """Split each grain's path during the first sweep into rests and hops.

    Returns the hop sizes (in node spacings, positive against the motion of
    the nodes) and the lengths of the rests that both start and end inside
    the sweep."""
    t_end = sweep_end(res) if t_end is None else t_end
    m = res["t"] < t_end
    t, X, N = res["t"][m], res["X"][m], res["nodes"][m]
    X = X[:, (X[0] > lo) & (X[0] < hi)]
    c = N.shape[1] // 2
    direction = -np.sign(N[-1, c] - N[0, c])
    spacing = float(np.mean(np.diff(N, axis=1))) if spacing is None else spacing
    dt = t[1] - t[0]
    hops, rests = [], []
    for j in range(X.shape[1]):
        still = np.abs(np.diff(X[:, j])) < 0.02
        runs, start = [], None
        for i, flag in enumerate(still):
            if flag and start is None:
                start = i
            if not flag and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(still)))
        runs = [r for r in runs if r[1] - r[0] >= 3]              # at rest for at least 3 samples
        for k, (i0, i1) in enumerate(runs):
            if i0 > 0 and i1 < len(still):
                rests.append((i1 - i0) * dt)
            if k:
                hops.append(direction * (X[i0, j] - X[runs[k - 1][1], j]) / spacing)
    return np.array(hops), np.array(rests)


def figure_mechanism(fwd, quiet, suffix="", window=(3.0, 7.6), n_paths=6):
    from forcing.run_paper_figures import pub_style
    plt = pub_style()
    fig = plt.figure(figsize=(16.0, 15.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.42, wspace=0.3,
                          left=0.08, right=0.975, top=0.96, bottom=0.07)
    ax = fig.add_subplot(gs[0, :])
    t_end = sweep_end(fwd)
    m = fwd["t"] <= t_end
    t, X, N = fwd["t"][m], fwd["X"][m], fwd["nodes"][m]
    for k in range(N.shape[1]):
        ax.fill_between(t, N[:, k] - quiet, N[:, k] + quiet, color="0.82", linewidth=0,
                        label="Quiet zone" if k == 0 else None)
        ax.plot(t, N[:, k], color="k", linestyle=DASH, linewidth=1.8, label="Node" if k == 0 else None)
    rng = np.random.default_rng(3)
    lo_w, hi_w = window
    start = np.where((X[0] > lo_w + 0.15 * (hi_w - lo_w)) & (X[0] < lo_w + 0.55 * (hi_w - lo_w)))[0]
    for i, j in enumerate(rng.choice(start, n_paths, replace=False)):
        ax.plot(t, X[:, j], color="k", linewidth=2.2, label="Single grain" if i == 0 else None)
    ax.set_xlim(0, t_end)
    ax.set_ylim(lo_w, hi_w)
    ax.set_xlabel("time since the sweep started")
    ax.set_ylabel("position $x$")
    ax.set_title("(A) Grain Paths During One Sweep, $a$ Rising")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.tick_params(direction="in", top=True, right=True)

    hops, rests = events(fwd)
    bx = fig.add_subplot(gs[1, 0])
    bins = np.linspace(-1.5, 2.5, 41)
    bx.hist(hops, bins=bins, color="0.6", edgecolor="0.6")
    for v in (0, 1, 2):
        bx.axvline(v, color="k", linestyle=DASH, linewidth=1.6)
    bx.set_xlabel("hop size (node spacings)")
    bx.set_ylabel("number of hops")
    bx.set_title("(B) Hop Size Between Rests")
    cx = fig.add_subplot(gs[1, 1])
    r = np.sort(rests)
    cx.step(r, 1.0 - np.arange(len(r)) / len(r), where="post", color="k", linewidth=2.6,
            label="Simulation")
    tau = rests.mean() - r[0]
    tt = np.linspace(r[0], r[-1], 100)
    cx.plot(tt, np.exp(-(tt - r[0]) / tau), color="0.5", linestyle=DASH, linewidth=2.6,
            label=f"Exponential,\nmean {rests.mean():.0f}")
    cx.set_yscale("log")
    cx.set_xlabel("rest time $t$")
    cx.set_ylabel("fraction of rests $> t$")
    cx.set_title("(C) Distribution of Rest Times")
    cx.legend(loc="upper right")
    cx.minorticks_off()
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_ratchet_mechanism{suffix}.{ext}", dpi=200)
    plt.close(fig)
    return hops, rests


def summarise(name, kind):
    r = load(name, kind)
    cycles = int(round(r["t"][-1] / r["period"]))
    i = cycle_marks(r["t"], r["period"], cycles)
    X0, X1 = r["X"][i[0]], r["X"][i[-1]]
    mean = [float(r["X"][k].mean()) for k in i]
    return dict(run=name, kind=kind, cycles=cycles, mean_start=mean[0], mean_end=mean[-1],
                first_cycle=mean[1] - mean[0], per_cycle=(mean[-1] - mean[0]) / cycles,
                median_displacement=float(np.median(X1 - X0)),
                lower_half_start=int((X0 < 5).sum()), lower_half_end=int((X1 < 5).sum()), grains=len(X0))


def main(argv=None):
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, default=5)
    p.add_argument("--quiet", type=float, default=None, help="half-width of the quiet zone")
    p.add_argument("--window", type=float, nargs=2, default=None, help="x range of the path panel")
    p.add_argument("--set", default="", help="run set inside the names, e.g. _g3k for the 3000-grain runs")
    p.add_argument("--names", nargs=2, default=None, help="file stems of the rising and falling runs, e.g. forward reverse (mode 11)")
    args = p.parse_args(argv)
    style()
    base = f"fig_ratchet_mode{args.mode}"
    fn, rn = (f"{base}{args.set}_fwd_s0", f"{base}{args.set}_rev_s0") if args.names is None else \
        tuple(f"{base}_{n}" for n in args.names)
    fwd, rev = load(fn, "ratchet"), load(rn, "ratchet")
    ctl = load(fn, "control")
    suffix = f"_mode{args.mode}{args.set}"
    figure_conveyor(fwd, rev, ctl, suffix)
    quiet = args.quiet
    if quiet is None:
        from forcing.string_model import TensionString
        from particles.experiment import quiet_half_width
        quiet = quiet_half_width(TensionString(N=201, beta=0.05), float(fwd["a"][0]), args.mode, GAMMA)
    window = args.window or (2.0, 8.0)
    hops, rests = figure_mechanism(fwd, quiet, suffix, tuple(window))
    print(f"quiet half-width {quiet:.3f}; first sweep ends at t = {sweep_end(fwd):.0f}; "
          f"node spacing {np.mean(np.diff(fwd['nodes'], axis=1)):.2f}")
    print(f"hops: {len(hops)}; same node {np.mean(np.abs(hops) <= 0.5):.2f}, one back "
          f"{np.mean((hops > 0.5) & (hops < 1.5)):.2f}, two or more back {np.mean(hops >= 1.5):.2f}, "
          f"forward {np.mean(hops < -0.5):.2f}; mean {hops.mean():+.3f} spacings")
    print(f"rests: {len(rests)}; mean {rests.mean():.0f}, median {np.median(rests):.0f}, "
          f"share longer than twice the mean {np.mean(rests > 2 * rests.mean()):.3f} "
          f"(exponential: {np.exp(-(2 * rests.mean() - rests.min()) / (rests.mean() - rests.min())):.3f})")
    if args.names is not None:
        return
    rows = []
    import re
    for f in sorted(glob.glob(str(RESULTS / f"{base}{args.set}_*_s*.npz"))):
        name = pathlib.Path(f).stem
        if not re.fullmatch(re.escape(base + args.set) + r"_(fwd|rev)_s\d+", name):
            continue
        d = np.load(f)
        for kind in ("ratchet", "control"):
            if f"{kind}_t" in d:
                rows.append(summarise(name, kind))
    keys = list(rows[0])
    with open(RESULTS / f"ratchet_summary_mode{args.mode}{args.set}.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    for label, res in (("a rising", fwd), ("a falling", rev)):
        off, dx, rest = offsets(res)
        print(f"{label}: at rest {rest.mean():.2f}; displacement against the nodes per grain per sweep "
              f"at rest {dx[rest].sum() / off.shape[1]:+.2f}, moving {dx[~rest].sum() / off.shape[1]:+.2f}")


if __name__ == "__main__":
    main()
