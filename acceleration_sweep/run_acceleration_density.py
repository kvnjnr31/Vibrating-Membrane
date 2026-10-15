"""Where the grains are at the end of the run, as smooth density curves.

Same data and layout as run_acceleration_histograms.py, but each histogram
is replaced by a smooth curve.  Every grain's signed offset from its nearest
node is pooled (0 = node, the two ends = the neighbouring antinodes) and
spread with a Gaussian of width --width (default 0.8 cm), so each cluster
of grains shows as a bell curve.  The offset is periodic over one node
spacing, so the smoothing wraps around the ends.  The two sides of the node
are equivalent, so each curve is averaged with its mirror image
(--no-mirror turns this off).  The density axis is linear; --log gives the
log axis of the histogram figure.

    python acceleration_sweep/run_acceleration_density.py

With --raw the unsmoothed histograms are drawn instead (0.5 cm bins, log
density axis, no mirror averaging) and written to
results/fig_accel_raw.{png,pdf} and results/accel_raw.csv.

Reads results/accel_sweep.npz and results/accel_histograms_extra.npz.
Writes results/Fig5.{png,pdf} and results/accel_density.csv.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
sys.path.insert(0, os.path.join(HERE, ".."))
from forcing.run_paper_figures import pub_style                              # noqa: E402

ROWS = [("held", "Held in Contact"), ("mu0.05", r"Rolling, $\mu_r$ = 0.05"),
        ("mu0.01", r"Rolling, $\mu_r$ = 0.01"), ("mu0.002", r"Rolling, $\mu_r$ = 0.002")]
CURVES = [(2.0, "-", "0.0", 2.8), (4.0, (0, (6, 3)), "0.0", 2.6), (6.0, (0, (1.2, 1.6)), "0.35", 3.0)]


def density(s, sp, width, n=800):
    """Periodic Gaussian smoothing of the offsets s over one node spacing sp."""
    edges = np.linspace(-sp / 2, sp / 2, n + 1)
    h = np.histogram(s, bins=edges)[0] / (len(s) * sp / n)
    x = 0.5 * (edges[:-1] + edges[1:])
    d = np.minimum(np.abs(x - x[0]), sp - np.abs(x - x[0]))
    kern = np.exp(-0.5 * (d / width) ** 2)
    rho = np.real(np.fft.ifft(np.fft.fft(h) * np.fft.fft(kern / kern.sum())))
    return np.r_[x, sp / 2], np.r_[rho, rho[0]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--width", type=float, default=0.8, help="smoothing width in cm")
    p.add_argument("--raw", action="store_true", help="draw the unsmoothed histograms (0.5 cm bins, log axis)")
    p.add_argument("--log", action="store_true", help="log density axis")
    p.add_argument("--no-mirror", action="store_true", help="do not average the two sides of the node")
    args = p.parse_args(argv)
    if args.raw:
        args.log = args.no_mirror = True
    R = np.load(os.path.join(OUT, "accel_sweep.npz"))
    E = np.load(os.path.join(OUT, "accel_histograms_extra.npz"))
    gam = R["gammas"]
    sp = float(R["L"]) / int(R["n_mode"]) * 100                  # node spacing, cm
    half = sp / 2

    def positions(name, rough, G):
        if not rough:
            return R[f"x_{name}"][int(np.argmin(abs(gam - G)))]
        key = "rough_held" if name == "held" else "rough_" + name
        if f"x_{key}" in R.files:
            return R[f"x_{key}"][int(np.argmin(abs(gam - G)))]
        return E[f"x_{key}_g{G:g}"]

    name_out = "raw" if args.raw else "density"
    pub_style()
    fig, axes = plt.subplots(4, 2, figsize=(16, 20.6), sharex=True, sharey=True, gridspec_kw=dict(
        hspace=0.3, wspace=0.12, left=0.1, right=0.975, top=0.925, bottom=0.06))
    table, head, top = [], ["offset_cm"], 0.0
    for j, (rough, col_title) in enumerate(((False, "Smooth Grains"), (True, "Rough Grains, $r$ = 0.3"))):
        for i, (name, title) in enumerate(ROWS):
            ax = axes[i, j]
            for G, ls, col, lw in CURVES:
                s = (positions(name, rough, G) * 100 + half) % sp - half
                if args.raw:
                    edges = np.linspace(-half, half, 41)
                    h = np.histogram(s, bins=edges, density=True)[0]
                    x, rho = edges, np.r_[h, h[-1]]
                else:
                    x, rho = density(s, sp, args.width)
                if not args.no_mirror:
                    rho = 0.5 * (rho + rho[::-1])
                top = max(top, rho.max())
                ax.plot(x, rho, color=col, linestyle=ls, linewidth=lw * (0.8 if args.raw else 1.0),
                        drawstyle="steps-post" if args.raw else "default", label=rf"$\Gamma_0$ = {G:g}")
                table.append(rho)
                head.append(f"{'rough' if rough else 'smooth'}_{name}_G{G:g}")
            ax.axhline(1 / sp, color="0.5", linewidth=1.2, linestyle=(0, (1, 3)))
            if args.log:
                ax.set_yscale("log")
                ax.set_ylim(2e-3, 4)
            ax.set_xlim(-half, half)
            ax.set_xticks([-10, -5, 0, 5, 10])
            ax.set_title(f"({'ABCD'[i]}{j + 1}) {title}")
            if j == 0:
                ax.set_ylabel("grain density (1/cm)")
        axes[0, j].text(0.5, 1.2, col_title, transform=axes[0, j].transAxes, ha="center", va="bottom",
                        fontsize=24, fontweight="bold")
        axes[-1, j].set_xlabel("offset from nearest node (cm)")
    if not args.log:
        axes[0, 0].set_ylim(0, 1.12 * top)
        axes[0, 0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axes[0, 0].legend(loc="upper center", ncol=3, handlelength=2.0, columnspacing=1.0, handletextpad=0.5)
    fname = f"fig_accel_{name_out}" if args.raw or args.log else "Fig5"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"), dpi=200)
    if args.raw:
        x, table, head[0] = x[:-1], [t[:-1] for t in table], "bin_left_edge_cm"
    np.savetxt(os.path.join(OUT, f"accel_{name_out}.csv"), np.column_stack([x] + table), delimiter=",",
               header=",".join(head), comments="", fmt="%.5g")
    print(f"saved {fname}")


if __name__ == "__main__":
    main()
