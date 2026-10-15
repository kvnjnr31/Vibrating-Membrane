"""Conveyor summary as a band: mean over modes with the highest and lowest mode.

    python particles/run_conveyor_band_figure.py --modes 5 7 9 11 13 15 19

Reads results/conveyor_hops_mode<n>_v2.csv and writes
results/Fig7.{png,pdf} and conveyor_band.csv.

At each sweep rate the middle curve is the mean over the modes.  The outer
curves are the largest and smallest value among the modes, and the space
between them is filled.  All modes share the same rates.

By default the three curves are smoothed: each is replaced by a polynomial
in log(rate) fitted to its 18 values (--degree, default 6).  --raw draws
the values as measured.  The table always holds both.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import pub_style                                  # noqa: E402
from particles.experiment import RESULTS                                     # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--modes", type=int, nargs="+", default=[5, 7, 9, 11, 13, 15, 19])
    p.add_argument("--tag", default="_v2")
    p.add_argument("--degree", type=int, default=6, help="degree of the smoothing polynomial in log(rate)")
    p.add_argument("--raw", action="store_true", help="draw the measured values without smoothing")
    args = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pub_style()
    data = [np.genfromtxt(RESULTS / f"conveyor_hops_mode{m}{args.tag}.csv", delimiter=",", names=True) for m in args.modes]
    rate = data[0]["rate"]
    assert all(np.allclose(d["rate"], rate) for d in data)
    speed = 1e4 * np.array([d["shift"] / d["period"] for d in data])
    hop = np.array([d["hop"] for d in data])
    modes = np.array(args.modes)

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 13.6), sharex=True, gridspec_kw=dict(
        hspace=0.2, left=0.15, right=0.965, top=0.955, bottom=0.075))
    out, smooth, dev = [rate], [], []
    for ax, y in ((axes[0], speed), (axes[1], hop)):
        lo, hi, mid = y.min(axis=0), y.max(axis=0), y.mean(axis=0)
        raw = [mid, lo, hi]
        x = rate
        if not args.raw:
            x = np.geomspace(rate[0], rate[-1], 300)
            fit = [np.polynomial.Polynomial.fit(np.log(rate), c, args.degree) for c in raw]
            dev.append([float(np.max(np.abs(f(np.log(rate)) - c)) / mid.max()) for f, c in zip(fit, raw)])
            mid, lo, hi = [f(np.log(x)) for f in fit]
            lo, hi = np.minimum(lo, mid), np.maximum(hi, mid)
            smooth += [f(np.log(rate)) for f in fit]
        ax.fill_between(x, lo, hi, color="0.82", linewidth=0, label="Range over modes")
        ax.plot(x, hi, color="0.3", linewidth=2.2, linestyle=(0, (6, 3)), label="Highest value")
        ax.plot(x, lo, color="0.3", linewidth=2.4, linestyle=(0, (1.2, 1.6)), label="Lowest value")
        ax.plot(x, mid, color="k", linewidth=3.2, label="Mean over modes")
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        out += raw + [modes[y.argmin(axis=0)], modes[y.argmax(axis=0)]]
    axes[0].set_ylabel(r"transport speed ($10^{-4}$)")
    axes[0].set_title(f"(A) Transport Speed, Modes {modes.min()} to {modes.max()}")
    axes[0].set_ylim(0, 11)
    axes[1].set_ylabel("nodes passed per cycle")
    axes[1].set_title("(B) Mean Hop in One Cycle")
    axes[1].set_xlabel(r"sweep rate $da/dt$")
    axes[1].set_ylim(0, 1.7)
    axes[1].axhline(1.0, color="0.5", linewidth=1.2, linestyle=(0, (1, 2)))
    axes[1].legend(loc="upper right", handlelength=2.4)
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"Fig7.{ext}", dpi=200)
    head = "rate,speed_mean,speed_min,speed_max,mode_of_min_speed,mode_of_max_speed,hop_mean,hop_min,hop_max,mode_of_min_hop,mode_of_max_hop"
    if smooth:
        out += smooth
        head += ",speed_mean_smooth,speed_min_smooth,speed_max_smooth,hop_mean_smooth,hop_min_smooth,hop_max_smooth"
        print("largest shift of a point by smoothing, as a share of the peak (mean, min, max):", np.round(100 * np.array(dev), 1), "%")
    np.savetxt(RESULTS / "conveyor_band.csv", np.array(out).T, delimiter=",", header=head, comments="", fmt="%.5g")
    i = int(np.argmax(out[1]))
    print(f"mean curve peaks at rate {rate[i]:.2e}: {out[1][i]:.2f} (range {out[2][i]:.2f} to {out[3][i]:.2f})")
    print("mode of min speed:", out[4].astype(int)); print("mode of max speed:", out[5].astype(int))
    print("band width / mean (%):", np.round(100 * (out[3] - out[2]) / out[1]).astype(int))


if __name__ == "__main__":
    main()
