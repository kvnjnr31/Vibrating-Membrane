"""Summary figure: ratchet transport speed against sweep rate for several modes.

    python particles/run_ratchet_scan_figure.py --modes 3 5 7 11

Reads results/ratchet_scan_mode<n>.npz (from run_ratchet_scan.py) and writes
results/fig_ratchet_tradeoff.{png,pdf} and ratchet_tradeoff.csv.

The speed is the shift of the mean grain position per unit time, taken over
the first cycles, before the grains pile up at the end of the string.
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from forcing.run_paper_figures import style                                  # noqa: E402
from particles.experiment import RESULTS                                     # noqa: E402

STYLES = {3: dict(linestyle=":", marker="v"), 5: dict(linestyle="-", marker="o"),
          7: dict(linestyle="--", marker="s"), 11: dict(linestyle="-.", marker="^")}


def load(mode, limit):
    d = np.load(RESULTS / f"ratchet_scan_mode{mode}.npz")
    rows = {}
    for rate, seed, period, spacing in d["meta"]:
        X = d[f"r{rate:.1e}_s{int(seed)}"]
        m = X.mean(axis=1) - X[0].mean()
        n = max(1, int(np.argmax(np.abs(m) > limit)) if np.any(np.abs(m) > limit) else len(m) - 1)
        rows.setdefault(rate, []).append((m[n] / (n * period), m[n] / n, period, spacing))
    rates = np.array(sorted(rows))
    v = np.array([[r[0] for r in rows[x]] for x in rates])
    step = np.array([[r[1] for r in rows[x]] for x in rates])
    return rates, v, step, np.array([rows[x][0][2] for x in rates]), rows[rates[0]][0][3]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--modes", type=int, nargs="+", default=[3, 5, 7, 11])
    p.add_argument("--limit", type=float, default=1.0, help="use cycles until the mean has moved this far")
    args = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    fig, axes = plt.subplots(2, 1, figsize=(11, 10.4), sharex=True, gridspec_kw=dict(
        hspace=0.22, left=0.11, right=0.97, top=0.95, bottom=0.08))
    table = []
    for mode in args.modes:
        rates, v, step, period, spacing = load(mode, args.limit)
        st = dict(color="k", linewidth=2.0, markersize=8, markerfacecolor="white", markeredgewidth=1.6, **STYLES[mode])
        for ax, y, scale in ((axes[0], v, 1e4), (axes[1], step, 1.0 / spacing)):
            ax.errorbar(rates, scale * y.mean(axis=1), yerr=scale * y.std(axis=1), capsize=3,
                        label=f"Mode {mode} (spacing {spacing:.1f})", **st)
        for r, vv, ss, per in zip(rates, v, step, period):
            table.append([mode, r, round(per, 0), round(spacing, 3), f"{ss.mean():.3f}", f"{ss.std():.3f}",
                          f"{vv.mean():.3e}", f"{vv.std():.2e}"])
    axes[0].set_ylabel(r"transport speed ($10^{-4}$)")
    axes[0].set_title("A: Transport Speed")
    axes[1].set_ylabel("shift per cycle / node spacing")
    axes[1].set_title("B: Grain Shift in One Cycle")
    axes[1].set_xlabel("sweep rate of the tension slope")
    for ax in axes:
        ax.set_xscale("log")
        ax.axhline(0, color="0.6", linewidth=0.8)
    axes[1].legend(loc="upper right")
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_ratchet_tradeoff.{ext}")
    with open(RESULTS / "ratchet_tradeoff.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "rate", "cycle_length", "node_spacing", "shift_per_cycle", "sd", "speed", "sd"])
        w.writerows(table)
    for row in table:
        print(*row)


if __name__ == "__main__":
    main()
