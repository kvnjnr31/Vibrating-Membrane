"""How fast can the nodes move before the grains are left behind?

For each drive: settle the grains once at slope a0, then sweep to a1 at each
rate from that same settled state and hold.  Records how far the grains
trail the nodes and how many stay with them.

Writes results/<model>_grain_rate_scan.{png,csv}.

    python particles/run_grain_rate_scan.py
    python particles/run_grain_rate_scan.py --model membrane --drives pll+agc
    python particles/run_grain_rate_scan.py --rates 1e-5 3e-5 1e-4 --gamma 8
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from particles.experiment import sweep, following_summary, quiet_half_width, RESULTS  # noqa: E402
from particles.run_grains import add_grain_arguments, build                           # noqa: E402

FIELDS = ["model", "drive", "rate", "sweep_time", "node_speed", "node_travel", "window",
          "band_trailing_before", "band_trailing_end_of_sweep", "band_trailing_after_hold",
          "member_trailing_end_of_sweep", "member_trailing_after_hold", "kept_end_of_sweep",
          "kept_after_hold", "at_any_node_after_hold", "acceleration_min", "acceleration_max"]


def plot(rows, args, quiet, name, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from forcing import plotstyle as ps
    ps.apply()
    drives = list(dict.fromkeys(r["drive"] for r in rows))
    panels = [("member_trailing_end_of_sweep", "How far grains trail their node at the end of the sweep"),
              ("member_trailing_after_hold", "... and after holding for %g time units" % args.t_hold),
              ("kept_after_hold", "Share of grains still with their node after the hold")]
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.8), sharex=True,
                             gridspec_kw=dict(hspace=0.3, left=0.11, right=0.97, top=0.86, bottom=0.07))
    travel = rows[0]["node_travel"]
    for ax, (key, title) in zip(axes, panels):
        for kind in drives:
            sel = sorted((r for r in rows if r["drive"] == kind), key=lambda r: r["rate"])
            ax.plot([r["rate"] for r in sel], [r[key] for r in sel], color=ps.DRIVE_COLOR[kind],
                    marker="o", markersize=6, markeredgecolor=ps.SURFACE, markeredgewidth=1.0)
        ax.set_xscale("log")
        ax.set_title(title)
        if key.startswith("member"):
            ax.axhline(travel, color=ps.REFERENCE, linestyle=(0, (1, 2)), linewidth=1.4)
            ax.axhline(quiet, color=ps.REFERENCE, linestyle=(0, (4, 3)), linewidth=1.4)
            top = max(1.1 * travel, 1.05 * max(r[key] for r in rows))
            ax.set_ylim(min(0.0, ax.get_ylim()[0]), top)
        else:
            ax.set_ylim(0, 1.05)
    axes[-1].set_xlabel("sweep rate |da/dt|")
    handles = [Line2D([], [], color=ps.DRIVE_COLOR[k], marker="o", markersize=6,
                      markeredgecolor=ps.SURFACE, label=ps.DRIVE_LABEL[k]) for k in drives]
    handles += [Line2D([], [], color=ps.REFERENCE, linestyle=(0, (1, 2)), linewidth=1.4,
                       label="Distance the nodes travel (grains did not move)"),
                Line2D([], [], color=ps.REFERENCE, linestyle=(0, (4, 3)), linewidth=1.4,
                       label="Half-width of the quiet zone around a node")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.54, 0.96))
    fig.suptitle(f"Grains on the {args.model}: sweep-rate scan, slope {args.a0} to {args.a1}, "
                 f"peak acceleration {args.gamma:g} g", x=0.11, y=0.985, ha="left", fontsize=12,
                 fontweight="bold")
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.png"
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_grain_arguments(p)
    p.add_argument("--drives", nargs="+", default=["pll+agc", "table+gain"])
    p.add_argument("--rates", type=float, nargs="+", default=[3e-5, 1e-4, 3e-4, 1e-3, 3e-3])
    p.add_argument("--name", default=None)
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)

    rows, quiet = [], None
    for kind in args.drives:
        settled = build(args, kind)
        if quiet is None:
            quiet = quiet_half_width(settled.model, args.a0, settled.mode, args.gamma)
            print(f"half-width of the quiet zone around a node: {quiet:.3f}")
        for rate in args.rates:
            res = sweep(settled, args.a1, rate, args.t_hold, verbose=False)
            s = following_summary(res)
            s.update(model=args.model, drive=kind, rate=rate,
                     sweep_time=abs(args.a1 - args.a0) / rate,
                     node_speed=s["node_travel"] * rate / abs(args.a1 - args.a0))
            rows.append(s)
            print(f"{kind:11s} rate {rate:<7g} node speed {s['node_speed']:.2e}  trailing: "
                  f"{s['member_trailing_end_of_sweep']:+.3f} at end of sweep, "
                  f"{s['member_trailing_after_hold']:+.3f} after hold;  kept "
                  f"{100 * s['kept_after_hold']:3.0f}%;  acceleration "
                  f"{s['acceleration_min']:.3f}-{s['acceleration_max']:.3f}", flush=True)
    name = args.name or f"{args.model}_grain_rate_scan"
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{name}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print("saved", RESULTS / f"{name}.csv")
    print("saved", plot(rows, args, quiet, name, args.show))
    return rows


if __name__ == "__main__":
    main()
