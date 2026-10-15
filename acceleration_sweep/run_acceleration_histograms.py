"""Where the grains are at the end of the run, folded onto one node.

Every grain's signed offset from its nearest node is pooled into a single
distribution: 0 is the node, the two ends are the neighbouring antinodes.
One panel per grain type, three antinode accelerations each.  Positions are
the snapshot at the end of the run (not a time average).

    python acceleration_sweep/run_acceleration_histograms.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
R = np.load(os.path.join(OUT, "accel_sweep.npz"))
gam = R["gammas"]
sp = float(R["L"]) / int(R["n_mode"]) * 100                  # node spacing, cm
half = sp / 2                                                # node to antinode
PANELS = [("held", "smooth, held in contact"), ("mu0.05", r"smooth, rolling $\mu_r=0.05$"),
          ("mu0.01", r"smooth, rolling $\mu_r=0.01$"), ("mu0.002", r"smooth, rolling $\mu_r=0.002$"),
          ("rough_held", "rough, held in contact"), ("rough_mu0.01", r"rough, rolling $\mu_r=0.01$")]
CURVES = [(2.0, "-", "0.0"), (4.0, "--", "0.0"), (6.0, ":", "0.0")]
bins = np.linspace(-half, half, 41)                          # 0.5 cm bins

matplotlib.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                            "legend.frameon": False, "lines.linewidth": 1.3})
fig, axes = plt.subplots(3, 2, figsize=(10, 9.5), sharex=True)
order = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]
for a, (name, title), tag in zip(order, PANELS, "abcdef"):
    for G, ls, col in CURVES:
        x = R[f"x_{name}"][int(np.argmin(abs(gam - G)))] * 100
        s = (x + half) % sp - half                           # signed offset from nearest node
        h, _ = np.histogram(s, bins=bins, density=True)
        a.step(np.r_[bins[:-1], bins[-1]], np.r_[h, h[-1]], where="post", color=col, ls=ls,
               label=rf"$\Gamma_0={G:g}$")
    a.axhline(1 / sp, color="0.5", lw=0.7, ls=(0, (1, 3)))
    a.set_yscale("log"); a.set_ylim(2e-3, 3)
    a.set_title(f"({tag}) {title}", loc="left")
    a.set_ylabel("grain density (1/cm)")
    a.legend(fontsize=8, loc="upper center", ncol=3)
for a in axes[-1]:
    a.set_xlabel("offset from nearest node (cm);  0 = node,  $\\pm$%g = antinode" % half)
    a.set_xlim(-half, half)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_accel_histograms.{ext}"), dpi=200)
print("saved fig_accel_histograms")
