"""Figure for the acceleration sweep (grey scale, line styles only).

    python acceleration_sweep/run_acceleration_figure.py [--quick]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
tag = "_quick" if "--quick" in sys.argv else ""
R = np.load(os.path.join(OUT, f"accel_sweep{tag}.npz"))
gam, disc = R["gammas"], int(R["discard"])
sp = float(R["L"]) / int(R["n_mode"]) * 100                   # node spacing, cm
STYLE = {"held": ("-", "o", "held in contact"), "mu0.05": ("--", "s", r"rolling, $\mu_r=0.05$"),
         "mu0.01": ("-.", "^", r"rolling, $\mu_r=0.01$"), "mu0.002": (":", "v", r"rolling, $\mu_r=0.002$"),
         "rough_held": ("-", "D", "rough, held in contact"), "rough_mu0.01": ("--", "x", r"rough, rolling $\mu_r=0.01$")}


def summary(name, stat):
    kept = R[f"blocks_{name}"][:, disc:, :, stat]             # (gamma, block, group)
    m = kept.mean(1)
    return m.mean(1), m.std(1, ddof=1) / np.sqrt(m.shape[1]), kept.mean(2)


matplotlib.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                            "legend.frameon": False, "lines.linewidth": 1.2})
fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))
for a, names, ttl in ((ax[0, 0], ["held", "mu0.05", "mu0.01", "mu0.002"], "(a) smooth grains"),
                      (ax[0, 1], ["rough_held", "rough_mu0.01", "held"], "(b) rough grains (roughness 0.3)")):
    for n in names:
        ls, mk, lab = STYLE[n]
        m, se, _ = summary(n, 0)
        grey = "0.55" if (n == "held" and a is ax[0, 1]) else "k"
        a.errorbar(gam, m * 100, yerr=se * 100, color=grey, ls=ls, marker=mk, ms=3, mfc="w", capsize=1.5,
                   label=lab + (" (smooth)" if grey != "k" else ""))
    a.axhline(sp / 4, color="0.4", lw=0.7, ls=(0, (1, 3)))
    a.text(gam[-1], sp / 4, " uniform", va="bottom", ha="right", fontsize=7, color="0.3")
    a.set(xlabel=r"antinode acceleration $\Gamma_0$", ylabel="mean distance to nearest node (cm)",
          ylim=(0, sp / 2), title=ttl)
    a.legend(fontsize=7.5, loc="lower left", ncol=2 if a is ax[0, 0] else 1)
a = ax[1, 0]
for n, stat, ls, mk, lab in (("rough_held", 1, "-", "D", "near a node, rough held"),
                             ("rough_mu0.01", 1, "--", "x", r"near a node, rough rolling"),
                             ("mu0.002", 2, ":", "v", r"near an antinode, smooth $\mu_r=0.002$"),
                             ("mu0.01", 2, "-.", "^", r"near an antinode, smooth $\mu_r=0.01$")):
    m, se, _ = summary(n, stat)
    a.errorbar(gam, m, yerr=se, color="k", ls=ls, marker=mk, ms=3, mfc="w", capsize=1.5, label=lab)
a.axhline(0.2, color="0.4", lw=0.7, ls=(0, (1, 3)))
a.set(xlabel=r"antinode acceleration $\Gamma_0$", ylabel="fraction of grains", ylim=(0, 1.45), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
      title=f"(c) grains within {0.1*sp:.0f} cm of a node / antinode")
a.legend(fontsize=7.5)
a = ax[1, 1]
for n in ("held", "mu0.01", "rough_held", "rough_mu0.01"):
    ls, mk, lab = STYLE[n]
    _, _, blk = summary(n, 0)
    a.plot(gam, (blk[:, -1] - blk[:, 0]) * 100, color="k", ls=ls, marker=mk, ms=3, mfc="w", label=lab)
a.axhline(0, color="0.4", lw=0.7)
a.set(xlabel=r"antinode acceleration $\Gamma_0$", ylabel="change, block 3 to block 6 (cm)",
      title="(d) drift still present in the averaging window")
a.legend(fontsize=7.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_accel_sweep{tag}.{ext}"), dpi=200)
print("saved fig_accel_sweep" + tag)
