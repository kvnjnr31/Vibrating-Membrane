"""Extra rough-grain runs for the density figure (main Fig. 5, Supplementary Fig. S9).

run_acceleration_sweep.py covers rough grains held in contact and rolling with
mu_r = 0.01.  Fig. 5 also needs rough grains rolling with mu_r = 0.05 and
0.002, at Gamma0 = 2, 4 and 6.  Same protocol and settings as the sweep:
2000 grains, 1800 drive periods, mode 5 of a 1 m string at 20 Hz, seed 1.

    python acceleration_sweep/run_acceleration_extra.py        # a few minutes on 2 cores

Writes results/accel_histograms_extra.npz with the final grain positions
x_rough_mu<mu>_g<Gamma0>.
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from run_acceleration_sweep import P
from standing_wave_grains import run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CASES = [("rough_mu0.05", 0.05), ("rough_mu0.002", 0.002)]
GAMMAS = (2.0, 4.0, 6.0)
ROUGH = 0.3


def job(arg):
    name, mu, G, N, cycles, groups = arg
    _, x = run(G, mu, True, N, cycles, P["steps"], P["L"], P["n_mode"], P["f"],
               P["e_n"], P["e_t"], P["stick"], ROUGH, groups, 1)
    return f"x_{name}_g{G:g}", x


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grains", type=int, default=2000)
    ap.add_argument("--cycles", type=int, default=1800)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    jobs = [(n, mu, G, a.grains, a.cycles, a.groups) for n, mu in CASES for G in GAMMAS]
    with Pool(a.workers) as pool:
        out = dict(pool.imap_unordered(job, jobs))
    for k in sorted(out):
        print(k, "share within 2 cm of a node:", round(float(np.mean(np.abs((out[k] * 100 + 10) % 20 - 10) < 2)), 3))
    np.savez(os.path.join(OUT, "accel_histograms_extra.npz"), **out)
    print("saved", os.path.join(OUT, "accel_histograms_extra.npz"))
