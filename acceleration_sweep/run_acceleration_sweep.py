"""Sweep the antinode acceleration Gamma0 for each contact rule.

Protocol: N grains start uniformly on the string; the run is cut into
--blocks equal time blocks and the first --discard are thrown away as
transient; the rest are averaged.  The standard error comes from --groups
independent subgroups of the grains.

    python acceleration_sweep/run_acceleration_sweep.py          # ~45 min on 2 cores
    python acceleration_sweep/run_acceleration_sweep.py --quick  # ~2 min check
"""
import argparse, csv, os
from multiprocessing import Pool
import numpy as np
from standing_wave_grains import run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
P = dict(L=1.0, n_mode=5, f=20.0, e_n=0.3, e_t=0.8, stick=0.05, steps=400)
# name, rolling resistance, grain in contact can roll, roughness
CASES = [("held", 0.0, False, 0.0), ("mu0.002", 0.002, True, 0.0), ("mu0.01", 0.01, True, 0.0),
         ("mu0.05", 0.05, True, 0.0), ("rough_held", 0.0, False, 0.3), ("rough_mu0.01", 0.01, True, 0.3)]


def job(a):
    name, mu, roll, rough, G, N, blocks, block, groups = a
    stats, x = run(G, mu, roll, N, blocks * block, P["steps"], P["L"], P["n_mode"], P["f"],
                   P["e_n"], P["e_t"], P["stick"], rough, groups, 1)
    return name, G, stats.reshape(blocks, block, groups, 3).mean(1), x


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grains", type=int, default=2000)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--discard", type=int, default=2)
    ap.add_argument("--block-cycles", type=int, default=300)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    gammas = np.round(np.arange(0.6, 6.01, 0.2), 2)
    if a.quick:
        gammas, a.grains, a.block_cycles = gammas[::4], 200, 50
    run(1.5, 0.01, True, 8, 2, P["steps"], P["L"], P["n_mode"], P["f"], P["e_n"], P["e_t"], P["stick"], 0.3, 4, 0)
    jobs = [(n, mu, r, ro, G, a.grains, a.blocks, a.block_cycles, a.groups) for n, mu, r, ro in CASES for G in gammas]
    B = {n: np.zeros((len(gammas), a.blocks, a.groups, 3)) for n, *_ in CASES}
    X = {n: np.zeros((len(gammas), a.grains)) for n, *_ in CASES}
    tag = "_quick" if a.quick else ""
    with Pool(a.workers) as pool:
        for name, G, b, x in pool.imap_unordered(job, jobs):
            i = int(np.argmin(abs(gammas - G)))
            B[name][i], X[name][i] = b, x
            print(name, G, np.round(b[:, :, 0].mean(1) * 100, 2), flush=True)
    np.savez(os.path.join(OUT, f"accel_sweep{tag}.npz"), gammas=gammas, discard=a.discard,
             **{f"blocks_{n}": B[n] for n in B}, **{f"x_{n}": X[n] for n in X}, **P)
    with open(os.path.join(OUT, f"accel_sweep{tag}.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["case", "gamma0", "mean_node_distance_cm", "se_cm", "drift_in_window_cm",
                     "node_fraction", "node_fraction_se", "antinode_fraction", "antinode_fraction_se"])
        for n in B:
            kept = B[n][:, a.discard:]                       # (gamma, block, group, stat)
            m = kept.mean(1)                                 # per subgroup
            se = m.std(1, ddof=1) / np.sqrt(a.groups)
            drift = (kept[:, -1] - kept[:, 0]).mean(1)[:, 0]
            for i, G in enumerate(gammas):
                mm = m[i].mean(0)
                wr.writerow([n, G, f"{mm[0]*100:.3f}", f"{se[i,0]*100:.3f}", f"{drift[i]*100:.3f}",
                             f"{mm[1]:.4f}", f"{se[i,1]:.4f}", f"{mm[2]:.4f}", f"{se[i,2]:.4f}"])
    print("DONE", flush=True)
