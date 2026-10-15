"""Grains on an ideal standing wave, h(x, t) = A sin(kx) sin(wt), in SI units.

Contact rules follow particles/fast_grains.py (free flight, inelastic impact
with normal and tangential restitution, resting contact, lift-off when the
surface falls away faster than g).  Two things differ:

* the surface is the analytic mode shape, not the wave solver, so the antinode
  acceleration Gamma0 = A w^2 / g is set exactly;
* a grain in contact can roll.  It feels the apparent gravity g + h_tt, is
  driven down the local slope with the solid-sphere factor 5/7, and is opposed
  by Coulomb rolling resistance mu_r.  With roll=False a grain in contact is
  held at its landing point (the infinite-friction limit).

Optional roughness, as in particles/grains.py: each impact adds a random
sideways velocity of size rough * (normal impact speed).

Grains do not interact, so one run of N grains is N independent trials.
"""
import numpy as np
from numba import njit

G = 9.81
ROLL = 5.0 / 7.0


@njit(cache=True)
def run(gamma0, mu_r, roll, N, n_cycles, steps, L, n_mode, f, e_n, e_t, stick, rough, n_groups, seed):
    """Returns (stats, x).  stats[c, j] = (mean node distance, fraction within
    20% of the node spacing of a node, same for an antinode) for subgroup j at
    the end of drive cycle c."""
    np.random.seed(seed)
    w = 2.0 * np.pi * f
    k = n_mode * np.pi / L
    A = gamma0 * G / w**2
    dt = 1.0 / (f * steps)
    sp = L / n_mode                       # node spacing; antinodes are sp/2 from a node
    v_stick = stick * G / w
    x = np.random.uniform(0.0, L, N)
    y = np.zeros(N); vx = np.zeros(N); vy = np.zeros(N)
    on = np.ones(N, dtype=np.bool_)
    stats = np.zeros((n_cycles, n_groups, 3))
    t = 0.0
    for c in range(n_cycles):
        for s in range(steps):
            t += dt
            st, ct = np.sin(w * t), np.cos(w * t)
            for i in range(N):
                sx = np.sin(k * x[i])
                if on[i]:
                    geff = G - A * w * w * sx * st            # apparent gravity
                    cx = np.cos(k * x[i])
                    if geff <= 0.0:                           # lift-off
                        on[i] = False
                        y[i] = A * sx * st
                        vy[i] = A * w * sx * ct + vx[i] * A * k * cx * st
                        continue
                    if not roll:
                        continue
                    ad = -ROLL * geff * A * k * cx * st       # drive down the slope
                    af = ROLL * mu_r * geff                   # rolling resistance
                    v0 = vx[i]
                    if v0 == 0.0:
                        if abs(ad) <= af:
                            continue
                        v1 = (ad - af * np.sign(ad)) * dt
                    else:
                        v1 = v0 + (ad - af * np.sign(v0)) * dt
                        if v1 * v0 < 0.0 and abs(ad) <= af:
                            v1 = 0.0                          # resistance brings it to rest
                    vx[i] = v1
                    x[i] += v1 * dt
                    if x[i] < 0.0:
                        x[i] = -x[i]; vx[i] = -vx[i]
                    elif x[i] > L:
                        x[i] = 2 * L - x[i]; vx[i] = -vx[i]
                    continue
                vy[i] -= G * dt
                x[i] += vx[i] * dt
                y[i] += vy[i] * dt
                if x[i] < 0.0:
                    x[i] = -x[i]; vx[i] = -vx[i]
                elif x[i] > L:
                    x[i] = 2 * L - x[i]; vx[i] = -vx[i]
                sx = np.sin(k * x[i])
                h = A * sx * st
                if y[i] <= h:
                    hx = A * k * np.cos(k * x[i]) * st
                    ht = A * w * sx * ct
                    nrm = np.sqrt(1.0 + hx * hx)
                    nx, ny = -hx / nrm, 1.0 / nrm
                    ux, uy = vx[i], vy[i] - ht
                    un = ux * nx + uy * ny
                    if un < 0.0:
                        utx, uty = ux - un * nx, uy - un * ny
                        un2 = -e_n * un
                        kick = 0.0
                        if rough > 0.0:                       # irregular grain / rough surface
                            kick = rough * abs(un) * np.random.standard_normal()
                        vx[i] = e_t * utx + un2 * nx + kick
                        vy[i] = e_t * uty + un2 * ny + ht
                        y[i] = h
                        if un2 < v_stick and G - A * w * w * sx * st > 0.0:
                            on[i] = True
                            vx[i] = e_t * utx + kick if roll else 0.0
        for i in range(N):
            d = abs((x[i] + sp / 2) % sp - sp / 2)
            j = i % n_groups
            stats[c, j, 0] += d
            if d < 0.1 * sp:
                stats[c, j, 1] += 1.0
            if d > 0.4 * sp:
                stats[c, j, 2] += 1.0
        stats[c] /= (N / n_groups)
    return stats, x
