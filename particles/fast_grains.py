"""Compiled version of Grains.step for grains on a string (one coordinate).

Same rules as particles/grains.py: free flight, reflecting side walls,
inelastic impact with a random sideways kick, and resting contact with
viscous friction.  Without roughness it reproduces Grains.step exactly;
with roughness the random numbers are drawn in a different order, so runs
agree in their statistics only.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _step(X, V, z, vz, on, u, v, ux, x0, dx, lo, hi, g, dt, e_n, e_t, fr, rest, rough, kicks):
    n = X.shape[0]
    m = u.shape[0]
    damp = np.exp(-fr * dt)
    impacts = 0
    for i in range(n):
        vz[i] -= g * dt
        x = X[i] + V[i] * dt
        z[i] += vz[i] * dt
        if x < lo:
            x = 2 * lo - x
            V[i] *= -e_n
        elif x > hi:
            x = 2 * hi - x
            V[i] *= -e_n
        X[i] = x
        s = (x - x0) / dx
        j = int(s)
        if j < 0:
            j = 0
        if j > m - 2:
            j = m - 2
        f = s - j
        h = u[j] + f * (u[j + 1] - u[j])
        w = v[j] + f * (v[j + 1] - v[j])
        gr = ux[j] + f * (ux[j + 1] - ux[j])
        hit = z[i] <= h
        on[i] = hit
        if not hit:
            continue
        norm = np.sqrt(1.0 + gr * gr)
        n_h = -gr / norm
        n_z = 1.0 / norm
        v_h = V[i]
        v_z = vz[i] - w
        vn = v_h * n_h + v_z * n_z
        vt_h = v_h - vn * n_h
        vt_z = v_z - vn * n_z
        impact = vn < 0.0 and -vn > rest * g * dt
        resting = vn < 0.0 and not impact
        vn_new = vn
        scale = 1.0
        if impact:
            vn_new = -e_n * vn
            scale = e_t
            impacts += 1
        elif resting:
            vn_new = 0.0
            scale = damp
        vt_h *= scale
        vt_z *= scale
        V_new = vt_h + vn_new * n_h
        if impact and rough > 0.0:
            V_new += rough * abs(vn) * kicks[i]
        V[i] = V_new
        vz[i] = w + vt_z + vn_new * n_z
        z[i] = h
    return impacts


class FastGrains:
    """Wraps a Grains object (string only) and steps it with compiled code."""

    def __init__(self, grains):
        g = grains
        self.g = g
        self.n = g.n
        self.x = np.ascontiguousarray(g.X[:, 0], dtype=np.float64)
        self.v = np.ascontiguousarray(g.V[:, 0], dtype=np.float64)
        self.z = np.ascontiguousarray(g.z, dtype=np.float64)
        self.vz = np.ascontiguousarray(g.vz, dtype=np.float64)
        self.on = np.ascontiguousarray(g.on_surface)
        self.rng = g.rng
        self._zeros = np.zeros(g.n)

    def step(self, dt, surface):
        g = self.g
        kicks = self.rng.standard_normal(self.n) if g.roughness > 0.0 else self._zeros
        g.impacts += _step(self.x, self.v, self.z, self.vz, self.on, surface.u, surface.v, surface.ux,
                           surface.x[0], surface.dx, g.lo[0], g.hi[0], g.g, dt, g.e_n, g.e_t,
                           g.friction_rate, g.rest_factor, g.roughness, kicks)
