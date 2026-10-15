"""A layer of inelastic disks on a flat, vertically shaking floor (x-z plane).

Experimental: separate from the string/membrane code.  Units: mean disk
diameter d = 1, gravity g = 1, so time is in sqrt(d/g).

Contacts are soft (linear spring and dashpot).  The spring and dashpot of
each contact are scaled with the reduced mass so that every collision
lasts ``t_c`` and has normal restitution ``e`` whatever the disk sizes.
Sliding friction is a viscous tangential force capped at mu times the
normal force, and it spins the disks.  The box is periodic in x.

Needs numba.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _forces(x, z, vx, vz, om, r, m, W, zf, vf, kfac, cfac, mu, cell, ncx, ncz, head, nxt,
            fx, fz, tq):
    n = x.size
    for i in range(n):
        fx[i] = 0.0
        fz[i] = -m[i]
        tq[i] = 0.0
    head[:] = -1
    for i in range(n):
        cx = int(x[i] / cell) % ncx
        cz = int((z[i] - zf) / cell) + 1
        cz = 0 if cz < 0 else (ncz - 1 if cz >= ncz else cz)
        c = cx * ncz + cz
        nxt[i] = head[c]
        head[c] = i
    for i in range(n):
        # floor
        d = r[i] - (z[i] - zf)
        if d > 0.0:
            vn = -(vz[i] - vf)
            fn = m[i] * (kfac * d + cfac * vn)
            if fn > 0.0:
                vt = vx[i] + om[i] * r[i]
                ft = cfac * m[i] * abs(vt)
                if ft > mu * fn:
                    ft = mu * fn
                if vt > 0.0:
                    ft = -ft
                fz[i] += fn
                fx[i] += ft
                tq[i] += r[i] * ft
        cx = int(x[i] / cell) % ncx
        cz = int((z[i] - zf) / cell) + 1
        cz = 0 if cz < 0 else (ncz - 1 if cz >= ncz else cz)
        for dx in (-1, 0, 1):
            c2x = (cx + dx) % ncx
            for dz in (-1, 0, 1):
                c2z = cz + dz
                if c2z < 0 or c2z >= ncz:
                    continue
                j = head[c2x * ncz + c2z]
                while j >= 0:
                    if j > i:
                        ex = x[j] - x[i]
                        if ex > 0.5 * W:
                            ex -= W
                        elif ex < -0.5 * W:
                            ex += W
                        ez = z[j] - z[i]
                        rr = r[i] + r[j]
                        d2 = ex * ex + ez * ez
                        if d2 < rr * rr:
                            dist = np.sqrt(d2)
                            nx_, nz_ = ex / dist, ez / dist
                            meff = m[i] * m[j] / (m[i] + m[j])
                            rvx, rvz = vx[i] - vx[j], vz[i] - vz[j]
                            vn = rvx * nx_ + rvz * nz_
                            fn = meff * (kfac * (rr - dist) + cfac * vn)
                            if fn > 0.0:
                                tx, tz = -nz_, nx_
                                vt = rvx * tx + rvz * tz + om[i] * r[i] + om[j] * r[j]
                                ft = cfac * meff * abs(vt)
                                if ft > mu * fn:
                                    ft = mu * fn
                                if vt > 0.0:
                                    ft = -ft
                                fx[i] += -fn * nx_ + ft * tx
                                fz[i] += -fn * nz_ + ft * tz
                                fx[j] += fn * nx_ - ft * tx
                                fz[j] += fn * nz_ - ft * tz
                                tq[i] += r[i] * ft
                                tq[j] += r[j] * ft
                    j = nxt[j]


@njit(cache=True)
def _run(x, z, vx, vz, om, r, m, W, t, dt, nsteps, every, A, w, kfac, cfac, mu, cell, ncx, ncz,
         frames_x, frames_z, frames_t):
    n = x.size
    head = np.empty(ncx * ncz, np.int64)
    nxt = np.empty(n, np.int64)
    fx, fz, tq = np.empty(n), np.empty(n), np.empty(n)
    k = 0
    for s in range(nsteps):
        zf, vf = A * np.sin(w * t), A * w * np.cos(w * t)
        if s % every == 0 and k < frames_t.size:
            frames_t[k] = t
            for i in range(n):
                frames_x[k, i] = x[i]
                frames_z[k, i] = z[i] - zf
            k += 1
        _forces(x, z, vx, vz, om, r, m, W, zf, vf, kfac, cfac, mu, cell, ncx, ncz, head, nxt,
                fx, fz, tq)
        for i in range(n):
            vx[i] += dt * fx[i] / m[i]
            vz[i] += dt * fz[i] / m[i]
            om[i] += dt * tq[i] / (0.5 * m[i] * r[i] * r[i])
            x[i] = (x[i] + dt * vx[i]) % W
            z[i] += dt * vz[i]
        t += dt
    return t, k


class Layer:
    """Disks on a floor at z_f(t) = A sin(w t).

    width      box width in disk diameters (periodic)
    depth      layer depth in disk diameters (sets the number of disks)
    frequency  drive frequency in sqrt(g/d)
    gamma      peak floor acceleration in units of g
    """

    def __init__(self, width=120.0, depth=8.0, frequency=0.14, gamma=3.5, restitution=0.6,
                 friction=0.5, t_c=0.05, spread=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.depth = float(width), float(depth)
        self.f, self.w = frequency, 2 * np.pi * frequency
        self.gamma, self.A = gamma, gamma / self.w ** 2
        self.e, self.mu, self.t_c = restitution, friction, t_c
        self.kfac = (np.pi ** 2 + np.log(restitution) ** 2) / t_c ** 2
        self.cfac = -2.0 * np.log(restitution) / t_c
        self.dt = t_c / 30.0
        nx, nz = int(width / 1.05), int(round(depth / 0.9))
        gx, gz = np.meshgrid((np.arange(nx) + 0.5) * width / nx, 0.6 + 1.05 * np.arange(nz))
        gx = gx + 0.5 * (width / nx) * (np.arange(nz)[:, None] % 2)
        self.x = (gx.ravel() + rng.normal(0, 0.02, gx.size)) % width
        self.z = gz.ravel().copy()
        self.n = self.x.size
        self.r = 0.5 * (1.0 + spread * rng.uniform(-1, 1, self.n))
        self.m = (2 * self.r) ** 2
        self.vx = rng.normal(0, 0.05, self.n)
        self.vz = rng.normal(0, 0.05, self.n)
        self.om = np.zeros(self.n)
        self.t = 0.0
        self.cell = 2 * self.r.max() * 1.001
        self.ncx = int(width / self.cell)
        self.cell_x = width / self.ncx
        self.ncz = int(40 * depth)

    @property
    def period(self):
        return 1.0 / self.f

    def run(self, periods, frames_per_period=16):
        """Advance by a whole number of drive periods.  Returns (t, x, h):
        frame times, disk x positions and disk heights above the floor."""
        every = int(round(self.period / frames_per_period / self.dt))
        dt = self.period / frames_per_period / every
        nsteps = every * frames_per_period * periods
        nfr = frames_per_period * periods
        fx = np.empty((nfr, self.n), np.float32)
        fz = np.empty((nfr, self.n), np.float32)
        ft = np.empty(nfr)
        self.t, k = _run(self.x, self.z, self.vx, self.vz, self.om, self.r, self.m, self.W, self.t,
                         dt, nsteps, every, self.A, self.w, self.kfac, self.cfac, self.mu,
                         self.cell_x, self.ncx, self.ncz, fx, fz, ft)
        return ft[:k], fx[:k], fz[:k]


def surface_profile(x, h, m, W, bins=60):
    """Mass-weighted mean height of the disks in each column, per frame."""
    idx = np.minimum((x / W * bins).astype(np.int64), bins - 1)
    out = np.empty((x.shape[0], bins))
    for k in range(x.shape[0]):
        mass = np.bincount(idx[k], weights=m, minlength=bins)
        out[k] = np.bincount(idx[k], weights=m * h[k], minlength=bins) / np.maximum(mass, 1e-12)
    return out
