"""Rough grains that roll, on the membrane of the wave solver.

The contact rules are those of acceleration_sweep/standing_wave_grains.py,
carried over to two horizontal coordinates and to a surface taken from the
wave solver:

* free flight under gravity;
* inelastic impact with normal and tangential restitution, plus a random
  sideways kick of size roughness * (normal impact speed) in x and in y;
* a grain whose rebound speed is below stick * g / w stays in contact;
* a grain in contact feels the apparent gravity g + u_tt, lifts off when
  that falls to zero, is driven down the local slope with the solid-sphere
  factor 5/7 and is opposed by rolling resistance mu_r (with roll=False it
  is held where it landed).

Grains do not interact and do not act back on the membrane.
"""
import numpy as np
from numba import njit

ROLL = 5.0 / 7.0


@njit(cache=True)
def _bil(F, i, j, s, t):
    return ((1 - s) * (1 - t) * F[i, j] + s * (1 - t) * F[i + 1, j]
            + (1 - s) * t * F[i, j + 1] + s * t * F[i + 1, j + 1])


@njit(cache=True)
def _step(x, y, vx, vy, z, vz, on, alive, u, v, acc, ux, uy, x0, y0, dx, dy, xlo, xhi, ylo, yhi,
          g, dt, e_n, e_t, v_stick, rough, mu_r, roll, kicks):
    nx_, ny_ = u.shape
    for n in range(x.shape[0]):
        if not alive[n]:
            continue
        fx = (x[n] - x0) / dx
        fy = (y[n] - y0) / dy
        i = min(max(int(fx), 0), nx_ - 2)
        j = min(max(int(fy), 0), ny_ - 2)
        s, t = fx - i, fy - j
        if on[n]:
            geff = g + _bil(acc, i, j, s, t)
            gx, gy = _bil(ux, i, j, s, t), _bil(uy, i, j, s, t)
            if geff <= 0.0:                                   # lift-off
                on[n] = False
                z[n] = _bil(u, i, j, s, t)
                vz[n] = _bil(v, i, j, s, t) + vx[n] * gx + vy[n] * gy
                continue
            z[n] = _bil(u, i, j, s, t)
            if not roll:
                continue
            ax, ay = -ROLL * geff * gx, -ROLL * geff * gy     # drive down the slope
            ad = np.sqrt(ax * ax + ay * ay)
            af = ROLL * mu_r * geff                           # rolling resistance
            sp = np.sqrt(vx[n] * vx[n] + vy[n] * vy[n])
            if sp == 0.0:
                if ad <= af:
                    continue
                v1x, v1y = (ax - af * ax / ad) * dt, (ay - af * ay / ad) * dt
            else:
                v1x = vx[n] + (ax - af * vx[n] / sp) * dt
                v1y = vy[n] + (ay - af * vy[n] / sp) * dt
                if v1x * vx[n] + v1y * vy[n] < 0.0 and ad <= af:
                    v1x, v1y = 0.0, 0.0                       # resistance brings it to rest
            vx[n], vy[n] = v1x, v1y
            x[n] += v1x * dt
            y[n] += v1y * dt
            if x[n] < xlo:
                x[n] = 2 * xlo - x[n]; vx[n] = -vx[n]
            elif x[n] > xhi:
                x[n] = 2 * xhi - x[n]; vx[n] = -vx[n]
            if y[n] < ylo:
                y[n] = 2 * ylo - y[n]; vy[n] = -vy[n]
            elif y[n] > yhi:
                y[n] = 2 * yhi - y[n]; vy[n] = -vy[n]
            continue
        # free flight
        vz[n] -= g * dt
        x[n] += vx[n] * dt
        y[n] += vy[n] * dt
        z[n] += vz[n] * dt
        if x[n] < xlo:
            x[n] = 2 * xlo - x[n]; vx[n] = -vx[n]
        elif x[n] > xhi:
            x[n] = 2 * xhi - x[n]; vx[n] = -vx[n]
        if y[n] < ylo:
            y[n] = 2 * ylo - y[n]; vy[n] = -vy[n]
        elif y[n] > yhi:
            y[n] = 2 * yhi - y[n]; vy[n] = -vy[n]
        fx = (x[n] - x0) / dx
        fy = (y[n] - y0) / dy
        i = min(max(int(fx), 0), nx_ - 2)
        j = min(max(int(fy), 0), ny_ - 2)
        s, t = fx - i, fy - j
        h = _bil(u, i, j, s, t)
        if z[n] <= h:
            gx, gy = _bil(ux, i, j, s, t), _bil(uy, i, j, s, t)
            ht = _bil(v, i, j, s, t)
            nrm = np.sqrt(1.0 + gx * gx + gy * gy)
            nx, ny, nz = -gx / nrm, -gy / nrm, 1.0 / nrm
            wx, wy, wz = vx[n], vy[n], vz[n] - ht
            un = wx * nx + wy * ny + wz * nz
            z[n] = h
            if un < 0.0:
                tx, ty, tz = wx - un * nx, wy - un * ny, wz - un * nz
                un2 = -e_n * un
                kx = rough * abs(un) * kicks[n, 0]
                ky = rough * abs(un) * kicks[n, 1]
                vx[n] = e_t * tx + un2 * nx + kx
                vy[n] = e_t * ty + un2 * ny + ky
                vz[n] = e_t * tz + un2 * nz + ht
                if un2 < v_stick and g + _bil(acc, i, j, s, t) > 0.0:
                    on[n] = True
                    if roll:
                        vx[n], vy[n] = e_t * tx + kx, e_t * ty + ky
                    else:
                        vx[n], vy[n] = 0.0, 0.0


class RollGrains:
    """State of n grains on a membrane model; step() advances them by dt."""

    def __init__(self, n, model, g, omega, restitution=0.3, tangential_restitution=0.8, stick=0.05,
                 roughness=0.3, mu_r=0.05, roll=True, wall_margin=0.01, seed=0):
        self.n, self.g = int(n), float(g)
        self.e_n, self.e_t = float(restitution), float(tangential_restitution)
        self.v_stick = float(stick) * self.g / float(omega)
        self.rough, self.mu_r, self.roll = float(roughness), float(mu_r), bool(roll)
        self.x0, self.y0, self.dx, self.dy = model.x[0], model.y[0], model.dx, model.dy
        self.xlo = model.x[0] + wall_margin * model.Lx
        self.xhi = model.x[-1] - wall_margin * model.Lx
        self.ylo = model.y[0] + wall_margin * model.Ly
        self.yhi = model.y[-1] - wall_margin * model.Ly
        self.rng = np.random.default_rng(seed)
        self.x = self.xlo + (self.xhi - self.xlo) * self.rng.random(self.n)
        self.y = self.ylo + (self.yhi - self.ylo) * self.rng.random(self.n)
        self.vx, self.vy = np.zeros(self.n), np.zeros(self.n)
        self.z, self.vz = np.zeros(self.n), np.zeros(self.n)
        self.on = np.ones(self.n, dtype=np.bool_)
        self.alive = np.ones(self.n, dtype=np.bool_)
        self._v_prev = None

    def step(self, sim):
        """Call after sim.step(); uses the membrane fields at the new time."""
        u, v, dt = sim.u, sim.v, sim.dt
        acc = np.zeros_like(v) if self._v_prev is None else (v - self._v_prev) / dt
        self._v_prev = v.copy()
        ux, uy = np.gradient(u, self.dx, self.dy)
        _step(self.x, self.y, self.vx, self.vy, self.z, self.vz, self.on, self.alive, u, v, acc, ux, uy,
              self.x0, self.y0, self.dx, self.dy, self.xlo, self.xhi, self.ylo, self.yhi, self.g, dt,
              self.e_n, self.e_t, self.v_stick, self.rough, self.mu_r, self.roll,
              self.rng.standard_normal((self.n, 2)))

    def place(self, idx, x, y, z, sim):
        """Put grains idx at (x, y) and height z above the membrane, at rest and in free flight."""
        self.x[idx], self.y[idx] = x, y
        self.vx[idx] = self.vy[idx] = self.vz[idx] = 0.0
        self.z[idx] = z
        self.on[idx] = False
        self.alive[idx] = True


def driven_membrane(model, mode, a_of_t, env_of_t, a_start, amplitude, gamma, dt, table=None, t_spin=50.0):
    """Membrane driven by the look-up table with the peak acceleration held.

    Returns (sim, g, omega): the simulation in its steady state at slope
    a_start, the gravity that gives a peak acceleration of gamma * g, and
    the resonance there.  a_of_t and env_of_t (drive envelope, 0 to 1) are
    functions of the time since the steady state was reached.
    """
    from forcing import schedules
    from forcing.compensation import response, compensation_gain
    from forcing.drive import TableDrive
    from forcing.simulation import Simulation, make_drive
    if table is not None:
        model.load_table(table)
    sched = schedules.constant(a_start)
    sim0 = Simulation(model, make_drive("table+gain", model, mode, sched, quantity="acceleration"), sched, mode, dt=dt)
    sim0.F0 = amplitude / response(model, a_start, mode, sim0.shape, "peak")
    sim0.start_in_steady_state().run(t_spin)
    w = model.resonance(a_start, mode)
    t0 = sim0.t
    a = lambda t: a_of_t(t - t0)                              # noqa: E731
    gain = compensation_gain(model, mode, a, sim0.shape, "acceleration", rate_correction=False)
    w_res = model.resonance_table(mode)
    drive = TableDrive(lambda t: w_res(a(t)), gain_of_t=lambda t: gain(t) * env_of_t(t - t0))
    sim = Simulation(model, drive, a, mode, dt=dt).continue_from(sim0)
    return sim, amplitude * w * w / gamma, w, t0
