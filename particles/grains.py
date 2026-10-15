"""Point grains bouncing on a moving surface (string: 1 horizontal
coordinate, membrane: 2).

Model
-----
* Between contacts a grain is in free flight under gravity g.
* A grain that reaches the surface collides with it inelastically.  In the
  frame of the surface point, the velocity component along the surface
  normal is reversed and multiplied by the restitution coefficient; the
  tangential component is multiplied by the tangential restitution.  A tilted
  surface therefore kicks the grain sideways.
* Optional roughness: each impact also adds a random sideways velocity of
  size roughness * (normal impact speed), standing in for irregular grains
  and a rough surface.
* A grain arriving with a small normal speed stays on the surface: it rides
  with it, slides under gravity along the slope with a friction rate, and
  lifts off again when the surface falls away faster than g.

What is NOT in the model yet
----------------------------
* The grains do not push back on the surface (one-way coupling).
* Grains do not collide with each other.
* Contacts are resolved at the end of the time step, not at the exact
  impact time.  The event-driven impact-time solve from the MATLAB code
  (matlab_ports/old_functions/wave_corr_old.py) is the natural upgrade.
* The surface moves only vertically (no in-plane motion of the string).
"""
import numpy as np


class Grains:
    def __init__(self, n, surface, g, restitution=0.4, tangential_restitution=0.8,
                 friction_rate=2.0, roughness=0.0, rest_factor=3.0, wall_margin=0.01, seed=0):
        self.n, self.g = int(n), float(g)
        self.e_n, self.e_t = float(restitution), float(tangential_restitution)
        self.friction_rate = float(friction_rate)
        self.rest_factor = float(rest_factor)
        self.roughness = float(roughness)
        self.lo = surface.lo + wall_margin * (surface.hi - surface.lo)
        self.hi = surface.hi - wall_margin * (surface.hi - surface.lo)
        self.rng = np.random.default_rng(seed)
        self.X = self.lo + (self.hi - self.lo) * self.rng.random((self.n, len(self.lo)))
        self.V = np.zeros_like(self.X)
        h, w, _ = surface.sample(self.X)
        self.z = h.copy()
        self.vz = w.copy()
        self.on_surface = np.ones(self.n, dtype=bool)
        self.impacts = 0

    def step(self, dt, surface):
        """Advance by dt; ``surface`` must already hold the fields at t + dt."""
        # free flight
        self.vz -= self.g * dt
        self.X += self.V * dt
        self.z += self.vz * dt

        # side walls: reflect
        for d in range(self.X.shape[1]):
            low = self.X[:, d] < self.lo[d]
            high = self.X[:, d] > self.hi[d]
            self.X[low, d] = 2 * self.lo[d] - self.X[low, d]
            self.X[high, d] = 2 * self.hi[d] - self.X[high, d]
            self.V[low | high, d] *= -self.e_n

        # contact with the surface
        h, w, grad = surface.sample(self.X)
        hit = self.z <= h
        self.on_surface = hit
        if not hit.any():
            return
        gr = grad[hit]
        norm = np.sqrt(1.0 + np.sum(gr * gr, axis=1))
        n_h = -gr / norm[:, None]                    # horizontal part of the unit normal
        n_z = 1.0 / norm                             # vertical part
        v_h = self.V[hit]                            # velocity relative to the surface point
        v_z = self.vz[hit] - w[hit]
        vn = np.sum(v_h * n_h, axis=1) + v_z * n_z
        vt_h = v_h - vn[:, None] * n_h
        vt_z = v_z - vn * n_z

        approaching = vn < 0.0
        impact = approaching & (-vn > self.rest_factor * self.g * dt)
        resting = approaching & ~impact
        vn_new = np.where(impact, -self.e_n * vn, np.where(resting, 0.0, vn))
        scale = np.where(impact, self.e_t, np.where(resting, np.exp(-self.friction_rate * dt), 1.0))
        vt_h = vt_h * scale[:, None]
        vt_z = vt_z * scale

        V_new = vt_h + vn_new[:, None] * n_h
        if self.roughness > 0.0:
            # irregular grains / rough surface: random sideways kick at each impact
            kick = self.roughness * np.abs(vn) * impact
            V_new = V_new + kick[:, None] * self.rng.standard_normal(V_new.shape)
        self.V[hit] = V_new
        self.vz[hit] = w[hit] + vt_z + vn_new * n_z
        self.z[hit] = h[hit]
        self.impacts += int(impact.sum())
