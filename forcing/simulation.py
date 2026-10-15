"""Time integration and diagnostics shared by the string and the membrane.

``Simulation`` advances a model with 4th-order Runge-Kutta under a drive and
a tension schedule, and measures once per drive cycle:

    energy    total mechanical energy (cycle average)
    peak      largest displacement anywhere on the surface
    profile   the demodulated standing-wave shape (what is oscillating at
              the drive frequency)
    purity    fraction of that shape that is the target mode at the current
              tension, 0..1
    nodes     node positions of the profile, and where the target mode's
              nodes should be

Smooth node transport means: energy stays on the steady-state curve, purity
stays near 1, and measured nodes follow the expected ones.
"""
import numpy as np


class Simulation:
    def __init__(self, model, drive, schedule, mode, F0=1.0, shape=None, dt=0.02):
        self.model, self.drive, self.schedule, self.mode = model, drive, schedule, mode
        self.F0 = float(F0)
        self.shape = model.uniform_shape() if shape is None else shape
        self.dt = float(dt)
        self.t = 0.0
        self.u = np.zeros(model.shape)
        self.v = np.zeros(model.shape)
        self.a = schedule(0.0)
        # demodulation accumulators (one drive cycle)
        self._C = np.zeros(model.shape)
        self._S = np.zeros(model.shape)
        self._E = 0.0
        self._nE = 0
        self._n = 0
        self._a_sum = 0.0
        self.energy_every = 4                 # sample the energy every few steps
        self._phase_mark = drive.phase
        self.history = []                     # one dict per completed cycle
        self.keep_profiles = False

    # ---------------------------------------------------------------- stepping
    def step(self):
        m, dt, t = self.model, self.dt, self.t
        u, v = self.u, self.v
        a1, a2, a4 = self.schedule(t), self.schedule(t + 0.5 * dt), self.schedule(t + dt)
        c1, c2, c4 = self.drive.stage_values(t, dt, self)
        F = self.F0 * self.shape

        k1u, k1v = v, m.accel(u, v, a1, F * c1)
        k2u = v + 0.5 * dt * k1v
        k2v = m.accel(u + 0.5 * dt * k1u, k2u, a2, F * c2)
        k3u = v + 0.5 * dt * k2v
        k3v = m.accel(u + 0.5 * dt * k2u, k3u, a2, F * c2)
        k4u = v + dt * k3v
        k4v = m.accel(u + dt * k3u, k4u, a4, F * c4)
        self.u = u + dt / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        self.v = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        self.t = t + dt
        self.a = a4
        self._accumulate()

    def _accumulate(self):
        ph = self.drive.phase
        self._C += self.u * np.cos(ph)
        self._S += self.u * np.sin(ph)
        if self._n % self.energy_every == 0:
            self._E += self.model.energy(self.u, self.v, self.a)
            self._nE += 1
        self._a_sum += self.a
        self._n += 1
        if abs(ph - self._phase_mark) >= 2.0 * np.pi:
            self._close_cycle()
            self._phase_mark = ph

    def _close_cycle(self):
        m, n = self.model, self._n
        C, S = 2.0 * self._C / n, 2.0 * self._S / n       # u ~ C cos(ph) + S sin(ph)
        # rotate to the phase that carries the standing wave
        theta = 0.5 * np.arctan2(2.0 * np.sum(C * S), np.sum(C * C) - np.sum(S * S))
        profile = C * np.cos(theta) + S * np.sin(theta)
        # the profile is an average over the cycle, so compare it with the
        # target mode at the cycle's mean tension, not the tension at its end
        a_mid = self._a_sum / n
        phi = m.mode_shape(a_mid, self.mode)
        overlap = m.inner(profile, phi)
        norm2 = m.inner(profile, profile)
        if overlap < 0:
            profile = -profile
        rec = dict(t=self.t, a=self.a, a_mid=a_mid, energy=self._E / max(self._nE, 1),
                   amplitude=float(np.max(np.abs(profile))),
                   peak=float(np.sqrt(np.max(C * C + S * S))),
                   modal_amplitude=abs(overlap),
                   purity=(overlap * overlap / norm2) if norm2 > 0 else 0.0,
                   nodes=m.nodes(profile, self.mode),
                   expected_nodes=m.nodes(phi, self.mode),
                   w=float(getattr(self.drive, "w", np.nan)),
                   gain=float(getattr(self.drive, "gain", 1.0)))
        if self.keep_profiles:
            rec["profile"] = profile.copy()
        self.history.append(rec)
        self._C[:] = 0.0
        self._S[:] = 0.0
        self._E = 0.0
        self._nE = 0
        self._n = 0
        self._a_sum = 0.0

    def run(self, t_end, callback=None):
        nsteps = int(round((t_end - self.t) / self.dt))
        for _ in range(nsteps):
            self.step()
            if callback is not None:
                callback(self)
        return self

    def start_in_steady_state(self, w=None):
        """Set u, v to the steady state for the drive F0 * shape * cos(w t) at
        the current tension, so no settling time is needed.

        Built from the modal sum; a short run afterwards removes what the
        truncated sum leaves out.  Assumes the drive phase is 0 at t = 0."""
        m = self.model
        w = m.resonance(self.a, self.mode) if w is None else w
        lam, phi = m.modal_basis(self.a)
        mu = np.tensordot(phi, self.shape, axes=self.shape.ndim) * m.weight
        amp = mu * self.F0 / np.sqrt((lam - w * w) ** 2 + (m.beta * w) ** 2)
        delta = np.arctan2(m.beta * w, lam - w * w)
        self.u = np.tensordot(amp * np.cos(delta), phi, axes=1)
        self.v = np.tensordot(amp * w * np.sin(delta), phi, axes=1)
        return self

    def rescale(self, factor):
        """Scale the state and the forcing together (the system is linear)."""
        self.u *= factor
        self.v *= factor
        self.F0 *= factor

    def continue_from(self, other):
        """Take over the state of a settled simulation (same model and drive
        type), so one settling run can be followed by several sweeps."""
        self.u, self.v, self.t = other.u.copy(), other.v.copy(), other.t
        self.F0 = other.F0
        self.drive.set_state(other.drive.get_state())
        self._phase_mark = self.drive.phase
        return self

    # ---------------------------------------------------------------- results
    def series(self, key):
        return np.array([h[key] for h in self.history])

    def node_tracks(self):
        """Measured and expected node positions as arrays (cycles, n_nodes).

        Cycles where the measured profile does not have the expected number
        of nodes are filled with NaN."""
        n_exp = len(self.history[-1]["expected_nodes"])
        meas = np.full((len(self.history), n_exp), np.nan)
        exp = np.full((len(self.history), n_exp), np.nan)
        for i, h in enumerate(self.history):
            if len(h["expected_nodes"]) == n_exp:
                exp[i] = h["expected_nodes"]
            if len(h["nodes"]) == n_exp:
                meas[i] = h["nodes"]
        return meas, exp

    def summary(self, t_from=0.0):
        """Headline numbers over the cycles after t_from."""
        t = self.series("t")
        sel = t >= t_from
        meas, exp = self.node_tracks()
        err = np.abs(meas - exp)[sel]
        lost = float(np.mean(np.isnan(meas[sel]).any(axis=1)))
        return dict(final_energy=float(self.series("energy")[-1]),
                    min_energy=float(self.series("energy")[sel].min()),
                    min_purity=float(self.series("purity")[sel].min()),
                    max_node_error=float(np.nanmax(err)) if np.isfinite(err).any() else float("nan"),
                    fraction_cycles_nodes_lost=lost)


def steady_state_energy(model, a, w, F0=1.0, shape=None):
    """Cycle-averaged energy of the steady state at slope a, drive frequency w.

    Modal sum: each mode j responds with amplitude
    mu_j F0 / sqrt((lam_j - w^2)^2 + beta^2 w^2) and carries energy
    (w^2 + lam_j) |q_j|^2 / 4.  This is the curve a perfectly smooth sweep
    would follow.
    """
    shape = model.uniform_shape() if shape is None else shape
    lam, phi = model.modal_basis(a)
    mu = np.tensordot(phi, shape, axes=shape.ndim) * model.weight
    q2 = (mu * F0) ** 2 / ((lam - w * w) ** 2 + (model.beta * w) ** 2)
    return float(np.sum(0.25 * (w * w + lam) * q2))


DRIVES = ("pll", "pll+gain", "pll+agc", "table", "table+gain")
LEGACY_DRIVES = ("fixed", "naive", "self")


def make_drive(kind, model, mode, schedule, shape=None, quantity="peak", **kw):
    """Build a drive by name.

    Working drive (no resonance table):
        pll          phase-locked loop
        pll+gain     with the feed-forward amplitude gain (amplitude table)
        pll+agc      with its own amplitude control loop (no table at all)
    Baseline (resonance looked up from the computed table):
        table        phase-continuous tracking
        table+gain   with the feed-forward amplitude gain
    Failed approaches, see legacy_drives.py:
        fixed, naive, self

    quantity : what the gain or the amplitude loop holds constant:
               "peak" displacement, peak "acceleration", and for the
               feed-forward gain also "modal" or "energy"
    Other keyword arguments go to the drive (``bandwidth``, ``w0``,
    ``agc_start``, ``rate_correction``, ...).
    """
    from .drive import TableDrive, PLLDrive, Sensor
    from .legacy_drives import FixedDrive, NaiveDrive, SelfExcitedDrive
    from .compensation import compensation_gain
    w_res = model.resonance_table(mode)
    w_of_t = lambda t: w_res(schedule(t))               # noqa: E731
    w_start = w_res(schedule(0.0))
    shape = model.uniform_shape() if shape is None else shape
    rate_correction = kw.pop("rate_correction", True)

    def feed_forward():
        return compensation_gain(model, mode, schedule, shape, quantity,
                                 rate_correction=rate_correction)

    if kind == "table":
        return TableDrive(w_of_t)
    if kind == "table+gain":
        return TableDrive(w_of_t, gain_of_t=feed_forward())
    if kind in ("pll", "pll+gain", "pll+agc"):
        gain = feed_forward() if kind == "pll+gain" else None
        return PLLDrive(kw.pop("w0", w_start), model.beta, Sensor(shape, model.weight),
                        gain_of_t=gain, agc=(kind == "pll+agc"), agc_measure=quantity, **kw)
    if kind == "fixed":
        return FixedDrive(w_start)
    if kind == "naive":
        return NaiveDrive(w_of_t)
    if kind == "self":
        return SelfExcitedDrive(kw.pop("w0", w_start), Sensor(shape, model.weight))
    raise ValueError(f"unknown drive '{kind}'")
