"""Drives: how the forcing follows the moving resonance.

A drive supplies the time factor of the forcing, f(t) = F0 * shape * c(t).
The simulation asks for c at the three Runge-Kutta stage times of each step
and for the drive phase (the reference for demodulation).

    PLLDrive    the working drive.  A phase-locked loop: an oscillator whose
                frequency is steered until the response lags the force by
                90 degrees.  Needs a sensor and a starting frequency, no
                table of the resonance.
    TableDrive  the baseline.  The frequency is looked up from the computed
                resonance w_res(a(t)) and integrated so the phase is
                continuous.  Exact when the system is known exactly.

Both accept an amplitude gain (see compensation.py); PLLDrive can also hold
the amplitude with its own control loop.

Approaches that were tried and failed are kept in legacy_drives.py.
"""
import numpy as np


class Drive:
    """Base class.  ``phase`` is the drive phase at the current time, ``w``
    the current drive frequency and ``gain`` the current amplitude factor."""

    phase = 0.0
    w = float("nan")
    gain = 1.0

    def stage_values(self, t, dt, state=None):
        """Return (c(t), c(t + dt/2), c(t + dt)) and advance to t + dt.

        ``state`` is the simulation (drives with feedback read u and v)."""
        raise NotImplementedError

    # numeric state, so a settled run can be continued under a new schedule
    def get_state(self):
        keep = (int, float, bool, type(None), np.floating, np.integer)
        return {k: v for k, v in self.__dict__.items() if isinstance(v, keep)}

    def set_state(self, state):
        self.__dict__.update(state)


class TableDrive(Drive):
    """Baseline: phase-continuous tracking of a known resonance.

    w_of_t    : drive frequency as a function of time, normally
                ``lambda t: w_res(a(t))`` from the model's resonance table
    gain_of_t : optional amplitude factor g(t)
    """

    def __init__(self, w_of_t, phase0=0.0, gain_of_t=None):
        self.w_of_t = w_of_t
        self.phase = float(phase0)
        self.gain_of_t = gain_of_t
        self.w = float(w_of_t(0.0))

    def stage_values(self, t, dt, state=None):
        wa, wb, wc = self.w_of_t(t), self.w_of_t(t + 0.5 * dt), self.w_of_t(t + dt)
        p0 = self.phase
        ph = p0 + 0.25 * dt * (wa + wb)                 # trapezoid to the midpoint
        p1 = p0 + dt * (wa + 4.0 * wb + wc) / 6.0       # Simpson over the step
        self.phase = p1
        self.w = wc
        c = [np.cos(p0), np.cos(ph), np.cos(p1)]
        if self.gain_of_t is not None:
            g = self.gain_of_t
            self.gain = g(t + dt)
            c = [c[0] * g(t), c[1] * g(t + 0.5 * dt), c[2] * self.gain]
        return tuple(c)


class Sensor:
    """Weighted displacement and velocity pickup, s = sum(field * weights)."""

    def __init__(self, weights, cell):
        self.weights = np.asarray(weights, dtype=float)
        self.cell = float(cell)

    def read(self, state):
        return (float(np.sum(state.u * self.weights)) * self.cell,
                float(np.sum(state.v * self.weights)) * self.cell)


class PLLDrive(Drive):
    """Phase-locked loop drive.

    The force is F0 * shape * gain * cos(phi).  A sensor reads displacement
    s_u and velocity s_v; with those two signals the response
    s_u = R cos(phi - delta) is resolved exactly, with no averaging:

        I = s_u cos(phi) - (s_v / w) sin(phi) = R cos(delta)
        Q = s_u sin(phi) + (s_v / w) cos(phi) = R sin(delta)

    At resonance the lag delta is 90 degrees, so I = 0.  The error
    e = I / R - e_set steers the oscillator with a PI law:

        w = w_int + kp * e,      d(w_int)/dt = ki * e

    Near resonance e = (2 / beta) * (w_res - w), and the resonator's phase
    answers a frequency change with the time constant 2 / beta.  That makes
    the loop second order with natural frequency sqrt(ki); the gains are set
    from one number, the loop ``bandwidth`` wn, for critical damping:

        ki = wn^2,     kp = 2 wn - beta / 2

    While the resonance moves at rate r = d(w_res)/dt the loop runs with a
    standing error e = r / wn^2, a detuning of e * beta / 2.

    Starting
    --------
    Close the loop on a steady state near the resonance (for example
    ``Simulation.start_in_steady_state``).  With ``calibrate=True`` the first
    reading sets e_set, which removes the constant phase offset that the
    off-resonant modes add to the sensor signal.

    Amplitude
    ---------
    gain_of_t : feed-forward gain g(t), for example from
                ``compensation.compensation_gain``
    agc       : automatic amplitude control.  A second, slow PI loop adjusts
                the gain so the measured amplitude stays at the value it had
                at ``agc_start``.  ``agc_measure`` selects what is held:
                "peak"          largest displacement on the surface,
                                max sqrt(u^2 + (v/w)^2)
                "acceleration"  that value times w^2
                Keep ``agc_bandwidth`` at about 0.01: faster settings made
                the two loops hunt.
    """

    def __init__(self, w0, beta, sensor, bandwidth=0.1, tau_filter=2.0, calibrate=True,
                 lock_range=0.5, gain_of_t=None, agc=False, agc_start=0.0,
                 agc_bandwidth=0.01, agc_measure="peak", phase0=0.0):
        self.w = self.w_int = float(w0)
        self.beta = float(beta)
        self.ki = bandwidth ** 2
        self.kp = max(2.0 * bandwidth - 0.5 * beta, 0.0)
        self.tau = float(tau_filter)
        self.w_min, self.w_max = (1 - lock_range) * w0, (1 + lock_range) * w0
        self.sensor = sensor
        self.phase = float(phase0)
        self.calibrate, self.e_set = calibrate, 0.0
        self.I = self.Q = None
        self.error = 0.0
        self.R = 0.0
        self.gain_of_t = gain_of_t
        self.gain = self.gain_agc = self.gain_int = 1.0
        self.agc, self.agc_start, self.agc_measure = agc, float(agc_start), agc_measure
        tau_r = 2.0 / beta
        self.agc_ki = tau_r * agc_bandwidth ** 2
        self.agc_kp = max(2.0 * tau_r * agc_bandwidth - 1.0, 0.0)
        self.amp = None
        self.amp_set = None

    def stage_values(self, t, dt, state=None):
        s_u, s_v = self.sensor.read(state)
        cp, sp = np.cos(self.phase), np.sin(self.phase)
        I = s_u * cp - (s_v / self.w) * sp
        Q = s_u * sp + (s_v / self.w) * cp
        k = min(dt / self.tau, 1.0)
        if self.I is None:                       # first reading
            self.I, self.Q = I, Q
            if self.calibrate:
                self.e_set = I / max(np.hypot(I, Q), 1e-300)
        else:
            self.I += k * (I - self.I)
            self.Q += k * (Q - self.Q)
        self.R = float(np.hypot(self.I, self.Q))
        self.error = self.I / max(self.R, 1e-300) - self.e_set

        # frequency loop
        self.w_int = float(np.clip(self.w_int + self.ki * self.error * dt, self.w_min, self.w_max))
        self.w = float(np.clip(self.w_int + self.kp * self.error, self.w_min, self.w_max))

        # amplitude loop
        if self.agc:
            amp = float(np.sqrt(np.max(state.u ** 2 + (state.v / self.w) ** 2)))
            if self.agc_measure == "acceleration":
                amp *= self.w ** 2
            self.amp = amp if self.amp is None else self.amp + k * (amp - self.amp)
            if t >= self.agc_start:
                if self.amp_set is None:
                    self.amp_set = self.amp
                ea = 1.0 - self.amp / self.amp_set
                self.gain_int = float(np.clip(self.gain_int + self.agc_ki * ea * dt, 0.0, 20.0))
                self.gain_agc = float(np.clip(self.gain_int + self.agc_kp * ea, 0.0, 20.0))

        g0 = self.gain
        ff = self.gain_of_t(t + dt) if self.gain_of_t is not None else 1.0
        self.gain = ff * self.gain_agc
        p0 = self.phase
        self.phase = p0 + self.w * dt
        return (g0 * np.cos(p0), 0.5 * (g0 + self.gain) * np.cos(p0 + 0.5 * self.w * dt),
                self.gain * np.cos(self.phase))
