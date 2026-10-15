"""Drives that were tried and do not work.  Kept for reference and for
figures that show the failure; nothing else depends on them.

    FixedDrive        the frequency never changes.  The resonance moves away
                      and the mode decays (thesis Fig. 6.2).
    NaiveDrive        direct substitution, cos(w(t) * t).  The phase is
                      w(t) * t, so the instantaneous frequency is its
                      derivative, w + t * dw/dt, not w.  With the smooth
                      tension ramps used here the drive is therefore detuned
                      by t * dw/dt (about 20 linewidths at the start of the
                      default string sweep, which begins at t = 1000), and
                      the error depends on where t = 0 was placed.  If w is
                      instead updated in steps, each step jumps the phase by
                      t * (change in w).  Either way the mode collapses, and
                      regrows once the tension stops changing.
    SelfExcitedDrive  force in phase with the measured velocity.  Nothing
                      selects the mode, so the loop moves to the mode it
                      couples to most strongly (mode 1 for uniform forcing).
"""
import numpy as np

from .drive import Drive


class FixedDrive(Drive):
    def __init__(self, w0):
        self.w0 = self.w = float(w0)

    def stage_values(self, t, dt, state=None):
        w = self.w0
        self.phase = w * (t + dt)
        return np.cos(w * t), np.cos(w * (t + 0.5 * dt)), np.cos(w * (t + dt))


class NaiveDrive(Drive):
    def __init__(self, w_of_t):
        self.w_of_t = w_of_t

    def stage_values(self, t, dt, state=None):
        th, t1 = t + 0.5 * dt, t + dt
        p0, self.phase = self.w_of_t(t) * t, self.w_of_t(t1) * t1
        self.w = (self.phase - p0) / dt          # instantaneous frequency, w + t dw/dt
        return np.cos(p0), np.cos(self.w_of_t(th) * th), np.cos(self.phase)


class SelfExcitedDrive(Drive):
    def __init__(self, w0, sensor, tau_frequency=20.0):
        self.w = float(w0)
        self.sensor = sensor
        self.tau = float(tau_frequency)
        self.phase = 0.0
        self._theta = None

    def stage_values(self, t, dt, state=None):
        s_u, s_v = self.sensor.read(state)
        # response s_u = R cos(theta); the force leads it by 90 degrees
        theta = float(np.arctan2(-s_v / self.w, s_u))
        if self._theta is not None:
            step = (theta - self._theta + np.pi) % (2.0 * np.pi) - np.pi
            w_inst = step / dt
            self.w += min(dt / self.tau, 1.0) * (np.clip(w_inst, 0.2 * self.w, 5.0 * self.w) - self.w)
            self.phase += step
        else:
            self.phase = theta + 0.5 * np.pi
        self._theta = theta
        p = theta + 0.5 * np.pi
        return np.cos(p), np.cos(p + 0.5 * self.w * dt), np.cos(p + self.w * dt)
