"""Tension schedules: how the slope ``a`` changes in time.

Each factory returns a function ``a(t)``.  The simulation only ever calls
that function, so new schedules can be added here without touching anything
else.
"""
import numpy as np


def constant(a0):
    return lambda t: a0


def step(a0, a1, t_start):
    """Jump from a0 to a1 at t_start."""
    return lambda t: a1 if t >= t_start else a0


def linear_ramp(a0, a1, t_start, rate):
    """Constant da/dt = rate from t_start until a1 is reached."""
    sign = 1.0 if a1 >= a0 else -1.0
    duration = abs(a1 - a0) / rate

    def a(t):
        s = min(max(t - t_start, 0.0), duration)
        return a0 + sign * rate * s
    return a


def smooth_ramp(a0, a1, t_start, duration):
    """S-shaped ramp (zero rate at both ends), a0 -> a1 over ``duration``."""
    def a(t):
        s = min(max((t - t_start) / duration, 0.0), 1.0)
        return a0 + (a1 - a0) * (3.0 * s * s - 2.0 * s ** 3)
    return a


def constant_frequency_rate(model, mode, a0, a1, t_start, duration, samples=400):
    """Ramp a0 -> a1 so that the resonant frequency changes at a constant rate.

    The frequency-vs-a curve is steep near the extreme slopes and flat near
    a = 0, so a constant rate in ``a`` moves the resonance unevenly.  This
    schedule spends longer where the resonance is most sensitive.
    """
    a_grid = np.linspace(a0, a1, samples)
    w_grid = np.array([model.resonance(a, mode) for a in a_grid])
    if not (np.all(np.diff(w_grid) > 0) or np.all(np.diff(w_grid) < 0)):
        raise ValueError("resonance is not monotonic between a0 and a1; "
                         "split the sweep at the turning point")
    w0, w1 = w_grid[0], w_grid[-1]
    order = np.argsort(w_grid)

    def a(t):
        s = min(max((t - t_start) / duration, 0.0), 1.0)
        return float(np.interp(w0 + (w1 - w0) * s, w_grid[order], a_grid[order]))
    return a
