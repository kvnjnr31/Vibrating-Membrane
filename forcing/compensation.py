"""Feed-forward amplitude compensation.

On resonance the target mode responds to the force F0 * shape with modal
amplitude

    q(a) = mu(a) * F0 / (beta * sqrt(lam(a) - beta^2 / 4)),

where mu is the overlap of the force shape with the mode.  Both mu and lam
change with the tension slope a, so the amplitude drifts during a sweep even
when the frequency is tracked perfectly.  Multiplying the force by

    gain(a) = response(a0) / response(a)

cancels that drift when the sweep is slow.

Rate correction
---------------
While the tension is changing, the envelope A of the mode obeys

    dA/dt = -(beta / 2) A - (w' / (2 w)) A + mu F0 gain / (2 w)

The middle term is the adiabatic change of amplitude with frequency (the
w^(-1/2) law of thesis Ch. 6): a rising frequency lowers the amplitude even
with perfect tracking.  To hold X = A * m(a) constant (m = 1 for the modal
amplitude, m = peak-per-modal for the peak displacement) the gain needs the
extra factor

    1 + ( dln(w)/dt - 2 dln(m)/dt ) / beta

which ``compensation_gain`` applies by default.  It matters once the sweep
changes w or the mode shape by a few percent within the ring-down time
2 / beta.
"""
import numpy as np


def response(model, a, mode, shape, quantity="peak"):
    """Steady-state response of the target mode per unit F0, on resonance.

    quantity : "modal"  modal amplitude q of the target mode
               "peak"   largest displacement anywhere, from the full
                        steady state (target mode plus the off-resonant
                        response of the other modes)
               "acceleration"  that peak times w^2 (what decides whether
                        grains lift off)
               "energy" square root of the mode's energy (so that the gain
                        holds the energy constant)
    """
    lam = model.eigenvalue(a, mode)
    phi = model.mode_shape(a, mode)
    mu = model.inner(phi, shape)
    q = abs(mu) / (model.beta * np.sqrt(lam - 0.25 * model.beta ** 2))
    if quantity == "modal":
        return q
    if quantity in ("peak", "acceleration"):
        w = np.sqrt(lam - 0.5 * model.beta ** 2)
        lams, phis = model.modal_basis(a)
        mus = np.tensordot(phis, shape, axes=shape.ndim) * model.weight
        amp = mus / np.sqrt((lams - w * w) ** 2 + (model.beta * w) ** 2)
        delta = np.arctan2(model.beta * w, lams - w * w)
        c = np.tensordot(amp * np.cos(delta), phis, axes=1)
        s = np.tensordot(amp * np.sin(delta), phis, axes=1)
        peak = float(np.sqrt(np.max(c * c + s * s)))
        return peak * w * w if quantity == "acceleration" else peak
    if quantity == "energy":
        w2 = lam - 0.5 * model.beta ** 2
        return q * np.sqrt(0.25 * (w2 + lam))
    raise ValueError(f"unknown quantity '{quantity}'")


def response_table(model, mode, shape, quantity="peak", a_min=None, a_max=None, samples=121):
    lim = 0.97 * model.a_limit
    a_grid = np.linspace(-lim if a_min is None else a_min, lim if a_max is None else a_max, samples)
    r_grid = np.array([response(model, a, mode, shape, quantity) for a in a_grid])
    return lambda a: float(np.interp(a, a_grid, r_grid))


def compensation_gain(model, mode, schedule, shape, quantity="peak", rate_correction=True,
                      samples=121):
    """gain(t) that holds the chosen quantity at its value at t = 0."""
    lim = 0.97 * model.a_limit
    a_grid = np.linspace(-lim, lim, samples)
    r_x = np.array([response(model, a, mode, shape, quantity) for a in a_grid])
    r_modal = np.array([response(model, a, mode, shape, "modal") for a in a_grid])
    w = np.array([model.resonance(a, mode) for a in a_grid])
    # d/da of ln(w) - 2 ln(m), with m = r_x / r_modal
    slope = np.gradient(np.log(w) - 2.0 * np.log(r_x / r_modal), a_grid)
    r0 = float(np.interp(schedule(0.0), a_grid, r_x))
    h = 0.5

    def gain(t):
        a = schedule(t)
        g = r0 / float(np.interp(a, a_grid, r_x))
        if rate_correction:
            a_dot = (schedule(t + h) - schedule(t - h)) / (2.0 * h)
            g *= 1.0 + a_dot * float(np.interp(a, a_grid, slope)) / model.beta
        return g
    return gain
