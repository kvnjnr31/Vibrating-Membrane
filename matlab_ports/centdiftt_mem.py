"""Right-hand side of the string ODE system  dy/dt = f(y, t).

Port of centdiftt_mem.m.

State layout (same rows as the MATLAB code, shifted to 0-based):
    u[0, :] = u_t  (velocity,      MATLAB u(1,:))
    u[1, :] = u    (displacement,  MATLAB u(2,:))

The spatial operator is  a*u_x + T(x)*u_xx,  i.e. (T u_x)_x for T = a*x + b
with unit density, using second-order central differences.
"""
import numpy as np


def centdiftt_mem(u, t, T, F_Cy, h1, h2, a, dt, jj):
    """Return du/dt with the same 2 x N layout as ``u``.

    F_Cy : contact accelerations applied at the string indices ``jj``
           (empty array -> nothing is added, as in the MATLAB code).
    jj   : 0-based indices of string points in contact with the bead.
    """
    u = np.asarray(u, dtype=float)

    # Shifted copies of the displacement (MATLAB: u(2,im), u(2,ip)).
    u_minus = np.roll(u[1], 1)
    u_plus = np.roll(u[1], -1)

    # Boundary padding, exactly as in the MATLAB code.
    u_plus[-1] = u_plus[-2]
    u_minus[0] = u_minus[1]

    u_out = np.empty_like(u)
    # u_t RHS
    u_out[1] = u[0]
    #            first spatial derivative      second spatial derivative
    # u_tt RHS
    u_out[0] = h1 * a * (u_plus - u_minus) + h2 * T * (u_plus - 2.0 * u[1] + u_minus)

    F_Cy = np.atleast_1d(np.asarray(F_Cy, dtype=float))
    if F_Cy.size > 0:
        jj = np.asarray(jj, dtype=int)
        u_out[0, jj] = u_out[0, jj] + F_Cy          # add forces
    return u_out
