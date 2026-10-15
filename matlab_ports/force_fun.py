"""Reaction force on the bead.  Port of force_fun.m (current, grid-based version)."""
import numpy as np

from centdiftt_mem import centdiftt_mem

G = 9.81


def force_fun(a, R, b_x, b_y, jj0, jj, x, u, t, T, h1, h2, dt, F_Cy, verbose=False):
    """Return (F_Tx, F_Ty, F_Cx, F_Cy).

    jj0 / F_Cy : contact indices and contact accelerations from the previous
                 step (used to evaluate the string acceleration).
    jj         : contact indices of the current step.

    The MATLAB file prints ``sum(r_proj_y)/2`` on every call because of a
    missing semicolon; pass ``verbose=True`` to reproduce that.
    """
    jj = np.asarray(jj, dtype=int)

    if jj.size > 0:
        # string acceleration at the contact points
        r = centdiftt_mem(u, t, T, F_Cy, h1, h2, a, dt, jj0)
        r = r[0]
        y = u[1]

        r_proj_x = r[jj] * (y[jj] - b_y) * (x[jj] - b_x) / (R * R)
        r_proj_y = r[jj] * (y[jj] - b_y) * (y[jj] - b_y) / (R * R)

        # total acceleration on the bead
        F_Tx = np.sum(r_proj_x)
        F_Ty = -np.sum(r_proj_y) / 2.0 - G / 2.0
        if verbose:
            print(np.sum(r_proj_y) / 2.0)

        # acceleration on the string points
        F_Cx = -(r[jj] + G)
        F_Cy = -(r_proj_y + G)
    else:
        F_Tx = -G
        F_Ty = -G
        F_Cx = np.array([0.0])
        F_Cy = np.array([0.0])

    return F_Tx, F_Ty, F_Cx, F_Cy
