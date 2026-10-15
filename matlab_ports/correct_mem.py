"""Corrector step for the string-bead system.  Port of correct_mem.m.

The string corrections are multiplied by zero in the MATLAB file, i.e. they
are switched off; only the bead is corrected.  The zeros are kept here so
the corrector can be re-enabled in the same place.
"""
import numpy as np


def correct_mem(u, u_old, B, t, T, F_Cy, F_Ty, Fby_pr, F0_Cy, vyb, h1, h2, a, dt, jj):
    """Return (u, B, F_Cy, Fby, vyb)."""
    u = np.array(u, dtype=float)
    B = np.array(B, dtype=float)
    jj = np.asarray(jj, dtype=int)
    F_Cy = np.atleast_1d(np.asarray(F_Cy, dtype=float))

    dr_tt = np.zeros(u.shape[1])
    c0, c1, c2 = 1.0 / 6.0, 5.0 / 6.0, 1.0

    # delta force
    u_tt_pr = (u[0, jj] - u_old[0, jj]) / dt
    dr_tt[jj] = F_Cy - u_tt_pr

    # correct the string (disabled by the factor 0, as in the MATLAB code)
    u[1] = u[1] + 0 * c0 * dt * dt * dr_tt / 2.0
    u[0] = u[0] + 0 * c1 * dt * dr_tt / 2.0
    F_Cy = F_Cy + 0 * c2 * dr_tt[jj]      # empty when there is no contact

    # bead acceleration
    dr_b = F_Ty - Fby_pr
    # correct the bead
    B[1] = B[1] + c0 * dt * dt * dr_b / 2.0
    vyb = vyb + c1 * dt * dr_b / 2.0
    Fby = F_Ty + c2 * dr_b

    return u, B, F_Cy, Fby, vyb
