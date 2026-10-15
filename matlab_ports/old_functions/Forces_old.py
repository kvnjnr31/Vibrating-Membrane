"""Wave and bead forces (penalty spring along the segment normal).

Port of old_functions/Forces_old.m.

PORT NOTE: the MATLAB function uses u_mag_2 without receiving or computing
it.  It is computed here from the two string points, which is how every
other file in the archive defines it.
"""
import numpy as np


def Forces(u, B, x, R, d, K, k, j):
    """Return (F_x, F_y).  k, j are 0-based string indices."""
    u_mag_2 = (x[k] - x[j]) ** 2 + (u[1, k] - u[1, j]) ** 2

    # scalar projection of b onto u
    b_dot_u = ((B[0] - x[j]) * (x[k] - x[j])
               + (B[1] - u[1, j]) * (u[1, k] - u[1, j])) / u_mag_2

    # projection vector components
    proj_x = b_dot_u * (x[k] - x[j])
    proj_y = b_dot_u * (u[1, k] - u[1, j])

    # b vector components
    b_x = B[0] - x[j]
    b_y = B[1] - u[1, j]

    # normal vector magnitude
    norm_mag = np.sqrt((b_x - proj_x) ** 2 + (b_y - proj_y) ** 2)

    # force vector components
    F_x = K * (d + R) * (b_x - proj_x) / norm_mag
    F_y = K * (d + R) * (b_y - proj_y) / norm_mag
    return F_x, F_y
