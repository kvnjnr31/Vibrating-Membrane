"""Bead-segment distance.  Port of old_functions/bead_segment_distance_old.m.

B = [x, y, x0, y0]
The bead is grid free.  Intersection happens at the end points u1, u2 or
along the segment joining them.

Notes
-----
(1) b_u1_mag_2 - squared distance between the bead and the first end point.
(2) b_u2_mag_2 - squared distance between the bead and the second end point.
"""
import numpy as np


def bead_segment_geometry(u, B, x):
    """Shared geometry: returns (j, k, u_mag_2, b_u1_mag_2, b_u2_mag_2, d_2).

    j, k are the 0-based string indices with x[j] < bead x < x[k]
    (strict inequalities, as in the MATLAB code).
    """
    bx, by = B[0], B[1]
    k = int(np.flatnonzero(bx < x).min())            # least upper bound: u2
    j = int(np.flatnonzero(bx > x).max())            # greatest lower bound: u1

    # squared vector magnitudes
    u_mag_2 = (x[k] - x[j]) ** 2 + (u[1, k] - u[1, j]) ** 2
    b_u1_mag_2 = (bx - x[j]) ** 2 + (by - u[1, j]) ** 2
    b_u2_mag_2 = (bx - x[k]) ** 2 + (by - u[1, k]) ** 2

    # (1) scalar projection of b onto u1
    b_dot_u_2 = (((bx - x[j]) * (x[k] - x[j])
                  + (by - u[1, j]) * (u[1, k] - u[1, j])) ** 2) / u_mag_2
    # (2) squared bead-segment distance
    d_2 = b_u1_mag_2 - b_dot_u_2
    return j, k, u_mag_2, b_u1_mag_2, b_u2_mag_2, d_2


def bead_segment_distance(u, B, x):
    """Return d, the distance from the bead centre to the line through the
    two string points that bracket it."""
    d_2 = bead_segment_geometry(u, B, x)[-1]
    return np.emath.sqrt(d_2)
