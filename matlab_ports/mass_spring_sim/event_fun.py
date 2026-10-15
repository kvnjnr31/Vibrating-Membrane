"""String-bead interactions (single-point version).  Port of mass_spring_sim/event_fun.m.

B = [x, y, ...]  bead position.
The bead is grid free.  Intersection happens at the end points u1, u2 or
along the segment joining them.

Notes
-----
(1) b_u1_mag_2 - squared distance between the bead and the first end point.
(2) b_u2_mag_2 - squared distance between the bead and the second end point.
(3) indx_L, indx_R - used to compute the matching points.

Unlike the newer event_fun in the parent folder, this version only checks the
two string points that bracket the bead, so at most one point is flagged as
non-physical.  All returned indices are 0-based.
"""
import numpy as np


def event_fun(u, B, x, Rad):
    """Return (u, d, L, R, ML, MR, indx_L, indx_R)."""
    bx, by = B[0], B[1]

    # ---- string points the bead x-coordinate lies between: j <= x <= k ----
    k = int(np.flatnonzero(bx <= x).min())           # least upper bound: u2
    j = int(np.flatnonzero(bx >= x).max())           # greatest lower bound: u1

    # squared vector magnitudes
    u_mag_2 = (x[k] - x[j]) ** 2 + (u[1, k] - u[1, j]) ** 2
    b_u1_mag_2 = (bx - x[j]) ** 2 + (by - u[1, j]) ** 2
    b_u2_mag_2 = (bx - x[k]) ** 2 + (by - u[1, k]) ** 2

    # ---- bead-segment distance d ----
    with np.errstate(divide="ignore", invalid="ignore"):
        b_dot_u_2 = np.float64(
            ((bx - x[j]) * (x[k] - x[j]) + (by - u[1, j]) * (u[1, k] - u[1, j])) ** 2
        ) / np.float64(u_mag_2)
    d_2 = b_u1_mag_2 - b_dot_u_2
    d = np.emath.sqrt(d_2)

    # ---- detect a non-physical string point ----
    indx_np = -1
    if np.sqrt(b_u1_mag_2) < Rad:                    # check left string point
        indx_np = j                                  # non-physical index
        j = j - 1                                    # shift left point left
    elif np.sqrt(b_u2_mag_2) < Rad:                  # check right string point
        indx_np = k                                  # non-physical index
        k = k + 1                                    # shift right point right
    if j < 0 or k >= x.size:
        raise IndexError("bead contact region reaches the end of the string")

    # string-bead matching points
    L = np.array([x[j], u[1, j]])                    # left matching point
    R = np.array([x[k], u[1, k]])                    # right matching point
    indx_L, indx_R = j, k

    # non-physical point  (MATLAB: indx_np > 0 with 1-based indices)
    if indx_np >= 0:
        ML = np.array([x[indx_np], u[1, indx_np]])
        MR = ML
        # u[1, indx_np] = -np.sqrt(Rad**2 - (x[indx_np] - B[0])**2) + B[1]  # correction
    else:
        ML = R                                       # right point as midpoint
        MR = L                                       # left point as midpoint

    return u, d, L, R, ML, MR, indx_L, indx_R
