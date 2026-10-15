"""Detect string-bead events.  Port of event_fun.m.

B = [x, y, x0, y0]  (bead position and previous position).
The bead is grid free.  Intersection happens at the end points u1, u2 or
along the segment joining them.

Notes
-----
(1) b_u1_mag_2 - squared distance between the bead and the first end point.
(2) b_u2_mag_2 - squared distance between the bead and the second end point.
(3) indx_L, indx_R - used to compute the matching points.

All returned indices are 0-based (MATLAB index minus one).
"""
import numpy as np


def event_fun(u, B, x, Rad):
    """Return (u, d, L, R, ML, MR, indx_L, indx_R, jj)."""
    bx, by = B[0], B[1]

    # ---- string points the bead x-coordinate lies between: j <= x <= k ----
    k = int(np.flatnonzero(bx <= x).min())           # least upper bound: u2
    j = int(np.flatnonzero(bx >= x).max())           # greatest lower bound: u1

    # squared vector magnitudes
    u_mag_2 = (x[k] - x[j]) ** 2 + (u[1, k] - u[1, j]) ** 2
    b_u1_mag_2 = (bx - x[j]) ** 2 + (by - u[1, j]) ** 2
    b_u2_mag_2 = (bx - x[k]) ** 2 + (by - u[1, k]) ** 2      # noqa: F841

    # ---- bead-segment distance d ----
    # (1) scalar projection of b onto u.  If the bead sits exactly on a grid
    #     point then j == k and this is 0/0 = NaN, as in MATLAB.
    with np.errstate(divide="ignore", invalid="ignore"):
        b_dot_u_2 = np.float64(
            ((bx - x[j]) * (x[k] - x[j]) + (by - u[1, j]) * (u[1, k] - u[1, j])) ** 2
        ) / np.float64(u_mag_2)
    # (2) bead-segment distance (complex if round-off makes d_2 < 0, as in MATLAB)
    d_2 = b_u1_mag_2 - b_dot_u_2
    d = np.emath.sqrt(d_2)

    # ---- detect all non-physical string points (inside the bead) ----
    dist = (bx - x) ** 2 + (by - u[1]) ** 2          # squared bead-u distance
    jj = np.flatnonzero(np.sqrt(dist) < Rad)         # non-physical indices

    if jj.size > 0:
        j = int(jj.min()) - 1                        # string point to the left
        k = int(jj.max()) + 1                        # string point to the right
        if j < 0 or k >= x.size:
            raise IndexError("bead contact region reaches the end of the string")

    # string-bead matching points
    L = np.array([x[j], u[1, j]])                    # left matching point
    R = np.array([x[k], u[1, k]])                    # right matching point
    indx_L, indx_R = j, k

    # non-physical points
    if jj.size > 0:
        ML = np.array([x[jj.min()], u[1, jj.min()]])
        MR = np.array([x[jj.max()], u[1, jj.max()]])
        # u[1, jj] = -np.sqrt(Rad**2 - (x[jj] - B[0])**2) + B[1]   # correction
    else:
        ML = R                                       # right point as midpoint
        MR = L                                       # left point as midpoint

    return u, d, L, R, ML, MR, indx_L, indx_R, jj
