"""Penalty-spring force between the bead and the string.

Port of old_functions/bead_segment_force.m.

B = [x, y, x0, y0]
The bead is grid free.  Intersection happens at the end points u1, u2 or
along the segment joining them.

PORT NOTES
* The MATLAB file is cut off: the outer ``if d>0`` is never closed and
  nothing is assigned to F / F_dist when d <= 0.  Here that case returns
  zero force, matching the inner "no contact" branch.
* If both bracketing string points lie inside the bead, the MATLAB lines
  ``(x_b - x_c)^2`` and ``F*g`` are not valid for two-element vectors.
  That case raises an error here too.
"""
import numpy as np


def bead_segment_force(u, B, T, x, R):
    """Return (u, F_dist, F)."""
    u = np.array(u, dtype=float)

    # string points the bead lies between: j < x < k
    k = int(np.flatnonzero(B[0] < x).min())          # least upper bound: u2
    j = int(np.flatnonzero(B[0] > x).max())          # greatest lower bound: u1

    # segment slope
    slope = (u[1, k] - u[1, j]) / (x[k] - x[j])

    # bead location
    x_b, y_b = B[0], B[1]

    # bead-segment depression d
    # (1) interpolate the string height at x_b
    y = slope * (x_b - x[j]) + u[1, j]
    # (2) bead-segment depression
    d = y - (y_b - R)

    F = 0.0
    F_dist = 0 * x

    if d > 0:
        # string point intersecting the sphere (if it exists)
        indx = np.array([j, k])
        M_dist = (np.array([x_b - x[j], x_b - x[k]]) ** 2
                  + np.array([y_b - u[1, j], y_b - u[1, k]]) ** 2)
        n = np.flatnonzero(M_dist <= R * R)

        if n.size > 0:
            if n.size > 1:
                raise ValueError("both bracketing string points are inside the bead; "
                                 "the original code does not handle this case")
            indx_c = int(indx[n][0])                 # index of collision
            x_c = x[indx_c]
            y_c = u[1, indx_c]                       # point of collision

            # distribute the spring force to the end points
            # (1) tension
            T_int = T[indx_c]                        # spring constant
            u[1, indx_c] = y_b - np.sqrt(R * R - (x_b - x_c) ** 2)   # wave correction
            # (2) spring force at the bead location
            F = -T_int * abs(u[1, indx_c] - y_c) * 5000
            # (3) Hann window of the force distribution
            g = 0 * x
            g[j:k + 1] = 0.5 * (1 - np.cos(2 * np.pi * (x[j:k + 1] - x_b) / R))

            F_dist = F * g                           # force distribution

    return u, F_dist, F
