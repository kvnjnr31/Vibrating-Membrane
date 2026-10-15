"""Matching (intersection) point between the bead and a string segment.

Port of contact_fun.m.

R        = circle radius
(x1, y1) = starting string point
(x2, y2) = ending string point
(k, h)   = bead centre
"""
import numpy as np


def contact_fun(x1, y1, x2, y2, k, h, R):
    """Return (x_star, y_star): where the line through the two string points
    meets the circle, on the side of the starting point."""
    # slope
    m = (y1 - y2) / (x1 - x2)

    # sign function
    s = (x1 - x2) / abs(x1 - x2)

    # quadratic coefficients
    a = 1.0 + m ** 2
    b = 2.0 * m * (y1 - h) - 2.0 * (m ** 2) * x1 - 2.0 * k
    c = k ** 2 + (m ** 2) * (x1 ** 2) - 2.0 * m * x1 * (y1 - h) + (y1 - h) ** 2 - R ** 2
    x_star = (-b + s * np.emath.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    y_star = m * (x_star - x1) + y1
    return x_star, y_star
