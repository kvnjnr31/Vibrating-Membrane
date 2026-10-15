"""Points of tangency from a string point to the bead.

Port of old_functions/tangent_fun.m.

PORT NOTE: kept exactly as written, including the naming: the MATLAB code
sets k = B(1), h = B(2) but then uses (h, k) as the circle centre, i.e. the
bead's x and y are swapped relative to the other files.
"""
import numpy as np

from circle import circle


def tangent_fun(ux, uy, B, R, plot=False):
    """Return (x_star, y_star)."""
    # circle centre
    k, h = B[0], B[1]            # (x, y) location of bead

    # the point u(x, y)
    x1, y1 = ux, uy

    # quadratic coefficients
    q = R ** 2 + x1 * h + y1 * k - (h * h + k * k)
    a = (x1 - h) * (x1 - h) + (y1 - k) * (y1 - k)
    b = 2 * (k * (x1 - h) * (y1 - k) - q * (x1 - h) - (y1 - k) * (y1 - k) * h)
    c = (q * q + (y1 - k) * (y1 - k) * (k * k + h * h)
         - 2 * q * (y1 - k) * k - (y1 - k) * (y1 - k) * R * R)

    x_star = (-b + np.emath.sqrt(b * b - 4 * a * c)) / (2 * a)
    y_star = (q - (x1 - h) * x_star) / (y1 - k)

    if plot:
        import matplotlib.pyplot as plt
        xunit, yunit = circle(h, k, R, 500)
        plt.figure(99)
        plt.clf()
        plt.plot(xunit, yunit, "k.", x_star, y_star, "ro", x1, y1, "go")
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.gca().set_box_aspect(1)

    return x_star, y_star
