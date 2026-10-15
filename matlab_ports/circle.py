"""Points on a circle, for plotting the bead.  Port of circle.m."""
import numpy as np


def circle(x, y, r, cir_pts):
    th = np.linspace(0.0, 2.0 * np.pi, cir_pts)
    xunit = r * np.cos(th) + x
    yunit = r * np.sin(th) + y
    return xunit, yunit
