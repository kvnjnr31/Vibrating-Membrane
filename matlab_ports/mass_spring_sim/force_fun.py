"""Reaction force on the bead from the string wrapped around it.

Port of mass_spring_sim/force_fun.m (analytic version: tension T = a*x + b
integrated over the arc between the two matching points).
"""
import numpy as np


def _atan_ratio(y, x):
    """atan(y/x) with MATLAB semantics: x == 0 gives +-pi/2 or NaN, and
    complex inputs (a string line that misses the bead) stay complex."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.arctan(np.asarray(y) / np.asarray(x))[()]


def force_fun(a, b, R, Cl_x, Cl_y, Cr_x, Cr_y, b_x, b_y):
    """Return (F_Tx, F_Ty)."""
    # left and right angles (coordinates relative to the bead centre)
    x = Cl_x - b_x
    y = Cl_y - b_y
    theta_i = np.pi + _atan_ratio(y, x)              # angle of left matching point

    x = Cr_x - b_x
    y = Cr_y - b_y
    theta_f = 2.0 * np.pi + _atan_ratio(y, x)        # angle of right matching point

    # total force
    F_Tx = -((a * R / 2.0) * (theta_f - theta_i)
             + (a * R / 4.0) * (np.sin(2 * theta_f) - np.sin(2 * theta_i))
             + b * (np.sin(theta_f) - np.sin(theta_i)))
    F_Ty = ((a * R / 2.0) * (np.cos(2 * theta_f) - np.cos(2 * theta_i))
            - b * (np.cos(theta_i) - np.cos(theta_f)))
    return F_Tx, F_Ty
