"""Scratch file.  Port of sctratch.m (kept for reference; nothing imports it).

The MATLAB file holds two commented-out fragments and an earlier analytic
version of force_fun.  The fragments are reproduced as comments.
"""
import numpy as np

# --- fragment 1 (commented out in MATLAB): quadratic for the impact time ---
# a = (Delta_Vx_b1*Delta_X_21 + Delta_Vy_b1*Delta_Y_21 + Delta_Vy_21*Delta_Y_b1)**2 \
#     - u_mag_2*(Delta_Vx_b1*Delta_Vx_b1 + Delta_Vy_b1*Delta_Vy_b1)
#
# b = 2*(Delta_Vx_b1*Delta_X_21 + Delta_Vy_b1*Delta_Y_21 + Delta_Vy_21*Delta_Y_b1) \
#       * (Delta_X_b1*Delta_X_21 + Delta_Y_b1*Delta_Y_21) \
#     - 2*u_mag_2*(Delta_Vx_b1*Delta_X_b1 + Delta_Y_b1*Delta_Y_b1)
#
# c = (Delta_X_b1*Delta_X_21 + Delta_Y_b1*Delta_Y_21)**2 \
#     - u_mag_2*(Delta_X_b1*Delta_X_b1 + Delta_Y_b1*Delta_Y_b1) + u_mag_2*R*R
#
# t0 = (-b - np.sqrt(b*b - 4*a*c))/(2*a)          # initial guess for solver

# --- fragment 2 (commented out in MATLAB) ---
# Cl_x, Cl_y = contact_fun(L[0], L[1], ML[0], ML[1], B[0], B[1], Rad)   # left matching point
# Cr_x, Cr_y = contact_fun(R[0], R[1], MR[0], MR[1], B[0], B[1], Rad)   # right matching point


def force_fun(a, b, R, Cl_x, Cl_y, Cr_x, Cr_y, b_x, b_y, jj, x, u, t, T, h1, h2, dt, vyb):
    """Earlier analytic force with per-contact-point forces.

    Returns (F_Tx, F_Ty, F_Cx, F_Cy).  ``jj`` holds 0-based contact indices.
    """
    jj = np.asarray(jj, dtype=int)
    # computed but never used in the MATLAB code
    with np.errstate(divide="ignore", invalid="ignore"):
        vrel = (1 + (u[0, jj] - vyb) / np.abs(u[0, jj] - vyb)) / 2      # noqa: F841

    if jj.size > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            # left and right angles, coordinates relative to the bead centre
            x0 = x[jj.min()] - b_x
            y0 = u[1, jj.min()] - b_y
            theta_i = np.pi + np.arctan(np.float64(y0) / np.float64(x0))
            x0 = x[jj.max()] - b_x
            y0 = u[1, jj.max()] - b_y
            theta_f = 2 * np.pi + np.arctan(np.float64(y0) / np.float64(x0))

        # total force on the bead and string segment
        F_Tx = -((a * R / 2) * (theta_f - theta_i)
                 + (a * R / 4) * (np.sin(2 * theta_f) - np.sin(2 * theta_i))
                 + b * (np.sin(theta_f) - np.sin(theta_i)))
        F_Ty = ((a * R / 2) * (np.cos(2 * theta_f) - np.cos(2 * theta_i))
                + b * (np.cos(theta_f) - np.cos(theta_i))) / 2 - 9.81 / 2

        # force on the contact points
        # PORT NOTE: MATLAB has length(jj+1), which equals length(jj);
        # length(jj)+1 may have been intended.  Kept as written.
        dtheta = (theta_f - theta_i) / jj.size
        F_Cx = (a * (x[jj] - b_x) + b) * (x[jj] - b_x) / R
        F_Cy = -(dtheta / 2) * (a * (x[jj] - b_x) + b) * (u[1, jj] - b_y) / R - dtheta * 9.81 / 2
    else:
        F_Tx = -9.81
        F_Ty = -9.81
        F_Cx = np.array([0.0])
        F_Cy = np.array([0.0])
    return F_Tx, F_Ty, F_Cx, F_Cy
