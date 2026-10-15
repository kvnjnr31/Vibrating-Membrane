"""Reverse the wave to the instant of bead impact (event-driven step).

Port of old_functions/wave_corr_old.m.

PORT NOTE: the MATLAB function uses u_mag_2, b_u1_mag_2, d_2 and R without
receiving them, so it cannot run as archived.  They are explicit arguments
here; Bead_Membrane_Dynamics_old.py passes the values it already computes.
"""
import numpy as np
from scipy.optimize import fsolve


def wave_corr(u, u_old, B, x, k, j, t, dt, u_mag_2, b_u1_mag_2, d_2, R,
              plot=False, verbose=True):
    """Return u linearly interpolated back to the bead-string impact time.

    k, j are 0-based indices of the string points bracketing the bead.
    ``t`` is updated locally only, as in MATLAB (it is not returned).
    """
    # ------------------- algorithm parameters -------------------
    # velocities of the bead and of the two string points
    vxb = (B[0] - B[2]) / dt
    vyb = (B[1] - B[3]) / dt                                       # bead
    vy1 = (u[1, j] - u_old[1, j]) / dt
    vy2 = (u[1, k] - u_old[1, k]) / dt                             # string

    # spatial differences (between initial coordinates)
    Delta_Y_21 = u_old[1, k] - u_old[1, j]
    Delta_X_21 = x[k] - x[j]
    Delta_Y_b1 = B[3] - u_old[1, j]
    Delta_X_b1 = B[2] - x[j]
    Delta_Y_b2 = B[3] - u_old[1, k]
    Delta_X_b2 = B[2] - x[k]

    # velocity differences
    Delta_Vx_b1 = vxb
    Delta_Vy_b1 = vyb - vy1
    Delta_Vy_21 = vy2 - vy1
    Delta_Vx_b2 = vxb
    Delta_Vy_b2 = vyb - vy2

    # angle between the vectors
    b_dot_u = ((B[0] - x[j]) * (x[k] - x[j])
               + (B[1] - u[1, j]) * (u[1, k] - u[1, j])) / np.sqrt(u_mag_2)
    theta_b1 = (360.0 / (2.0 * np.pi)) * np.arccos(b_dot_u / np.sqrt(b_u1_mag_2))

    # reverse time
    t = t - dt

    # -------------- bead-string intersection time: tf --------------
    tf = None
    if (d_2 <= R * R) and (theta_b1 < 90):                 # bead - segment

        def fun(t1):
            seg2 = Delta_X_21 * Delta_X_21 + (Delta_Vy_21 * t1 + Delta_Y_21) ** 2
            return 1e7 * (
                ((Delta_Vx_b1 * t1 + Delta_X_b1) * Delta_X_21
                 + (Delta_Vy_b1 * t1 + Delta_Y_b1) * (Delta_Vy_21 * t1 + Delta_Y_21)) ** 2
                - seg2 * (Delta_Vx_b1 * t1 + Delta_X_b1) ** 2
                - seg2 * (Delta_Vy_b1 * t1 + Delta_Y_b1) ** 2
                + u_mag_2 * R ** 2)

        if plot:
            import matplotlib.pyplot as plt
            tt = np.linspace(-1e-2, 5e-2, 200)
            plt.figure(200)
            plt.clf()
            plt.plot(tt, fun(tt), "ko-")

        sol, info, ier, msg = fsolve(fun, dt, full_output=True)
        tf = float(sol[0])
        if verbose:
            print("fval =", float(info["fvec"][0]))
            print("bead segment")

    elif d_2 <= R * R:                                     # bead - u1
        Delta_v_dot_Delta_r = Delta_Vx_b1 * Delta_X_b1 + Delta_Vy_b1 * Delta_Y_b1
        Delta_r_dot_Delta_r = Delta_X_b1 * Delta_X_b1 + Delta_Y_b1 * Delta_Y_b1
        Delta_v_dot_Delta_v = Delta_Vx_b1 * Delta_Vx_b1 + Delta_Vy_b1 * Delta_Vy_b1
        d = Delta_v_dot_Delta_r ** 2 - Delta_v_dot_Delta_v * (Delta_r_dot_Delta_r - R * R)

        tf = -(Delta_v_dot_Delta_r + np.emath.sqrt(d)) / Delta_v_dot_Delta_v
        if verbose:
            print("bead u1")

    elif d_2 <= R * R:                                     # bead - u2
        # PORT NOTE: same condition as the branch above, so this branch can
        # never be reached (true in the MATLAB file as well).
        Delta_v_dot_Delta_r = Delta_Vx_b2 * Delta_X_b2 + Delta_Vy_b2 * Delta_Y_b2
        Delta_r_dot_Delta_r = Delta_X_b2 * Delta_X_b2 + Delta_Y_b2 * Delta_Y_b2
        Delta_v_dot_Delta_v = Delta_Vx_b2 * Delta_Vx_b2 + Delta_Vy_b2 * Delta_Vy_b2
        d = Delta_v_dot_Delta_r ** 2 - Delta_v_dot_Delta_v * (Delta_r_dot_Delta_r - R * R)

        tf = -(Delta_v_dot_Delta_r + np.emath.sqrt(d)) / Delta_v_dot_Delta_v
        if verbose:
            print("bead u2")

    if tf is None:
        # MATLAB stops here with "undefined variable tf".
        raise RuntimeError("wave_corr: no impact branch taken (d_2 > R^2), tf is undefined")

    # wave correction
    u = np.array(u, dtype=float)
    u[0] = tf * (u[0] - u_old[0]) / dt + u_old[0]          # corrects string u_t
    u[1] = tf * (u[1] - u_old[1]) / dt + u_old[1]          # corrects string u

    t = t + tf                                             # noqa: F841 (local only)
    return u
