"""Event-driven bead-string step.  Port of old_functions/Bead_Membrane_Dynamics_old.m.

B = [x, y, x0, y0]
The bead is grid free.  Intersection happens at the end points u1, u2 or
along the segment joining them.
"""
from bead_segment_distance_old import bead_segment_geometry
from event_fun_old import event_fun
from wave_corr_old import wave_corr


def Bead_Membrane_Dynamics(u, u_old, B, state_new, x, R, t, dt, plot=False, verbose=True):
    """Return (u, B, state_new, t)."""
    # string points bracketing the bead, vector magnitudes, bead-segment distance
    j, k, u_mag_2, b_u1_mag_2, b_u2_mag_2, d_2 = bead_segment_geometry(u, B, x)

    # ------------------------ determine impact event ------------------------
    state_new, impact = event_fun(d_2, b_u1_mag_2, b_u2_mag_2, R, state_new)

    # ------------------ reverse wave to state at bead impact -----------------
    if impact:
        # PORT NOTE: u_mag_2, b_u1_mag_2, d_2 and R are passed explicitly
        # (the MATLAB wave_corr used them without receiving them).
        u = wave_corr(u, u_old, B, x, k, j, t, dt, u_mag_2, b_u1_mag_2, d_2, R,
                      plot=plot, verbose=verbose)

    return u, B, state_new, t
