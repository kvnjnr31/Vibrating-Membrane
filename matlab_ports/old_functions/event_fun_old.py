"""System state for the event-driven scheme.  Port of old_functions/event_fun_old.m.

Notes
-----
(1) state  = 0  bead and string not in contact
(2) state  = 1  bead and string in contact
(3) impact != 0 the contact state changed this step
    (-1: new contact, +1: contact released; MATLAB tests ``if (impact)``,
     which is true for both).
"""


def event_fun(d_2, b_u1_mag_2, b_u2_mag_2, R, state_new):
    """Return (state_new, impact)."""
    if (d_2 <= R * R) or (b_u1_mag_2 <= R * R) or (b_u2_mag_2 <= R * R):
        state = 1                                    # bead in contact with string
    else:
        state = 0                                    # bead not in contact

    impact = state_new - state
    state_new = state
    return state_new, impact
