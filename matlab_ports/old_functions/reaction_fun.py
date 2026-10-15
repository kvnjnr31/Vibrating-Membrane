"""Find the reaction force vector (early geometric draft).

Port of old_functions/reaction_fun.m (a script that stops after computing
the projection of the bead vector onto the chord between contact points).

PORT NOTES - the MATLAB script cannot run as archived:
* the first contact_fun call passes ``r``, which is never defined; the bead
  radius ``Rad`` is used here, as in the second call;
* ``ClB_mag2`` is never defined.  It is computed here as |ClB|^2, which is
  what the name says.  Note that the projection of ClB onto ClCr would
  normally be divided by |ClCr|^2 (``ClCr_mag2``, computed but unused).
"""
import numpy as np

from contact_fun import contact_fun


def main():
    B = np.array([5.0, 6.0])          # bead location
    Rad = 5.0                         # bead radius
    L = np.array([-2.0, 1.0])         # left string point
    M = np.array([3.0, 8.0])          # mid string point
    R = np.array([7.0, 8.0])          # right string point

    # left contact point
    Cl_x, Cl_y = contact_fun(L[0], L[1], M[0], M[1], B[0], B[1], Rad)
    # right contact point
    Cr_x, Cr_y = contact_fun(R[0], R[1], M[0], M[1], B[0], B[1], Rad)

    # project the Cl-B vector onto the Cl-Cr vector
    ClB = np.array([B[0] - Cl_x, B[1] - Cl_y])        # order gives direction
    ClCr = np.array([Cr_x - Cl_x, Cr_y - Cl_y])

    dot = ClCr[0] * ClB[0] + ClCr[1] * ClB[1]
    ClCr_mag2 = ClCr[0] * ClCr[0] + ClCr[1] * ClCr[1]
    ClB_mag2 = ClB[0] * ClB[0] + ClB[1] * ClB[1]      # see PORT NOTES

    proj_x = dot * ClCr[0] / ClB_mag2
    proj_y = dot * ClCr[1] / ClB_mag2
    return dict(Cl=(Cl_x, Cl_y), Cr=(Cr_x, Cr_y), ClB=ClB, ClCr=ClCr, dot=dot,
                ClCr_mag2=ClCr_mag2, proj=(proj_x, proj_y))


if __name__ == "__main__":
    for key, val in main().items():
        print(f"{key:10s} {val}")
