"""Find and draw the reaction force vector on a bead resting in a string.

Port of mass_spring_sim/reaction_fun.m (a script).  The tension slope ``a``
is stepped each frame and the resulting force vector is drawn.

Run:
    python reaction_fun.py              # 200 frames, 0.5 s apart, as in MATLAB
    python reaction_fun.py --frames 20 --pause 0.05
"""
import argparse

import numpy as np

from event_fun import event_fun
from contact_fun import contact_fun
from force_fun import force_fun
from circle import circle

N_DEFAULT = 50


def frames(nframes=200, N=N_DEFAULT):
    """Yield one dict of results per frame (no plotting)."""
    x = np.linspace(0.0, 10.0, N)                 # spatial discretisation
    Rad = 1.0                                     # bead radius
    a, b = 0.0, 1.0                               # tension parameters
    B = np.array([4.5, 0.5])                      # bead location

    for k in range(1, nframes + 1):
        a = a - 0.01 * k
        # string amplitudes (initial state)
        u = np.zeros((2, N))
        u[0] = 0 * x + 0 * (np.random.rand(N) - 0.5)
        u[1] = 0 * x + 0 * (np.random.rand(N) - 0.5)

        # detect events
        u, d, L, R, ML, MR, indx_L, indx_R = event_fun(u, B, x, Rad)[:8]

        # left matching point
        Cl_x, Cl_y = contact_fun(L[0], L[1], ML[0], ML[1], B[0], B[1], Rad)
        # right matching point
        Cr_x, Cr_y = contact_fun(R[0], R[1], MR[0], MR[1], B[0], B[1], Rad)

        # reaction force
        F_Tx, F_Ty = force_fun(a, b, Rad, Cl_x, Cl_y, Cr_x, Cr_y, B[0], B[1])

        yield dict(k=k, a=a, x=x, u=u, B=B, Rad=Rad, L=L, R=R, ML=ML, MR=MR,
                   indx_L=indx_L, indx_R=indx_R, Cl=(Cl_x, Cl_y), Cr=(Cr_x, Cr_y),
                   F=(F_Tx, F_Ty))


def main(nframes=200, pause=0.5, N=N_DEFAULT):
    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure(1)
    for f in frames(nframes, N):
        x, u, B = f["x"], f["u"], f["B"]
        (Cl_x, Cl_y), (Cr_x, Cr_y), (F_Tx, F_Ty) = f["Cl"], f["Cr"], f["F"]
        iL, iR = f["indx_L"], f["indx_R"]
        xunit, yunit = circle(B[0], B[1], f["Rad"], 20)

        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(xunit, yunit, "k", linewidth=2)
        ax.plot(Cl_x, Cl_y, "r*", Cr_x, Cr_y, "r*", linewidth=2)
        ax.plot(np.append(x[: iL + 1], Cl_x), np.append(u[1, : iL + 1], Cl_y), "ko-", linewidth=2)
        ax.plot(np.append(Cr_x, x[iR:]), np.append(Cr_y, u[1, iR:]), "ko-", linewidth=2)
        ax.plot(f["ML"][0], f["ML"][1], "ko", linewidth=2)
        ax.plot([B[0], B[0] + F_Tx], [B[1], B[1] + F_Ty], "k", linewidth=2)
        ax.set_xlim(0, 10)
        ax.set_ylim(-5, 5)
        ax.set_box_aspect(1)
        plt.pause(pause)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--pause", type=float, default=0.5)
    p.add_argument("--points", type=int, default=N_DEFAULT)
    args = p.parse_args()
    main(args.frames, args.pause, args.points)
