"""Simulate a bead on a string in 1D.  Port of membrane_1D.m.

B = [x, y, x0, y0]
The bead is grid free.  Intersection happens at the end points u1, u2 or
along the segment joining them.

Run:
    python membrane_1D.py                 # animate, as the MATLAB script did
    python membrane_1D.py --nsteps 3000 --no-plot
"""
import argparse

import numpy as np

from Runge_Kutta_mem import Runge_Kutta_mem
from event_fun import event_fun
from force_fun import force_fun
from correct_mem import correct_mem
from circle import circle


def simulate(nsteps=100000, plot=True, plot_every=1, verbose=False, record=None):
    """Run the simulation and return the final state.

    record : optional callable ``record(n, u, B, vyb, jj)`` called every step
             (n is the 1-based step number, as in the MATLAB loop).
    """
    # simulation parameters
    points = 300                          # number of grid points
    L = 10.0                              # length of the string
    dx = L / points                       # spatial discretisation
    h2 = 1.0 / (dx * dx)
    h1 = 1.0 / (2.0 * dx)
    dt = 0.01                             # time step
    a = 0.0                               # tension slope
    b = 10.0                              # tension intercept
    t = 0.0                               # initial time (never advanced, as in MATLAB)

    # tension
    x = np.linspace(0.0, L, points)
    T = a * x + b

    # string setup
    u = np.zeros((2, points))             # row 0: velocity, row 1: amplitude
    # u[1, 149] = 0                       # initial condition
    u_old = u.copy()                      # previous solution

    # bead setup
    Rad = 1.0                             # bead radius
    vyb = 0.0                             # y-velocity
    vxb = 0.0                             # x-velocity
    B = np.array([3.95, 1.1, 3.95, 1.1])  # bead

    # force initialisation
    Fby = -9.81
    F_Cy = np.array([0.0])
    jj = np.array([], dtype=int)

    if plot:
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure(99)

    # ----------------------------------------------------------------------
    #                              Main loop
    # ----------------------------------------------------------------------
    for n in range(1, nsteps + 1):
        # bead motion
        B[3] = B[1]                               # store old y-solution
        vyb0 = vyb
        vyb = Fby * dt + vyb
        B[1] = vyb * dt + B[1]                    # y-position of bead

        Fby_pr = (vyb - vyb0) / dt
        B[2] = B[0]                               # store old x-solution
        B[0] = vxb * dt + B[0]
        vxb = (B[0] - B[2]) / dt                  # x-velocity of bead

        # integrate the string (4th-order Runge-Kutta)
        u_old = u
        u = Runge_Kutta_mem(u, t, T, F_Cy, h1, h2, a, dt, jj)

        # detect events
        jj0 = jj
        u, d, Lpt, Rpt, ML, MR, indx_L, indx_R, jj = event_fun(u, B, x, Rad)

        # reaction force
        F0_Cy = F_Cy
        F_Tx, F_Ty, F_Cx, F_Cy = force_fun(a, Rad, B[0], B[1], jj0, jj, x, u, t, T,
                                           h1, h2, dt, F0_Cy, verbose=verbose)

        # correct the system
        u, B, F_Cy, Fby, vyb = correct_mem(u, u_old, B, t, T, F_Cy, F_Ty, Fby_pr,
                                           F0_Cy, vyb, h1, h2, a, dt, jj)

        wave = u[1]
        if record is not None:
            record(n, u, B, vyb, jj)

        if plot and n % plot_every == 0:
            xunit, yunit = circle(B[0], B[1], Rad, 100)
            fig.clf()
            ax = fig.add_subplot(111)
            ax.plot(x, wave, "r", linewidth=2)
            ax.plot(xunit, yunit, "k.", linewidth=2)
            ax.set_xlabel("x coordinate")
            ax.set_ylabel("u(x,t)")
            ax.set_xlim(0, 10)
            ax.set_ylim(-5, 5)
            ax.set_box_aspect(1)
            plt.pause(0.001)

    return x, u, B, vyb


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nsteps", type=int, default=100000)
    p.add_argument("--plot-every", type=int, default=1,
                   help="redraw every this many steps (MATLAB script: 1)")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--verbose", action="store_true",
                   help="print sum(r_proj_y)/2 each step, as the MATLAB script did")
    args = p.parse_args()
    x, u, B, vyb = simulate(args.nsteps, plot=not args.no_plot,
                            plot_every=args.plot_every, verbose=args.verbose)
    print("final bead position:", B[:2], " bead y-velocity:", vyb)
    if not args.no_plot:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()
