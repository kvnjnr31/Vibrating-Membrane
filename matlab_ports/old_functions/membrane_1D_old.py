"""Integrate the string in 1D with an event-driven bead (first version).

Port of old_functions/membrane_1D_old.m.

PORT NOTES
* The MATLAB script calls an older, 7-argument Runge_Kutta_mem(u, t, T, h1,
  h2, a, dt) that is not in the archive.  The current integrator from the
  parent folder is used with no contact force.
* h1 = 1/dx is kept as written.  The current scripts use h1 = 1/(2*dx);
  with the current right-hand side, 1/dx doubles the a*u_x term.

Run:
    python membrane_1D_old.py
    python membrane_1D_old.py --no-plot
"""
import argparse
import pathlib
import sys

import numpy as np

# the integrator lives in the parent folder (appended so this folder wins)
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from Runge_Kutta_mem import Runge_Kutta_mem               # noqa: E402
from Bead_Membrane_Dynamics_old import Bead_Membrane_Dynamics  # noqa: E402


def simulate(nsteps=210, plot=True, verbose=True, record=None):
    # simulation parameters
    points = 100                          # number of grid points
    L = 10.0                              # length of the string
    dx = L / points                       # spatial discretisation
    h2 = 1.0 / (dx * dx)
    h1 = 1.0 / dx
    dt = 0.001                            # time step
    a = 1.0                               # tension slope

    # tension
    x = np.linspace(0.0, L, points)
    T = a * x + 1.0

    # string setup
    u = np.zeros((2, points))             # row 0: velocity, row 1: amplitude
    u[1, 39] = 0.1                        # initial condition (MATLAB u(2,40))
    t = 0.0                               # initial time

    # bead setup: B = [x, y, x0, y0]
    R = 1e-2                              # bead radius
    vyb = 0.0                             # y-velocity
    vxb = -1.0                            # x-velocity
    B = np.array([3.95, 0.12, 3.95, 0.12])

    # system state
    state_new = 0

    no_force = np.array([])
    no_contact = np.array([], dtype=int)

    if plot:
        import matplotlib.pyplot as plt
        plt.ion()

    def draw(fignum, wave):
        fig = plt.figure(fignum)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(x, wave, "ko-", linewidth=2)
        ax.plot(B[0], B[1], "ro", linewidth=2)
        ax.plot([B[0], B[0]], [B[1], B[1] - R], "b", linewidth=2)
        ax.set_xlabel("x coordinate")
        ax.set_ylabel("u(x,t)")
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.25, 0.25)
        ax.set_box_aspect(1)

    # ----------------------------------------------------------------------
    #                              Main loop
    # ----------------------------------------------------------------------
    for n in range(1, nsteps + 1):
        # bead motion
        B[3] = B[1]                                   # store old y-solution
        B[1] = -9.81 * (dt ** 2) + vyb * dt + B[1]
        vyb = (B[1] - B[3]) / dt                      # y-velocity of bead

        B[2] = B[0]                                   # store old x-solution
        B[0] = vxb * dt + B[0]
        vxb = (B[0] - B[2]) / dt                      # x-velocity of bead

        u_old = u                                     # previous solution

        # integrate the string (4th-order Runge-Kutta)
        u = Runge_Kutta_mem(u, t, T, no_force, h1, h2, a, dt, no_contact)

        if plot:
            draw(1, u[1])

        # wave correction
        u, B, state_new, t = Bead_Membrane_Dynamics(u, u_old, B, state_new, x, R, t, dt,
                                                    plot=plot, verbose=verbose)
        if record is not None:
            record(n, u, B, vyb, state_new)

        if plot:
            draw(2, u[1])                             # new solution
            plt.pause(0.001)

    return x, u, B, state_new


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nsteps", type=int, default=210)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    x, u, B, state = simulate(args.nsteps, plot=not args.no_plot)
    print("final bead position:", B[:2], " contact state:", state)
    if not args.no_plot:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()
