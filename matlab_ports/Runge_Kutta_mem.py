"""Classical 4th-order Runge-Kutta step for the string.  Port of Runge_Kutta_mem.m."""
from centdiftt_mem import centdiftt_mem


def Runge_Kutta_mem(u, t, T, F_Cy, h1, h2, a, dt, jj):
    k1 = dt * centdiftt_mem(u, t, T, F_Cy, h1, h2, a, dt, jj)
    k2 = dt * centdiftt_mem(u + 0.5 * k1, t + 0.5 * dt, T, F_Cy, h1, h2, a, dt, jj)
    k3 = dt * centdiftt_mem(u + 0.5 * k2, t + 0.5 * dt, T, F_Cy, h1, h2, a, dt, jj)
    k4 = dt * centdiftt_mem(u + k3, t + dt, T, F_Cy, h1, h2, a, dt, jj)

    u = u + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Apply boundary conditions (fixed ends).
    u[1, 0] = 0.0
    u[1, -1] = 0.0
    u[0, 0] = 0.0
    u[0, -1] = 0.0
    return u
