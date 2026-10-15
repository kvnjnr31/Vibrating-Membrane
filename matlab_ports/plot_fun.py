"""Plot the string-bead system with its matching points.  Port of plot_fun.m.

Indices ``indx_L`` / ``indx_R`` are 0-based.  Pass ``None`` (MATLAB ``[]``)
for the contact quantities to plot the bare string.
"""
import numpy as np
import matplotlib.pyplot as plt

from circle import circle


def _empty(v):
    return v is None or np.size(v) == 0


def plot_fun(Cl_x, Cl_y, Cr_x, Cr_y, ML, MR, x, u, B, F_Tx, F_Ty, indx_L, indx_R, Rad):
    xunit, yunit = circle(B[0], B[1], Rad, 50)

    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(xunit, yunit, "k", linewidth=2)

    if not _empty(Cl_x):
        ax.plot(Cl_x, Cl_y, "r*", Cr_x, Cr_y, "r*", linewidth=2)
        ax.plot(x, u[1], "ko", linewidth=2)
        ax.plot(np.append(x[: indx_L + 1], Cl_x), np.append(u[1, : indx_L + 1], Cl_y),
                "ko-", linewidth=2)
        ax.plot(np.append(Cr_x, x[indx_R:]), np.append(Cr_y, u[1, indx_R:]),
                "ko-", linewidth=2)
        ax.plot(ML[0], ML[1], "bo", MR[0], MR[1], "bo", linewidth=2)
        ax.plot([B[0], B[0] + F_Tx], [B[1], B[1] + F_Ty], "k", linewidth=2)
    else:
        # MATLAB plot(x, u) draws both rows of u (velocity and displacement)
        ax.plot(x, np.asarray(u).T, "ko-", linewidth=2)

    ax.set_xlim(0, 10)
    ax.set_ylim(-5, 5)
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=14, width=2)
    return ax
