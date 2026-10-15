"""Analytic (Bessel) modes of the string with a linear tension gradient.

For T(x) = a (x - L/2) + b the eigenvalue problem -(T phi')' = lam * phi
becomes, with T itself as the variable, T phi_TT + phi_T + (lam / a^2) phi = 0,
whose solutions are J0 and Y0 of the argument

    z(x) = 2 sqrt(lam * T(x)) / |a|.

With phi(0) = phi(L) = 0 the eigenvalues are the roots of

    J0(z0) Y0(zL) - J0(zL) Y0(z0) = 0,     z0 = z(0), zL = z(L),

and the modes are phi(x) = J0(z(x)) Y0(z0) - Y0(z(x)) J0(z0)
(thesis Ch. 2-3).  For a = 0 the string is uniform and the modes are sines.

These are the modes of the continuous string.  The simulation uses the modes
of its own finite-difference operator (string_model.TensionString.eigen);
run_bessel_check.py compares the two.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, y0

SMALL_SLOPE = 1e-6


def _z(lam, T, a):
    return 2.0 * np.sqrt(lam * T) / abs(a)


def characteristic(lam, a, L=10.0, b=1.0):
    """Function whose roots in lam are the eigenvalues."""
    z0, zL = _z(lam, b - 0.5 * a * L, a), _z(lam, b + 0.5 * a * L, a)
    return j0(z0) * y0(zL) - j0(zL) * y0(z0)


def eigenvalues(a, nmodes=6, L=10.0, b=1.0):
    """The lowest ``nmodes`` eigenvalues lam_n(a), n = 1..nmodes."""
    if abs(a) < SMALL_SLOPE:
        return b * (np.arange(1, nmodes + 1) * np.pi / L) ** 2
    if abs(a) >= 2.0 * b / L:
        raise ValueError("|a| must stay below 2b/L")
    # scan upward for sign changes; spacing well below the gap between modes
    lam_scale = b * (np.pi / L) ** 2
    grid = lam_scale * np.linspace(0.02, (nmodes + 1.5) ** 2, 60 * (nmodes + 2))
    f = characteristic(grid, a, L, b)
    roots = []
    for i in np.flatnonzero(np.sign(f[1:]) != np.sign(f[:-1])):
        roots.append(brentq(characteristic, grid[i], grid[i + 1], args=(a, L, b), xtol=1e-15, rtol=1e-14))
        if len(roots) == nmodes:
            break
    if len(roots) < nmodes:
        raise RuntimeError("could not bracket all requested eigenvalues")
    return np.array(roots)


def eigenvalue(a, mode, L=10.0, b=1.0):
    return eigenvalues(a, mode, L, b)[mode - 1]


def mode_shape(x, a, mode, L=10.0, b=1.0):
    """Mode shape on the points x, normalised to unit L2 norm (trapezoid),
    first lobe positive."""
    x = np.asarray(x, dtype=float)
    lam = eigenvalue(a, mode, L, b)
    if abs(a) < SMALL_SLOPE:
        phi = np.sin(mode * np.pi * x / L)
    else:
        z = _z(lam, a * (x - 0.5 * L) + b, a)
        z0 = _z(lam, b - 0.5 * a * L, a)
        phi = j0(z) * y0(z0) - y0(z) * j0(z0)
    phi = phi / np.sqrt(np.trapezoid(phi * phi, x))
    i = int(np.argmax(np.abs(phi[: max(2, len(x) // (2 * mode))])))
    return phi * np.sign(phi[i])


def nodes(a, mode, L=10.0, b=1.0):
    """Interior node positions of the mode (roots of the analytic shape)."""
    if abs(a) < SMALL_SLOPE:
        return L * np.arange(1, mode) / mode
    lam = eigenvalue(a, mode, L, b)
    z0 = _z(lam, b - 0.5 * a * L, a)

    def phi(x):
        z = _z(lam, a * (x - 0.5 * L) + b, a)
        return j0(z) * y0(z0) - y0(z) * j0(z0)

    xs = np.linspace(0.0, L, 400 * mode + 1)[1:-1]
    f = phi(xs)
    return np.array([brentq(phi, xs[i], xs[i + 1], xtol=1e-14)
                     for i in np.flatnonzero(np.sign(f[1:]) != np.sign(f[:-1]))])


def resonance(a, mode, beta, L=10.0, b=1.0):
    """Frequency of largest steady-state amplitude, sqrt(lam - beta^2 / 2)."""
    return float(np.sqrt(eigenvalue(a, mode, L, b) - 0.5 * beta ** 2))


def resonance_table(mode, beta, L=10.0, b=1.0, samples=401):
    """Fast function w_res(a) interpolated from the analytic eigenvalues."""
    lim = 0.97 * 2.0 * b / L
    a_grid = np.linspace(-lim, lim, samples)
    w_grid = np.array([resonance(a, mode, beta, L, b) for a in a_grid])
    return lambda a: float(np.interp(a, a_grid, w_grid))
