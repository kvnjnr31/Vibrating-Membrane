"""Closed-form modes of the membrane with tension T(x) = a (x - Lx/2) + b.

Writing u = phi(x) sin(k y), k = m pi / Ly, the x-problem is

    (T phi')' - k^2 T phi + lambda phi = 0,   phi(0) = phi(Lx) = 0.

With s = T(x), kappa = k/|a| and z = 2 kappa s, phi = exp(-z/2) w(z) and w
obeys Kummer's equation  z w'' + (1 - z) w' - alpha w = 0  with

    alpha = 1/2 - lambda / (2 |a| k).

So phi = exp(-z/2) [ c1 M(alpha, 1, z) + c2 U(alpha, 1, z) ].  The Bessel
functions of the string are the limit k -> 0.  Eigenvalues are the roots of

    M(alpha,1,z0) U(alpha,1,zL) - M(alpha,1,zL) U(alpha,1,z0) = 0.

Used only to check the grid modes; needs mpmath.
"""
import numpy as np
import mpmath as mp


def _det(lam, a, k, T0, TL):
    kap = k / abs(a)
    al = mp.mpf(0.5) - lam / (2 * abs(a) * k)
    z0, zL = 2 * kap * T0, 2 * kap * TL
    return (mp.hyp1f1(al, 1, z0) * mp.hyperu(al, 1, zL)
            - mp.hyp1f1(al, 1, zL) * mp.hyperu(al, 1, z0))


def eigenvalue(a, mode, Lx=10.0, Ly=7.0, b=1.0, guess=None):
    """Eigenvalue of mode (n, m); ``guess`` should be the grid value."""
    n, m = mode
    k = m * np.pi / Ly
    T0, TL = b - a * Lx / 2, b + a * Lx / 2
    mp.mp.dps = 30
    f = lambda lam: _det(lam, a, k, T0, TL)          # noqa: E731
    return float(mp.findroot(f, (guess * 0.999, guess * 1.001), solver="secant", tol=1e-12, verify=False))
