"""Damped, forced string with a linear tension gradient (thesis Ch. 2-6).

    u_tt = (T(x) u_x)_x - beta * u_t + f(x, t),      u(0) = u(L) = 0
    T(x) = a * (x - L/2) + b,                        |a| < 2b/L

The discretisation is conservative (tension at the half points), so the
discrete operator is symmetric and its eigenmodes are the exact modes of the
simulated system.  Resonant frequencies are taken from those eigenmodes, so
the drive is on resonance for the grid actually being integrated.  The
analytic Bessel modes are in bessel.py; at 301 points the two agree to
0.003 linewidths for mode 3 (run_bessel_check.py).
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal


class TensionString:
    ndim = 1

    def __init__(self, L=10.0, b=1.0, N=301, beta=0.015):
        self.L, self.b, self.N, self.beta = float(L), float(b), int(N), float(beta)
        self.x = np.linspace(0.0, self.L, self.N)
        self.dx = self.x[1] - self.x[0]
        self.xh = 0.5 * (self.x[1:] + self.x[:-1])       # half points
        self.shape = (self.N,)
        self.weight = self.dx                            # quadrature weight
        self.a_limit = 2.0 * self.b / self.L             # tension reaches zero here

    # ------------------------------------------------------------ tension
    def tension(self, a, x=None):
        x = self.x if x is None else x
        return a * (x - 0.5 * self.L) + self.b

    def _check(self, a):
        if abs(a) >= self.a_limit:
            raise ValueError(f"|a| = {abs(a):.4g} must stay below 2b/L = {self.a_limit:.4g}")

    # ------------------------------------------------------------ dynamics
    def accel(self, u, v, a, f):
        """u_tt for displacement u, velocity v, slope a, forcing f (array or 0)."""
        Th = self.tension(a, self.xh)
        flux = Th * (u[1:] - u[:-1]) / self.dx
        acc = np.zeros_like(u)
        acc[1:-1] = (flux[1:] - flux[:-1]) / self.dx - self.beta * v[1:-1]
        acc[1:-1] += f[1:-1] if np.ndim(f) else f
        return acc

    def energy(self, u, v, a):
        Th = self.tension(a, self.xh)
        return 0.5 * self.dx * (np.sum(v * v) + np.sum(Th * ((u[1:] - u[:-1]) / self.dx) ** 2))

    def uniform_shape(self):
        return np.ones(self.N)

    # ------------------------------------------------------------ modes
    def eigen(self, a, nmodes=6):
        """Lowest eigenpairs of -(T phi')' = lam * phi.

        Returns lam (nmodes,) and phi (nmodes, N), normalised so that
        sum(phi^2) * dx = 1 with the first lobe positive.
        """
        self._check(a)
        Th = self.tension(a, self.xh)
        d = (Th[:-1] + Th[1:]) / self.dx ** 2
        e = -Th[1:-1] / self.dx ** 2
        lam, vec = eigh_tridiagonal(d, e, select="i", select_range=(0, nmodes - 1))
        phi = np.zeros((nmodes, self.N))
        phi[:, 1:-1] = vec.T
        phi /= np.sqrt(np.sum(phi ** 2, axis=1, keepdims=True) * self.dx)
        phi *= np.sign(phi[:, 1])[:, None]
        return lam, phi

    def eigenvalue(self, a, mode):
        return self.eigen(a, mode)[0][mode - 1]

    def mode_shape(self, a, mode):
        return self.eigen(a, mode)[1][mode - 1]

    def resonance(self, a, mode):
        """Frequency of maximum steady-state amplitude, sqrt(lam - beta^2/2)."""
        return float(np.sqrt(self.eigenvalue(a, mode) - 0.5 * self.beta ** 2))

    def resonance_table(self, mode, a_min=None, a_max=None, samples=401):
        """Return a fast function w_res(a), interpolated from a table."""
        lim = 0.97 * self.a_limit
        a_min = -lim if a_min is None else a_min
        a_max = lim if a_max is None else a_max
        a_grid = np.linspace(a_min, a_max, samples)
        w_grid = np.array([self.resonance(a, mode) for a in a_grid])
        return lambda a: float(np.interp(a, a_grid, w_grid))

    def modal_basis(self, a, nmodes=40):
        """(lam, phi) for the lowest modes, used for steady-state predictions."""
        return self.eigen(a, nmodes)

    def inner(self, p, q):
        return float(np.sum(p * q) * self.dx)

    # ------------------------------------------------------------ nodes
    def nodes(self, profile, mode=None):
        """Interior zero crossings of a profile (linear interpolation)."""
        p = np.asarray(profile, dtype=float)
        i = np.flatnonzero(p[1:-2] * p[2:-1] < 0) + 1
        return self.x[i] + self.dx * p[i] / (p[i] - p[i + 1])

    def mode_nodes(self, a, mode):
        return self.nodes(self.mode_shape(a, mode))

    def still_points(self, a, mode):
        """Places that do not move: the nodes plus the two fixed ends.
        Returned as a 1-tuple (one array per horizontal coordinate)."""
        return (np.concatenate(([0.0], self.mode_nodes(a, mode), [self.L])),)
