"""Damped, forced rectangular membrane with a tension gradient along x.

    u_tt = d/dx( T(x) u_x ) + T(x) u_yy - beta * u_t + f(x, y, t)
    T(x) = a * (x - Lx/2) + b,        u = 0 on all four edges

The tension is isotropic and depends on x only, so the modes separate:

    u = phi(x) * sin(m pi y / Ly),   -(T phi')' + T * kappa_m^2 * phi = lam * phi

Note that this x-equation is NOT the string equation: the extra term
T(x) * kappa_m^2 * phi means the membrane modes are not products of the
string's Bessel modes.  Modes are labelled (n, m): n-th x-eigenfunction
(n - 1 interior nodal lines x = const) for the m-th sine in y.

Arrays are indexed u[ix, iy].
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal


class TensionMembrane:
    ndim = 2

    def __init__(self, Lx=10.0, Ly=7.0, b=1.0, Nx=101, Ny=71, beta=0.015):
        self.Lx, self.Ly, self.b, self.beta = float(Lx), float(Ly), float(b), float(beta)
        self.Nx, self.Ny = int(Nx), int(Ny)
        self.x = np.linspace(0.0, self.Lx, self.Nx)
        self.y = np.linspace(0.0, self.Ly, self.Ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.xh = 0.5 * (self.x[1:] + self.x[:-1])
        self.shape = (self.Nx, self.Ny)
        self.weight = self.dx * self.dy
        self.a_limit = 2.0 * self.b / self.Lx

    # ------------------------------------------------------------ tension
    def tension(self, a, x=None):
        x = self.x if x is None else x
        return a * (x - 0.5 * self.Lx) + self.b

    def _check(self, a):
        if abs(a) >= self.a_limit:
            raise ValueError(f"|a| = {abs(a):.4g} must stay below 2b/Lx = {self.a_limit:.4g}")

    # ------------------------------------------------------------ dynamics
    def accel(self, u, v, a, f):
        Th = self.tension(a, self.xh)[:, None]
        Tn = self.tension(a)[1:-1, None]
        flux = Th * (u[1:, 1:-1] - u[:-1, 1:-1]) / self.dx
        acc = np.zeros_like(u)
        acc[1:-1, 1:-1] = ((flux[1:] - flux[:-1]) / self.dx
                           + Tn * (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / self.dy ** 2
                           - self.beta * v[1:-1, 1:-1])
        acc[1:-1, 1:-1] += f[1:-1, 1:-1] if np.ndim(f) else f
        return acc

    def energy(self, u, v, a):
        Th = self.tension(a, self.xh)[:, None]
        Tn = self.tension(a)[:, None]
        ux = (u[1:, :] - u[:-1, :]) / self.dx
        uy = (u[:, 1:] - u[:, :-1]) / self.dy
        return 0.5 * self.weight * (np.sum(v * v) + np.sum(Th * ux * ux) + np.sum(Tn * uy * uy))

    def uniform_shape(self):
        return np.ones(self.shape)

    def patch_shape(self, x0, y0, width):
        """Gaussian forcing patch centred at (x0, y0); excites every mode
        that is non-zero there."""
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * width ** 2))

    # ------------------------------------------------------------ modes
    def _kappa2(self, m):
        """Discrete eigenvalue of -d2/dy2 for the m-th sine on the grid."""
        return (4.0 / self.dy ** 2) * np.sin(0.5 * m * np.pi / (self.Ny - 1)) ** 2

    def eigen_x(self, a, m, nmodes=6):
        """Eigenpairs of the x-problem for the m-th sine in y."""
        self._check(a)
        Th = self.tension(a, self.xh)
        Tn = self.tension(a)[1:-1]
        d = (Th[:-1] + Th[1:]) / self.dx ** 2 + Tn * self._kappa2(m)
        e = -Th[1:-1] / self.dx ** 2
        lam, vec = eigh_tridiagonal(d, e, select="i", select_range=(0, nmodes - 1))
        phi = np.zeros((nmodes, self.Nx))
        phi[:, 1:-1] = vec.T
        phi *= np.sign(phi[:, 1])[:, None]
        return lam, phi

    def eigenvalue(self, a, mode):
        n, m = mode
        return self.eigen_x(a, m, n)[0][n - 1]

    def mode_shape(self, a, mode):
        """2D mode shape, normalised so that sum(phi^2) dx dy = 1."""
        n, m = mode
        phi_x = self.eigen_x(a, m, n)[1][n - 1]
        shape = np.outer(phi_x, np.sin(m * np.pi * self.y / self.Ly))
        return shape / np.sqrt(np.sum(shape ** 2) * self.weight)

    def resonance(self, a, mode):
        return float(np.sqrt(self.eigenvalue(a, mode) - 0.5 * self.beta ** 2))

    def load_table(self, path):
        """Use a look-up table written by run_make_table.py for its mode.

        After this, ``resonance_table(mode)`` interpolates the file instead
        of solving the eigenproblem.
        """
        data = np.genfromtxt(path, delimiter=",", names=True, skip_header=1)
        with open(path) as fh:
            head = fh.readline()
        n, m = (int(v) for v in head.split("mode")[1].split()[0].strip("()").split(","))
        if not hasattr(self, "_tables"):
            self._tables = {}
        self._tables[(n, m)] = (data["a"].copy(), data["omega_res"].copy())
        return (n, m)

    def resonance_table(self, mode, a_min=None, a_max=None, samples=201):
        if tuple(mode) in getattr(self, "_tables", {}):
            a_tab, w_tab = self._tables[tuple(mode)]
            return lambda a: float(np.interp(a, a_tab, w_tab))
        lim = 0.97 * self.a_limit
        a_min = -lim if a_min is None else a_min
        a_max = lim if a_max is None else a_max
        a_grid = np.linspace(a_min, a_max, samples)
        w_grid = np.array([self.resonance(a, mode) for a in a_grid])
        return lambda a: float(np.interp(a, a_grid, w_grid))

    def spectrum(self, a, n_max=6, m_max=6):
        """All resonant frequencies up to (n_max, m_max) as {(n, m): w}."""
        out = {}
        for m in range(1, m_max + 1):
            lam = self.eigen_x(a, m, n_max)[0]
            for n in range(1, n_max + 1):
                out[(n, m)] = float(np.sqrt(lam[n - 1] - 0.5 * self.beta ** 2))
        return out

    def nearest_neighbour(self, a, mode, n_max=6, m_max=6):
        """Closest other mode in frequency: returns (mode, frequency gap).

        A gap of a few times beta or less means the drive will excite both
        modes and the pattern will be a mixture."""
        spec = self.spectrum(a, n_max, m_max)
        w0 = spec[tuple(mode)]
        other = min((k for k in spec if k != tuple(mode)), key=lambda k: abs(spec[k] - w0))
        return other, abs(spec[other] - w0)

    def modal_basis(self, a, n_max=10, m_max=9):
        """(lam, phi) for modes up to (n_max, m_max), phi normalised."""
        lams, phis = [], []
        for m in range(1, m_max + 1):
            lam, phx = self.eigen_x(a, m, n_max)
            sy = np.sin(m * np.pi * self.y / self.Ly)
            for n in range(n_max):
                shape = np.outer(phx[n], sy)
                phis.append(shape / np.sqrt(np.sum(shape ** 2) * self.weight))
                lams.append(lam[n])
        return np.array(lams), np.array(phis)

    def inner(self, p, q):
        return float(np.sum(p * q) * self.weight)

    # ------------------------------------------------------------ nodes
    def nodes(self, profile, mode=None):
        """x-positions of the nodal lines, read along the row y = const where
        the profile is strongest."""
        p = np.asarray(profile, dtype=float)
        j = int(np.argmax(np.sum(p * p, axis=0)))
        row = p[:, j]
        i = np.flatnonzero(row[1:-2] * row[2:-1] < 0) + 1
        return self.x[i] + self.dx * row[i] / (row[i] - row[i + 1])

    def mode_nodes(self, a, mode):
        return self.nodes(self.mode_shape(a, mode))

    def still_points(self, a, mode):
        """Lines that do not move, as (x-positions, y-positions): nodal lines
        of the mode plus the clamped edges."""
        m = mode[1]
        xs = np.concatenate(([0.0], self.mode_nodes(a, mode), [self.Lx]))
        ys = self.Ly * np.arange(0, m + 1) / m
        return xs, ys
