"""Sample a vibrating surface (height, velocity, slope) at grain positions.

The wave solver keeps u and u_t on a grid; grains are grid free, so the
fields are interpolated linearly to wherever each grain is.
"""
import numpy as np


class StringSurface:
    def __init__(self, model):
        self.x, self.dx = model.x, model.dx
        self.lo = np.array([model.x[0]])
        self.hi = np.array([model.x[-1]])

    def update(self, u, v):
        self.u, self.v = u, v
        self.ux = np.gradient(u, self.dx)

    def sample(self, X):
        """X has shape (n, 1).  Returns height (n,), velocity (n,), slope (n, 1)."""
        xp = X[:, 0]
        return (np.interp(xp, self.x, self.u), np.interp(xp, self.x, self.v),
                np.interp(xp, self.x, self.ux)[:, None])


class MembraneSurface:
    def __init__(self, model):
        self.x, self.y, self.dx, self.dy = model.x, model.y, model.dx, model.dy
        self.lo = np.array([model.x[0], model.y[0]])
        self.hi = np.array([model.x[-1], model.y[-1]])

    def update(self, u, v):
        self.u, self.v = u, v
        self.ux, self.uy = np.gradient(u, self.dx, self.dy)

    def _weights(self, X):
        fx = np.clip((X[:, 0] - self.x[0]) / self.dx, 0, len(self.x) - 1 - 1e-9)
        fy = np.clip((X[:, 1] - self.y[0]) / self.dy, 0, len(self.y) - 1 - 1e-9)
        i, j = fx.astype(int), fy.astype(int)
        return i, j, fx - i, fy - j

    @staticmethod
    def _bilinear(F, i, j, s, t):
        return ((1 - s) * (1 - t) * F[i, j] + s * (1 - t) * F[i + 1, j]
                + (1 - s) * t * F[i, j + 1] + s * t * F[i + 1, j + 1])

    def sample(self, X):
        """X has shape (n, 2).  Returns height (n,), velocity (n,), slope (n, 2)."""
        i, j, s, t = self._weights(X)
        b = self._bilinear
        grad = np.stack([b(self.ux, i, j, s, t), b(self.uy, i, j, s, t)], axis=1)
        return b(self.u, i, j, s, t), b(self.v, i, j, s, t), grad
