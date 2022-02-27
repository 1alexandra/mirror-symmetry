# Source:
# https://stackoverflow.com/questions/12643079/bézier-curve-fitting-with-scipy

import numpy as np
from scipy.special import comb
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import cv2

from zhu.vertex import Vertex


class BezierCurve:
    def __init__(self, points, degree=3):
        self.origin = np.array(points)
        self.degree = degree
        self.approx_items = 50

        assert degree > 1
        assert len(points) > degree
        self._key_points = None
        self._approx = None
        self._length = None

        self.draw_kwargs = {
            'color': (255, 0, 255),
            'radius': 5,
            'thickness': -1,
        }

    def _bmatrix(self, ts):
        ts = np.array(ts).reshape((-1, 1))
        n = self.degree
        k = np.arange(n + 1)
        m = comb(n, k) * ts ** k * (1 - ts) ** (n - k)
        return np.matrix(m)

    def _dmatrix(self, ts):
        ts = np.array(ts).reshape((-1, 1))
        n = self.degree
        k = np.arange(n + 1)
        m = comb(n, k) * (
            k * ts ** (k - 1) * (1 - ts) ** (n - k)
            - (n - k) * (1 - ts) ** (n - k - 1) * ts ** k
        )
        return np.matrix(m)

    @property
    def key_points(self):
        if self._key_points is None:
            ts = np.linspace(0, 1, len(self.origin))
            bm = self._bmatrix(ts)
            m = np.linalg.pinv(bm)
            final = np.array(m * self.origin)
            for i in (0, -1):
                final[i] = self.origin[i]
            self._key_points = final
        return self._key_points

    @property
    def approx(self):
        if self._approx is None:
            t = np.linspace(0.0, 1.0, self.approx_items)
            pmatrix = self._bmatrix(t)
            self._approx = np.array(pmatrix * self.key_points)
        return self._approx

    def value(self, t):
        pmatrix = self._bmatrix([t])
        return np.array(pmatrix * self.key_points).ravel()

    def grad(self, t):
        dmatrix = self._dmatrix([t])
        return np.array(dmatrix * self.key_points).ravel()

    def _tangent_scalar_product(self, t, vertex):
        x, y = vertex.x, vertex.y
        x_0, y_0 = self.value(t)
        x_1, y_1 = self.grad(t)
        return (x_0 - x) * x_1 + (y_0 - y) * y_1

    def _tangent_vector_product_z(self, t, vertex):
        x, y = vertex.x, vertex.y
        x_0, y_0 = self.value(t)
        x_1, y_1 = self.grad(t)
        return (x_0 - x) * y_1 - (y_0 - y) * x_1

    def coord(self, vertex):
        result = fsolve(self._tangent_scalar_product, 0.5, (vertex))
        t = result[0]
        curve_point = Vertex(*self.value(t))
        x = t * self.length
        sign = self._tangent_vector_product_z(t, vertex)
        y = curve_point.distance(vertex) * np.sign(sign)
        return (x, y)

    @property
    def length(self):
        if self._length is None:
            xvals, yvals = self.approx.T
            xvals_shift = np.roll(xvals, -1)[:-1]
            yvals_shift = np.roll(yvals, -1)[:-1]
            xdelta = xvals[:-1] - xvals_shift
            ydelta = yvals[:-1] - yvals_shift
            self._length = (((xdelta ** 2) + (ydelta ** 2)) ** 0.5).sum()
        return self._length

    def plot(self, n=50):
        xpoints, ypoints = self.origin.T
        plt.plot(xpoints, ypoints, "ro", label='Original Points')
        x_val, y_val = self.key_points.T
        plt.plot(x_val, y_val, 'k--o', label='Control Points')
        xvals, yvals = self.approx.T
        plt.plot(xvals, yvals, 'b-', label='B Curve')
        plt.legend()
        plt.show()

    def draw(self, board, n=50):
        for x, y in self.approx:
            coord = (int(x), int(y))
            board = cv2.circle(
                board, coord, **self.draw_kwargs)
            if type(board) is cv2.UMat:
                board = board.get()
        return board
