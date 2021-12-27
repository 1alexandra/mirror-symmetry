# Source:
# https://stackoverflow.com/questions/12643079/bézier-curve-fitting-with-scipy

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import cv2


class BezierCurve:
    def __init__(self, points, degree=3):
        self.origin = np.array(points)
        self.degree = degree
        assert degree > 1
        assert len(points) > degree
        self._key_points = None

        self.draw_kwargs = {
            'color': (255, 0, 255),
            'radius': 10,
            'thickness': -1,
        }

    def bmatrix(self, ts):
        ts = np.array(ts).reshape((-1, 1))
        n = self.degree
        k = np.arange(n + 1)
        m = comb(n, k) * ts ** k * (1 - ts) ** (n - k)
        return np.matrix(m)

    @staticmethod
    def least_square_fit(points, m):
        m_ = np.linalg.pinv(m)
        return np.array(m_ * points)

    @property
    def key_points(self):
        if self._key_points is None:
            ts = np.linspace(0, 1, len(self.origin))
            m = self.bmatrix(ts)
            final = self.least_square_fit(self.origin, m)
            for i in (0, -1):
                final[i] = self.origin[i]
            self._key_points = final
        return self._key_points

    def bezier_curve(self, n=50):
        t = np.linspace(0.0, 1.0, n)
        pmatrix = self.bmatrix(t)
        return np.array(pmatrix * self.key_points)

    def bezier_curve_t(self, t):
        pmatrix = self.bmatrix([t])
        return np.array(pmatrix * self.key_points)

    def bezier_curve_length(self, n=100):
        xvals, yvals = self.bezier_curve(n).T
        xvals_shift = np.roll(xvals, -1)[:-1]
        yvals_shift = np.roll(yvals, -1)[:-1]
        xdelta = xvals[:-1] - xvals_shift
        ydelta = yvals[:-1] - yvals_shift
        return (((xdelta ** 2) + (ydelta ** 2)) ** 0.5).sum()

    def plot(self, n=50):
        xpoints, ypoints = self.origin.T
        plt.plot(xpoints, ypoints, "ro", label='Original Points')
        x_val, y_val = self.key_points.T
        plt.plot(x_val, y_val, 'k--o', label='Control Points')
        xvals, yvals = (self.bezier_curve(n=n)).T
        plt.plot(xvals, yvals, 'b-', label='B Curve')
        plt.legend()
        plt.show()

    def draw(self, board, n=50):
        for x, y in (self.bezier_curve(n=n)):
            coord = (int(x), int(y))
            board = cv2.circle(
                board, coord, **self.draw_kwargs)
            if type(board) is cv2.UMat:
                board = board.get()
        return board
