import numpy as np

from zhu import MIN_THETA_DIFF
from zhu import Point
from zhu.tools import close


class Vector(Point):
    """Complex vector."""

    def __init__(self, p: Point):
        super().__init__(p)
        self.z = p.z
        self.angle = np.angle(self.z) % np.pi

    def collinear(self, other):
        if close(abs(self.z), 0.0) or close(abs(other.z), 0.0):
            return True
        theta1 = np.angle(self.z) % np.pi
        theta2 = np.angle(other.z) % np.pi
        return close(theta1, theta2, MIN_THETA_DIFF)

    def __str__(self):
        return f'Vector: {self.z}, angle={self.angle}'
