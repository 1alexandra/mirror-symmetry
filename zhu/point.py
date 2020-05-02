from functools import total_ordering
import numpy as np

from zhu import EPS
from zhu.tools import close


@total_ordering
class Point:
    """Complex point."""

    def __init__(self, z: complex):
        self.z = z

    x = property()
    y = property()

    @x.getter
    def x(self):
        return np.real(self.z)

    @y.getter
    def y(self):
        return np.imag(self.z)

    def __lt__(self, other):
        return (self.y < other.y) or (self.y == other.y and self.x < other.x)

    def __eq__(self, other):
        return close(self.z, other.z, 2 * EPS)

    def __str__(self):
        return f'Point {self.z}'
