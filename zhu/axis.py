import numpy as np

from zhu import Point
from zhu import Vector
from zhu.tools import between, unique, index_neibs


class Axis:
    def __init__(self, p1: Point, p2: Point):
        self._p1 = p1
        self._p2 = p2
        if p1 > p2:
            self._p1, self._p2 = self._p2, self._p1

    z1 = property()
    z2 = property()
    Vec = property()

    @z1.getter
    def z1(self):
        return self._p1.z

    @z1.setter
    def z1(self, value: complex):
        self._p1 = Point(value)

    @z2.getter
    def z2(self):
        return self._p2.z

    @z2.setter
    def z2(self, value: complex):
        self._p2 = Point(value)

    @Vec.getter
    def Vec(self):
        v = self.z1 - self.z2
        if np.abs(v) == 0:
            ans = 0.0 + 0.0j
        else:
            ans = v / np.abs(v)
        return Vector(Point(ans))

    def __str__(self):
        return '\n'.join([
            '======',
            f'Axis: {self._p1} --- {self._p2}',
            f'z1={self.z1}, z2={self.z2}, vec={self.Vec}',
            '======'
        ])

    def distance(self, u):
        a = (self._p2.y - self._p1.y) * np.real(u)
        b = - (self._p2.x - self._p1.x) * np.imag(u)
        c = self._p1.y * self._p2.x - self._p2.y * self._p1.x
        return np.abs(a + b + c)

    def intersection(self, other):
        v1 = self.Vec
        v2 = other.Vec
        if v1.collinear(v2):
            return None
        a = np.array([[v1.x, -v2.x],
                      [v1.y, -v2.y]])
        b = np.array([other._p1.x - self._p1.x,
                      other._p1.y - self._p1.y])
        solution = np.linalg.solve(a, b)
        return Point(self.z1 + solution[0] * v1.z)

    def nearest_ind(self, u):
        return np.argmin(self.distance(u))

    def vertexes_ind(self, u):
        i1 = self.nearest_ind(u)
        mask = np.ones(len(u), dtype=bool)
        mask[index_neibs(u, i1, 0.5)] = False
        u_other = np.array(u, dtype=complex)[mask]
        i2_ = self.nearest_ind(u_other)
        i2 = np.arange(len(u))[mask][i2_]
        if i1 > i2:
            i1, i2 = i2, i1
        return i1, i2

    def vertexes(self, u):
        i1, i2 = self.vertexes_ind(u)
        u1, u2 = Point(u[i1]), Point(u[i2])
        if u1 > u2:
            u1, u2 = u2, u1
        return u1, u2

    def limit(self, small: Point, big: Point):
        ps = []
        lines = [
            Axis(Point(small.x + 0.0j), Point(small.x + 1.0j)),
            Axis(Point(0.0 + small.y * 1j), Point(1.0 + small.y * 1j)),
            Axis(Point(big.x + 0.0j), Point(big.x + 1.0j)),
            Axis(Point(0.0 + big.y * 1j), Point(1.0 + big.y * 1j)),
        ]
        for line in lines:
            s = self.intersection(line)
            if (s is not None
                    and between(small.x, s.x, big.x)
                    and between(small.y, s.y, big.y)):
                ps.append(s)
        ps = unique(ps)
        if len(ps) != 2:
            raise Exception('Axis limit error')
        s1, s2 = ps
        if s1 > s2:
            s1, s2 = s2, s1
        return s1, s2
