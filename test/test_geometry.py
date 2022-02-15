import numpy as np

from zhu import Axis, SymAxis
from zhu import Point
from zhu import Scaler
from zhu import Vector

from zhu.tools import close, close_np

from zhu import EPS


def test_point():
    x1, x2 = 0.0, 1.0
    y1, y2 = 0.0, 0.5
    x1_, y1_ = x1 + EPS/2, y1 - EPS/2
    p1 = Point(x1 + y1 * 1j)
    p1_ = Point(x1_ + y1_ * 1j)
    p2 = Point(x2 + y1 * 1j)
    p3 = Point(x2 + y2 * 1j)
    assert close(p2.x, x2)
    assert close(p2.y, y1)
    assert p1 < p2 and p2 < p3 and p1 < p3
    assert p1 == p1_


def test_vector():
    x1, x2 = 0.0, 1.0
    y1, y2 = 0.5, 1.0
    x1_, y1_ = x1 + EPS, y1 - EPS
    p1 = Point(x1 + y1 * 1j)
    p1_ = Point(x1_ + y1_ * 1j)
    p2 = Point(x1 + y2 * 1j)
    p3 = Point(x2 + y2 * 1j)
    v1 = Vector(p1)
    v1_ = Vector(p1_)
    v2 = Vector(p2)
    v3 = Vector(p3)
    print(f'v3 = {v3}')
    print(f'v2 = {v2}')
    print(f'v1 = {v1}')
    print(f'v1_ = {v1_}')
    assert v1.collinear(v2) and v2.collinear(v1)
    assert v1.collinear(v1_) and v2.collinear(v1_)
    assert not v1.collinear(v3) and not v3.collinear(v1)


def test_axis_basic():
    z1 = 0 + 0j
    z2 = 1 + 0j
    z3 = 0 + 1j
    z_vec = Vector(Point(z2-z1))
    print('z_vec:', z_vec)
    line = Axis(Point(z1), Point(z2))
    print('line:', line)
    line2 = Axis(Point(z1), Point(z3))
    print('line2:', line2)
    u = [z3]
    vec = line.Vec
    print('vec', vec)
    assert close(line.z1, z1)
    assert close(line.z2, z2)
    assert vec.collinear(z_vec)
    assert close(abs(vec.z), 1.0)
    assert line.intersection(line) is None
    assert line.intersection(line2) == Point(z1)
    assert close_np(line.distance(u), [1.0])
    assert line.nearest_ind(u) == 0


def test_axis_vertexes():
    u = [1.0 + 1.0j,
         0.0 + 5.0j,
         2.0 + 7.0j,
         3.0 + 6.0j]
    z1, z2 = Point(u[1]), Point(u[3])
    line = Axis(z1, z2)
    print('line:', line)
    i1, i2 = line.vertexes_ind(u)
    print(i1, i2)
    p1, p2 = line.vertexes(u)
    print(p1, p2)
    assert p1 == z1 and p2 == z2


def test_axis_limit():
    x1, x2 = 0.0, 10.0
    y1, y2 = 0.0, 10.0
    small = Point(x1 + y1 * 1j)
    big = Point(x2 + y2 * 1j)
    points = [Point(x1 + y1 * 1j),
              Point(x1 + y2 * 1j),
              Point(x2 + y2 * 1j),
              Point(x2 + y1 * 1j)]
    for i, p1 in enumerate(points):
        for p2 in points[i+1:]:
            line = Axis(p1, p2)
            s1, s2 = line.limit(small, big)
            assert line._p1 == s1 and line._p2 == s2


def test_symaxis():
    z1, z2 = 0+0j, 1+1j
    q = 5.0
    s = SymAxis(Point(z1), Point(z2), q, None)
    print('s:', s)
    assert s.q == q


def test_scaler():
    N = 10
    u = np.random.rand(N) + 1j * np.random.rand(N)
    s = Scaler(u)
    print('s:', s)
    v = s.predict(u)
    w = s.inverse(v)
    assert close(np.min(np.real(v)), 0.0)
    assert close(np.min(np.imag(v)), 0.0)
    assert close(np.max(np.abs(v)), 1.0)
    assert close_np(u, w)
