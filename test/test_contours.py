import os

import numpy as np

from zhu import Binarizer
from zhu import Contour
from zhu import Point, Axis, SymAxis
from zhu import FourierDescriptor
from zhu import SymAxisList
from zhu import SymContour
from zhu import SymImage
from zhu import DataFolder

from zhu.tools import close, close_np

from zhu import EPS, MIN_THETA_DIFF
from zhu import MIN_CONTOUR_AREA


def test_binarizer():
    b = Binarizer()
    print('b:', b)
    assert b.gauss == 0
    folder = 'test/files'
    images = ['flies.bmp', 'camel.gif', '50.mask.jpg']
    for img in images:
        image_path = folder + '/' + img
        # assert b.imread_bw(image_path) is not None
        assert b.binarize(image_path) is not None
        res_path = folder + '/results/res_' + img
        b.create_mono_bmp(image_path, res_path)


def test_contour():
    u = [0.+5.j, 1.+1.j, 3.+6.j, 2.+7.j]
    area = 8.5
    per = 3 * 2**0.5 + 17**0.5 + 29**0.5
    mult_coef = 2
    c = Contour(u, mult_coef)
    print('c:', c)
    assert close(c.Area, area)
    assert close(c.Perimeter, per)
    p = c.Pixels
    p_next = np.roll(p, 1)
    assert np.all(np.abs(p - p_next) <= 2**0.5 + EPS)
    ch = c.Convex_hull
    assert close_np(c.origin, ch.origin)
    m = c.Edge_middles
    assert len(m) == len(u)
    assert (u[0] + u[1]) / 2 in m


def test_sym_axis_list():
    u = [0.+5.j, 1.+1.j, 2.+5.j, 1.+7.j]
    q1, q2, q3 = 1, 2, 10
    line1 = SymAxis(Point(u[0]), Point(u[1]), q1, u)
    line2 = SymAxis(Point(u[0]), Point(u[2]), q2, u)
    line3 = SymAxis(Point(u[0]), Point(u[3]), q3, u)
    axis_list = SymAxisList([line1, line2, line3])
    print('axis_list:', axis_list)
    assert axis_list[1] is line2
    for q, n in [(q1-1, 1), (q1, 1), (q2, 2), (q3, 3), (q3+1, 3)]:
        selected = axis_list.select(q)
        assert len(selected) == n
    line1 = Axis(Point(u[0]), Point(u[2]))
    line2 = Axis(Point(u[1]), Point(u[2]))
    sym_axis_list = SymAxisList([line1, line2])
    print('sym_axis_list:', sym_axis_list)
    selected_sym = sym_axis_list.refinement(u, 2, 1.0).select(np.inf)
    assert len(selected_sym) == 1
    sym_axis = selected_sym[0]
    print('sym_axis:', sym_axis)
    assert sym_axis.q <= EPS


def test_sym_contour():
    u = [0.+5.j, 1.-1.j, 2.+5.j, 1.+6.j]
    sc = SymContour(u, n_max_pixels=1)
    print('sc:', sc)
    ch = sc.Hull_based
    assert len(ch) == len(u) * 2
    ap = sc.Approximate
    assert len(ap) > 0
    sa = sc.Axis_list
    assert len(sa) == 1
    a = sa[0]
    print('a:', a)
    assert close(a.Vec.angle, np.pi / 2, MIN_THETA_DIFF)
    p1, p2 = a.vertexes(u)
    assert close(p1.z, u[1]) and close(p2.z, u[3])


def test_contours_list():
    b = Binarizer()
    assert b.gauss == 0
    folder = 'test/files'
    pre = folder + '/preprocessed'
    images = ['flies.bmp', 'camel.gif', '50.mask.jpg']
    for img in images:
        cs = SymImage(folder, img, binarizer_object=b, tmp_folder=pre)
        print('cs:', cs)
        cs.create_txt()
        assert os.path.isdir(pre)
        assert os.path.isfile(pre + '/' + img[:-4] + '.txt')
        assert os.path.isfile(pre + '/' + img[:-4] + '.bmp')

        assert len(cs.Contours_list) == 1
        assert len(cs.Contours_list[0].origin) > 1
    img = images[0]
    cs2 = SymImage(folder, img, single=False, binarizer_object=b,
                   tmp_folder=pre)
    print('cs2:', cs2)
    assert len(cs2.Contours_list) > 10
    for u in cs2.Contours_list:
        assert u.area >= MIN_CONTOUR_AREA


def test_datafolder():
    folder = 'test/files'
    images = ['flies.bmp', 'camel.gif', '50.mask.jpg']
    df = DataFolder(folder)
    print('df:', df)
    keys = set(df.Contours_dict.keys())
    for image in images:
        assert image in keys


def test_fd():
    u = [5+0j, 7+3j, 5+5j, 0+0j, 5-5j, 7-3j]
    u = np.array(u, dtype=complex)
    fd = FourierDescriptor(u)
    print('fd:', fd)
    N = len(u)
    assert close_np(np.imag(fd.f_0)[1:], np.zeros(N-1))
    assert not close_np(np.imag(fd.f_s(1)), np.zeros(N-1))
    assert close_np(np.imag(fd.f_s(3)), np.zeros(N-1))
    for beta in np.arange(0.01, 1.01, 0.01):
        garms = min(round(max(2, N * beta)), N-1)
        assert len(fd.f_s(0, beta=beta, ret_zero=False)) == garms
        assert len(fd.f_s(0, beta=beta, ret_zero=True)) == garms + 1
