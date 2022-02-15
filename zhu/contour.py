import numpy as np
import cv2
import scipy.spatial

from zhu import MULT_INIT, MIN_CONTOUR_N
from zhu.tools import round_complex


class Contour:
    def __init__(
        self,
        u,
        mult_coef=MULT_INIT,
        n_min=MIN_CONTOUR_N,
        n_max=None
    ):
        self.origin = np.array(u, dtype=complex)
        self.mult_coef = mult_coef
        self.n_min = n_min
        self.n_max = n_max
        self.cnt = None
        self.area = None
        self.perimeter = None
        self.step = None
        self.signal = None
        self.pixels = None
        self.hull = None
        self.middles = None
        self.centroid = None

    Contour_cv = property()
    Area = property()
    Perimeter = property()
    Step = property()
    Signal = property()
    Pixels = property()
    Convex_hull = property()
    Edge_middles = property()
    Centroid = property()

    @Contour_cv.getter
    def Contour_cv(self):
        if self.cnt is None:
            u = round_complex(self.origin)
            u = np.array([[np.real(u), np.imag(u)]]).T.reshape((-1, 1, 2))
            self.cnt = u.astype(np.int32)
        return self.cnt

    @Area.getter
    def Area(self):
        if self.area is None:
            self.area = cv2.contourArea(self.Contour_cv)
        return self.area

    @Perimeter.getter
    def Perimeter(self):
        if self.perimeter is None:
            shifted = np.roll(self.origin, 1)
            self.perimeter = np.sum(np.abs(self.origin - shifted))
        return self.perimeter

    @Signal.getter
    def Signal(self):
        if self.signal is None:
            self.signal = self.discretization(self.Step)
        return self.signal

    @Pixels.getter
    def Pixels(self):
        if self.pixels is None:
            self.pixels = self.discretization(self.get_pixel_step())
        return self.pixels

    @Convex_hull.getter
    def Convex_hull(self):
        if self.hull is None:
            u = self.origin
            data = np.array([np.real(u), np.imag(u)]).T
            index = scipy.spatial.ConvexHull(data).vertices
            self.hull = Contour(u[index], self.mult_coef)
        return self.hull

    @Edge_middles.getter
    def Edge_middles(self):
        if self.middles is None:
            u = self.origin
            ms = [(u[i] + u[(i + 1) % len(u)]) / 2 for i in range(len(u))]
            self.middles = np.array(ms, dtype=complex)
        return self.middles

    @Centroid.getter
    def Centroid(self):
        if self.centroid is None:
            self.centroid = np.mean(self.Pixels)
        return self.centroid

    @Step.getter
    def Step(self):
        if self.step is None:
            n = len(self.origin)
            n_max = self.n_max or int(self.Perimeter + 1)
            n_new = min(max(int(round(n * self.mult_coef)), self.n_min), n_max)
            self.step = self.Perimeter / n_new
        return self.step

    def get_pixel_step(self):
        return self.Step / int(round(self.Step))

    def discretization(self, step):
        """Walk the contour in max(len(u)*mult, 3) constant steps."""
        u = self.origin
        seg_ind, seg_start, cur_step = 0, u[0], step
        w = []
        for i in range(int(round(self.Perimeter / step))):
            w.append(seg_start)
            while True:
                seg_end = u[(seg_ind + 1) % len(u)]
                seg_vec = seg_end - seg_start
                seg_len = abs(seg_vec)
                if seg_len < cur_step:
                    seg_ind += 1
                    seg_start = seg_end
                    cur_step -= seg_len
                else:
                    seg_start += seg_vec / seg_len * cur_step
                    cur_step = step
                    break
        return np.array(w)

    def __len__(self):
        return len(self.origin)

    def __getitem__(self, i):
        return self.origin[i]

    def __str__(self):
        return '\n'.join([
            '=========',
            'Contour:',
            f'origin = {self.origin}',
            f'mult = {self.mult_coef}',
            f'Area = {self.Area}',
            f'Perimeter = {self.Perimeter}',
            f'Signal: {self.Signal}',
            f'Pixels: {self.Pixels}',
            f'Hull: {self.Convex_hull.origin}',
            f'Middles: {self.Edge_middles}',
            f'Centroid: {self.Centroid}',
            '=========='
        ])
