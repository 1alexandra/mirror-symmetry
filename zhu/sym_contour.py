import numpy as np

from zhu import Point, Axis
from zhu import Contour
from zhu import Scaler
from zhu import SymAxisList

from zhu import BETA_SIG, BETA_PIX, MULT_INIT, NEIB_HULL, NEIB_APPR
from zhu import Q_SIGNAL, Q_PIXELS


class SymContour(Contour):
    def __init__(
        self,
        u,
        mult_coef=MULT_INIT,
        neibs_hull=NEIB_HULL,
        neibs_approximate=NEIB_APPR,
        beta_signal=BETA_SIG,
        beta_pixels=BETA_PIX,
        q_max_signal=Q_SIGNAL,
        q_max_pixels=Q_PIXELS
    ):
        super().__init__(u, mult_coef)
        self.neibs_hull = neibs_hull
        self.neibs_appr = neibs_approximate
        self.beta_sig = beta_signal
        self.beta_pix = beta_pixels
        self.q_max_sig = q_max_signal
        self.q_max_pix = q_max_pixels
        self.scaler = Scaler(self.Pixels)
        self.hull_based = None
        self.approximate = None
        self.axis_list = None
        self.sym_measure = None

    Hull_based = property()
    Approximate = property()
    Axis_list = property()
    Sym_measure = property()

    @Hull_based.getter
    def Hull_based(self):
        if self.hull_based is None:
            if self.neibs_hull is None:
                return np.arange(len(self.signal))
            ch = self.Convex_hull
            vs = np.vstack((ch.origin, ch.Edge_middles)).T.ravel()
            c = Point(self.Centroid)
            lines = [Axis(c, Point(v)) for v in vs]
            self.hull_based = SymAxisList(lines, self.scaler)
        return self.hull_based

    @Approximate.getter
    def Approximate(self):
        if self.approximate is None:
            lines = self.Hull_based
            self.approximate = lines.refinement(
                self.Signal,
                self.neibs_hull,
                self.beta_sig
            )
        return self.approximate

    @Axis_list.getter
    def Axis_list(self):
        if self.axis_list is None:
            lines = self.Approximate
            self.axis_list = lines.refinement(
                self.Pixels,
                self.neibs_appr,
                self.beta_pix
            ).select(self.q_max_pix)
        return self.axis_list

    @Sym_measure.getter
    def Sym_measure(self):
        if self.sym_measure is None:
            if len(self.Axis_list):
                self.sym_measure = self.Axis_list[0].q
        return self.sym_measure

    def symmetrical(self):
        if self.Sym_measure is None:
            return False
        return self.Sym_measure < self.q_max_pix

    def __str__(self):
        return '\n'.join([
            '===========',
            'SymContour',
            super().__str__(),
            f'Hull_based = {self.Hull_based}',
            f'Approximate = {self.Approximate}',
            f'Axis_list = {self.Axis_list}',
            f'Sym_measure = {self.Sym_measure}',
            f'Is symmetrical: {self.symmetrical()}',
            '============'
        ])
