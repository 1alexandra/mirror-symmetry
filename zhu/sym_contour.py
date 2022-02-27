import numpy as np
import cv2

from zhu import Point, Axis
from zhu import Contour
from zhu import Scaler
from zhu import SymAxisList

from zhu import BETA_SIG, BETA_PIX, MULT_INIT, NEIB_HULL, NEIB_APPR
from zhu import Q_SIGNAL, Q_PIXELS
from zhu import N_MAX_SIGNAL, N_MAX_PIXELS
from zhu import USE_HULL


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
        q_max_pixels=Q_PIXELS,
        n_max_signal=N_MAX_SIGNAL,
        n_max_pixels=N_MAX_PIXELS,
        use_hull=USE_HULL,
        train_neibs=False,
    ):
        super().__init__(u, mult_coef)

        self.neibs_hull = neibs_hull
        self.neibs_appr = neibs_approximate
        self.beta_sig = beta_signal
        self.beta_pix = beta_pixels
        self.q_max_sig = q_max_signal
        self.q_max_pix = q_max_pixels
        self.n_max_sig = n_max_signal
        self.n_max_pix = n_max_pixels
        self.use_hull = use_hull
        self.train_neibs = train_neibs

        self.scaler = Scaler(self.Pixels)
        self.hull_based = None
        self.approximate = None
        self.axis_list = None
        self.sym_measure = None
        self.trained_neibs_hull = None
        self.trained_neibs_appr = None

    Hull_based = property()
    Approximate = property()
    Axis_list = property()
    Sym_measure = property()
    Trained_neibs_hull = property()
    Trained_neibs_appr = property()

    @Hull_based.getter
    def Hull_based(self):
        if self.hull_based is None:
            if self.neibs_hull is None or not self.use_hull:
                return SymAxisList(None, self.scaler)
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
                self.beta_sig,
                train=self.train_neibs,
            ).select(self.q_max_sig, self.n_max_sig)
        return self.approximate

    @Axis_list.getter
    def Axis_list(self):
        if self.axis_list is None:
            lines = self.Approximate
            self.axis_list = lines.refinement(
                self.Pixels,
                self.neibs_appr,
                self.beta_pix,
                train=self.train_neibs,
            ).select(self.q_max_pix, self.n_max_pix)
        return self.axis_list

    @Sym_measure.getter
    def Sym_measure(self):
        if self.sym_measure is None:
            if len(self.Axis_list):
                self.sym_measure = self.Axis_list[0].q
        return self.sym_measure

    @Trained_neibs_hull.getter
    def Trained_neibs_hull(self):
        if self.trained_neibs_hull is None:
            if not len(self):
                return None
            self.trained_neibs_hull = self.Hull_based.neibs_trained
        return self.trained_neibs_hull

    @Trained_neibs_appr.getter
    def Trained_neibs_appr(self):
        if self.trained_neibs_appr is None:
            if not len(self):
                return None
            self.trained_neibs_appr = self.Approximate.neibs_trained
        return self.trained_neibs_appr

    def symmetrical(self):
        if self.Sym_measure is None:
            return False
        return self.Sym_measure < self.q_max_pix

    def draw(self, board):
        board = super().draw(board)
        if len(self.Axis_list) > 0:
            axis = self.Axis_list[0]
            z1, z2 = axis.vertexes(self.Pixels)
            p1 = (int(z1.z.real), int(z1.z.imag))
            p2 = (int(z2.z.real), int(z2.z.imag))

            board = cv2.line(
                cv2.UMat(board), p1, p2,
                color=(255, 255, 255),
                thickness=self.draw_kwargs['thickness']*2)

            board = cv2.line(
                cv2.UMat(board), p1, p2, **self.draw_kwargs)
            if type(board) is cv2.UMat:
                board = board.get()
        return board

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
