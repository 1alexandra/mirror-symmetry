import os
import numpy as np

from zhu import SymImage
from zhu import MIN_CONTOUR_AREA, MULT_INIT
from zhu import Q_SIGNAL, Q_PIXELS


class DataFolder:
    def __init__(
        self,
        folder,
        mult_coef=MULT_INIT,
        single=True,
        min_area=MIN_CONTOUR_AREA,
        q_max_signal=Q_SIGNAL,
        q_max_pixels=Q_PIXELS,
        number=None,
    ):
        self.folder = folder
        self.filenames = os.listdir(path='./' + folder)
        if number is not None and number < len(self.filenames):
            index = np.random.permutation(len(self.filenames))[:number]
            self.filenames = [self.filenames[i] for i in index]
        self.mult_coef = mult_coef
        self.single = single
        self.min_area = min_area
        self.q_max_signal = q_max_signal
        self.q_max_pixels = q_max_pixels
        self.cnt_dict = None

    Contours_dict = property()

    @Contours_dict.getter
    def Contours_dict(self):
        if self.cnt_dict is None:
            ans = {}
            for filename in self.filenames:
                ans[filename] = SymImage(
                    self.folder,
                    filename,
                    mult_coef=self.mult_coef,
                    single=self.single,
                    min_area=self.min_area,
                    q_max_signal=self.q_max_signal,
                    q_max_pixels=self.q_max_pixels,
                )
            self.cnt_dict = ans
        return self.cnt_dict

    def __len__(self):
        cd = self.Contours_dict
        return len(cd)

    def __getitem__(self, i):
        cd = self.Contours_dict
        return cd[self.filenames[i]]

    def __str__(self):
        return '\n'.join([
            '========',
            f'DataFolder from {self.folder}',
            *[str(cl) for cl in self.Contours_dict],
            '========'
        ])
