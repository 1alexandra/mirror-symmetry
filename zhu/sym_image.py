import os
import subprocess

from zhu import Binarizer
from zhu import SymContour
from zhu import DIMA_BORDERS_EXE
from zhu import MIN_CONTOUR_AREA
from zhu import MULT_INIT

from zhu.constants import Q_SIGNAL, Q_PIXELS


class SymImage:
    def __init__(
        self,
        data_folder,
        image_filename,
        mult_coef=MULT_INIT,
        single=True,
        min_area=MIN_CONTOUR_AREA,
        tmp_folder=None,
        binarizer_object=None,
        q_max_signal=Q_SIGNAL,
        q_max_pixels=Q_PIXELS,
        gauss=0
    ):
        self.img_path = data_folder + '/' + image_filename
        self.tmp_folder = tmp_folder or data_folder + '/preprocessed'
        if not os.path.isdir(self.tmp_folder):
            os.mkdir(self.tmp_folder)
        self.name, _ = os.path.splitext(image_filename)
        self.txt_path = self.tmp_folder + '/' + self.name + '.txt'
        self.mult_coef = mult_coef
        self.single = single
        self.min_area = min_area
        self.q_max_signal = q_max_signal
        self.q_max_pixels = q_max_pixels
        self.binarizer = binarizer_object or Binarizer(gauss)
        self.u_list = None

    Contours_list = property()
    Sym_measure = property()

    @Contours_list.getter
    def Contours_list(self):
        if self.u_list is None:
            self.u_list = self._get_contours()
        return self.u_list

    @Sym_measure.getter
    def Sym_measure(self):
        cl = self.Contours_list
        if len(cl):
            min_cnt = min(cl, key=lambda x: x.Sym_measure)
            return min_cnt.Sym_measure
        return None

    def __getitem__(self, i):
        cl = self.Contours_list
        return cl[i]

    def __len__(self):
        cl = self.Contours_list
        return len(cl)

    def create_txt(self, rm=True):
        tmp_name = self.tmp_folder + '/' + self.name
        bmp_path = tmp_name + '.bmp'
        self.binarizer.create_mono_bmp(self.img_path, bmp_path)
        subprocess.run([DIMA_BORDERS_EXE, tmp_name])
        if rm:
            try:
                os.remove(bmp_path)
            except FileNotFoundError:
                pass

    def _get_contours(self):
        if not os.path.isfile(self.txt_path):
            if not os.path.exists(self.img_path) or \
                    not os.path.isfile(self.img_path):
                return []
            self.create_txt()
        with open(self.txt_path, 'r') as f:
            text = f.read()
        ps = text.split('Polygon')[1:]

        ps = [u.split('\n')[1:] for u in ps]
        ps = [u for u in ps if len(u) >= 3]

        def to_complex(v):
            x, y = v
            return float(x) + 1j * float(y)

        ps = [[v.split() for v in u] for u in ps]
        ps = [[v for v in u if len(v) == 2] for u in ps]
        ps = [[to_complex(v) for v in u] for u in ps]
        ps = [
            SymContour(
                u,
                mult_coef=self.mult_coef,
                q_max_signal=self.q_max_signal,
                q_max_pixels=self.q_max_pixels
            ) for u in ps]

        if not len(ps):
            return []
        ans = [c for c in ps if c.Area >= self.min_area]
        if not self.single:
            ans.sort(key=lambda x: x.Sym_measure)
            return ans
        return [max(ans, key=len)]

    def __str__(self):
        return '\n'.join([
            '========',
            f'SymImage from {self.img_path}',
            *[str(c) for c in self.Contours_list],
            '========'
        ])
