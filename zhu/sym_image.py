import os
import subprocess

from zhu import Binarizer
from zhu import SymContour
from zhu import DIMA_BORDERS_EXE
from zhu import MIN_CONTOUR_AREA


class SymImage:
    def __init__(
        self,
        data_folder,
        image_filename,
        sym_contour_kwargs={},
        single=True,
        min_area=MIN_CONTOUR_AREA,
        tmp_folder=None,
        binarizer_object=None,
        gauss=0
    ):
        self.img_path = os.path.join(data_folder, image_filename)
        self.tmp_folder = tmp_folder or os.path.join(
            data_folder, 'preprocessed')
        if not os.path.isdir(self.tmp_folder):
            os.mkdir(self.tmp_folder)
        self.name = os.path.basename(image_filename)
        self.txt_path = os.path.join(self.tmp_folder, self.name + '.txt')
        self.sc_kwargs = sym_contour_kwargs
        self.single = single
        self.min_area = min_area
        self.binarizer = binarizer_object or Binarizer(gauss)
        self.u_list = None
        self.trained_neibs_hull = None
        self.trained_neibs_appr = None
        self.rm_binary = False

    Contours_list = property()
    Sym_measure = property()
    Trained_neibs_hull = property()
    Trained_neibs_appr = property()

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

    @Trained_neibs_hull.getter
    def Trained_neibs_hull(self):
        if self.trained_neibs_hull is None:
            if not len(self):
                return None
            neibs = [sc.Trained_neibs_hull for sc in self]
            neibs = [n for n in neibs if n is not None]
            if not len(neibs):
                return None
            self.trained_neibs_hull = max(neibs)
        return self.trained_neibs_hull

    @Trained_neibs_appr.getter
    def Trained_neibs_appr(self):
        if self.trained_neibs_appr is None:
            if not len(self):
                return None
            neibs = [sc.Trained_neibs_appr for sc in self]
            neibs = [n for n in neibs if n is not None]
            if not len(neibs):
                return None
            self.trained_neibs_appr = max(neibs)
        return self.trained_neibs_appr

    def __getitem__(self, i):
        cl = self.Contours_list
        return cl[i]

    def __len__(self):
        cl = self.Contours_list
        return len(cl)

    def create_txt(self):
        tmp_name = self.tmp_folder + '/' + self.name
        bmp_path = tmp_name + '.bmp'
        self.binarizer.create_mono_bmp(self.img_path, bmp_path)
        subprocess.run([DIMA_BORDERS_EXE, tmp_name])
        if self.rm_binary:
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
        ps = [SymContour(u, **self.sc_kwargs) for u in ps]

        if not len(ps):
            return []
        ans = [c for c in ps if c.Area >= self.min_area]
        ans.sort(key=lambda x: x.Sym_measure)
        if not self.single:
            return ans
        return [max(ans, key=len)]

    def __str__(self):
        return '\n'.join([
            '========',
            f'SymImage from {self.img_path}',
            *[str(c) for c in self.Contours_list],
            '========'
        ])
