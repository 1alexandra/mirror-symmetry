import os
import numpy as np

from zhu import SymImage


class DataFolder:
    def __init__(
        self,
        folder,
        sym_image_kwargs={},
        number=None,
    ):
        self.folder = folder
        path = './' + folder if not folder.startswith('.') else folder
        self.filenames = os.listdir(path=path)
        if number is not None and number < len(self.filenames):
            index = np.random.permutation(len(self.filenames))[:number]
            self.filenames = [self.filenames[i] for i in index]
        self.si_kwargs = sym_image_kwargs
        self.cnt_dict = None
        self.trained_neibs_hull = None
        self.trained_neibs_appr = None

    Contours_dict = property()
    Trained_neibs_hull = property()
    Trained_neibs_appr = property()

    @Contours_dict.getter
    def Contours_dict(self):
        if self.cnt_dict is None:
            ans = {}
            for filename in self.filenames:
                ans[filename] = SymImage(self.folder, filename,
                                         **self.si_kwargs)
            self.cnt_dict = ans
        return self.cnt_dict

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
