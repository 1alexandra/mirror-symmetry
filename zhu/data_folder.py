import os

from zhu import SymImage
from zhu import MIN_CONTOUR_AREA, MULT_INIT


class DataFolder:
    def __init__(
        self,
        folder,
        mult_coef=MULT_INIT,
        single=True,
        min_area=MIN_CONTOUR_AREA
    ):
        self.folder = folder
        self.filenames = os.listdir(path='./' + folder)
        self.mult_coef = mult_coef
        self.single = single
        self.min_area = min_area
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
                    self.mult_coef,
                    self.single,
                    self.min_area
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
            'DataFolder from {self.folder}',
            *[str(cl) for cl in self.Contours_dict],
            '========'
        ])
