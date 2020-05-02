import numpy as np
import cv2

from zhu.constants import MIN_CONTOUR_AREA
from zhu.binarizer import Binarizer


def cnt_to_u(cnt):
    """Convert cv2 contour to complex numpy array."""
    cnt_ = cnt.reshape((-1, 2))
    return cnt_[:, 0] + 1j * cnt_[:, 1]


class ContourFinderFromImage:
    def __init__(
        self,
        data_folder,
        image_filename,
        single=True,
        min_area=MIN_CONTOUR_AREA,
        gauss=0
    ):
        self.path = data_folder + '/' + image_filename
        self.single = single
        self.min_area = min_area
        self.gauss = gauss
        self.img = cv2.imread(self.path)
        self.img_b = Binarizer().binarize(self.path)

    def get_contours(self):
        if self.img_b is None:
            return []
        w, h = self.img_b.shape
        margin = 1
        img = np.zeros((w + 2*margin, h + 2*margin), dtype=np.uint8)
        img[margin:-margin, margin:-margin] = self.img_b
        cs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        measure = np.array([cv2.contourArea(c) for c in cs])
        if self.single:
            index = [np.argmax(measure)]
        else:
            index = np.arange(len(measure))[measure >= self.min_area]
        return [cnt_to_u(cs[i]) - margin*(1+1j) for i in index]
