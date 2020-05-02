import os

import numpy as np
import cv2
from PIL import Image


class Binarizer:
    def __init__(self, gauss=0):
        self.gauss = 0

    def imread_bw(self, image_path):
        if not image_path.endswith('.gif'):
            return cv2.imread(image_path, 0)
        img = Image.open(image_path)
        tmp_path = os.path.splitext(image_path)[0] + '.bmp'
        img.save(tmp_path, 'BMP')
        img = cv2.imread(tmp_path, 0)
        os.remove(tmp_path)
        return img

    def binarize(self, image_path):
        img = self.imread_bw(image_path)
        if img is None:
            return None
        if self.gauss:
            size = (self.gauss, self.gauss)
            img = cv2.GaussianBlur(img, size, 0)
        img = np.array(img, dtype=np.uint8)[::-1]
        img = img * (255 // np.max(img))
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        if img[0, 0] == 255:
            img = 255 - img
        return img

    def create_mono_bmp(self, image_path, res_path):
        data = self.binarize(image_path)
        if data is None:
            return
        data = (255 - data) // 255
        img = Image.new('1', data.shape)
        pixels = img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                pixels[i, j] = (data[i][j],)
        img = img.transpose(Image.ROTATE_90)
        img.save(res_path)

    def __str__(self):
        return 'Binarizer'
