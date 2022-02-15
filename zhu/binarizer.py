import os

import numpy as np
import cv2
from PIL import Image

from zhu.draw_tools import imread_bw


class Binarizer:
    def __init__(self, gauss=0, use_otsu=True):
        self.gauss = gauss
        self.otsu_dilate_kernel_size = 3
        self.otsu_erose_kernel_size = 5
        self.binarize = self.binarize_otsu if use_otsu \
            else self.binarize_mono

    def binarize_mono(self, image_path):
        img = imread_bw(image_path)
        if img is None:
            return None
        if self.gauss:
            size = (self.gauss, self.gauss)
            img = cv2.GaussianBlur(img, size, 0)
        img = np.array(img, dtype=np.uint8)[::-1]
        img = img * (255 // np.max(img))
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return img

    def binarize_otsu(self, image_path):
        # Leaves binarization by Kirill Berber
        im = Image.open(image_path)
        im = np.array(im)
        if len(im.shape) == 2:
            ret, im_FIN = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
        else:
            red = im[:, :, 0]
            green = im[:, :, 1]
            blue = im[:, :, 2]

            ret_r, th_r = cv2.threshold(red, 0, 255, cv2.THRESH_OTSU)
            ret_g, th_g = cv2.threshold(green, 0, 255, cv2.THRESH_OTSU)
            ret_b, th_b = cv2.threshold(blue, 0, 255, cv2.THRESH_OTSU)

            im_FIN = th_r * th_g * th_b

        kernel = np.ones((self.otsu_dilate_kernel_size,
                          self.otsu_dilate_kernel_size), np.uint8)
        kernel_2 = np.ones((self.otsu_erose_kernel_size,
                            self.otsu_erose_kernel_size), np.uint8)

        dilate_img = cv2.dilate(im_FIN, kernel, iterations=1)
        erosion = cv2.erode(dilate_img, kernel_2, iterations=1)

        return erosion[::-1]

    def create_mono_bmp(self, image_path, res_path=None):
        if res_path is None:
            path, ext = os.path.splitext(image_path)
            res_path = path + '.bmp'
        data = self.binarize(image_path)
        if data is None:
            return
        if data[0, 0] == 255:
            data = 255 - data
        data = (255 - data) // 255
        img = Image.new('1', data.shape)
        pixels = img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                pixels[i, j] = int(data[i][j])
        img = img.transpose(Image.ROTATE_90)
        img.save(res_path)

    def __str__(self):
        return 'Binarizer'
