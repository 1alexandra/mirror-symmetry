import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm


class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.start_val = start_val
        self.stop_val = stop_val
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

    def get_rgb_index(self):
        return [self.get_rgb(i) for i in range(self.start_val, self.stop_val)]


def bw_to_rgb(img):
    w, h = img.shape
    rgb = np.zeros((w, h, 3), dtype=int)
    for i in range(3):
        rgb[:, :, i] = img.copy()
    return rgb
