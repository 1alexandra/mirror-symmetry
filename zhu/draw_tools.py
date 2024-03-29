import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
import cv2
from PIL import Image

from zhu import Point

from zhu import CMAP_DEFAULT, CMAP_ALTERNATIVE, CMAP_OTHER


def imread_bw(image_path):
    if not image_path.endswith('.gif'):
        return cv2.imread(image_path, 0)
    img = Image.open(image_path)
    tmp_path = os.path.splitext(image_path)[0] + '.bmp'
    img.save(tmp_path, 'BMP')
    img = cv2.imread(tmp_path, 0)
    os.remove(tmp_path)
    return img


def prepare_scene(width=9, height=6):
    """Prettify diagrams for science reports."""
    params = {
        'legend.fontsize': 'x-large',
        'figure.figsize': (width, height),
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large'
    }
    pylab.rcParams.update(params)


def save_plot(name, format_='png', res_folder='../plot_results'):
    if not os.path.isdir(res_folder):
        os.mkdir(res_folder)
    path = res_folder + '/' + name + '.' + format_
    plt.savefig(path, format=format_, bbox_inches='tight')


def imshow_bw(img, q=None, cmap=CMAP_DEFAULT, ax=None, inverse=True):
    if inverse:
        img = 255 - img
    title = f'Q = {round(q, 3)}' if q is not None else ''
    if ax is None:
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=cmap)
    else:
        ax.set_title(title, fontsize=30)
        ax.imshow(img, cmap=cmap)


def choose_cmap(is_sym, last_cmap=None, change=False):
    if not is_sym:
        return CMAP_OTHER
    if not change:
        if last_cmap is not None:
            return last_cmap
        return CMAP_DEFAULT
    if last_cmap is not CMAP_DEFAULT:
        return CMAP_DEFAULT
    return CMAP_ALTERNATIVE


def get_box(u):
    left, right = np.min(np.real(u)), np.max(np.real(u))
    down, up = np.min(np.imag(u)), np.max(np.imag(u))
    margin = max(right - left, up - down) * 0.1
    small = Point(left - margin + 1j * (down - margin))
    big = Point(right + margin + 1j * (up + margin))
    return small, big
