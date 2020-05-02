import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab

from zhu import Point

from zhu import CMAP_DEFAULT, CMAP_ALTERNATIVE, CMAP_OTHER


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


def save_plot(name, format_='png', res_folder='results'):
    if not os.path.isdir(res_folder):
        os.mkdir(res_folder)
    path = res_folder + '/' + name + '.' + format_
    plt.savefig(path, format=format_, bbox_inches='tight')


def imshow_bw(img, q=None, cmap=CMAP_DEFAULT, ax=None):
    title = f'Q = {round(q, 3)}' if q is not None else ''
    if ax is None:
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(255 - img, cmap=cmap)
    else:
        ax.set_title(title, fontsize=30)
        ax.imshow(255 - img, cmap=cmap)


def choose_cmap(is_sym, last_cmap=None):
    if not is_sym:
        return CMAP_OTHER
    if last_cmap is not CMAP_DEFAULT:
        return CMAP_DEFAULT
    return CMAP_ALTERNATIVE


def get_box(u):
    left, right = np.min(np.real(u)), np.max(np.real(u))
    down, up = np.min(np.imag(u)), np.max(np.imag(u))
    margin = max(right - left, up - down) * 0.1
    small = Point(left - margin, down - margin)
    big = Point(right + margin, up + margin)
    return small, big
