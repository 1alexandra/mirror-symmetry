import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image

from zhu import Point, Axis
from zhu import Contour
from zhu import FourierDescriptor
from zhu import Scaler
from zhu import SymContour

from zhu.tools import round_complex, join_neibs
from zhu.draw_tools import get_box
from zhu.draw_tools import save_plot
from zhu.draw_tools import prepare_scene
from zhu.draw_tools import imread_bw

from other.rgb import MplColorHelper


def plot_complex_contour(
    u,
    axis_list=[],
    point_marker=None,
    edge_color=None,
    fill_color=None,
    label='',
    ax=plt
):
    label = label if label == '' else label + ' '
    if edge_color or fill_color:
        x = np.concatenate((np.real(u), [np.real(u[0])]))
        y = np.concatenate((np.imag(u), [np.imag(u[0])]))
        if fill_color:
            ax.fill(x, y, facecolor=fill_color, edgecolor=edge_color,
                    label=label + 'silhouette')
        else:
            ax.plot(x, y, color=edge_color, label=label + 'contour')
    if point_marker:
        ax.plot(np.real(u), np.imag(u), point_marker, label=label + 'points')
    if len(axis_list):
        color_helper = MplColorHelper('hsv', 0, len(axis_list))
        colors = color_helper.get_rgb_index()
        small, big = get_box(u)
        for i, line in enumerate(axis_list):
            s1, s2 = line.limit(small, big)
            c = colors[i]
            axis_label = label + 'symmetry axis'
            if len(axis_list) > 1:
                axis_label += f' {i+1}'
            try:
                q = line.q
                theta = line.Vec.angle
                axis_label += f'\ntheta = {round(theta,2)}, Q = {round(q,3)}'
            except Exception:
                pass
            ax.plot([s1.x, s2.x], [s1.y, s2.y], linestyle='-', c=c,
                    label=axis_label)


def draw_complex_contour(u, axis_list=[]):
    left, right = np.min(np.real(u)), np.max(np.real(u))
    down, up = np.min(np.imag(u)), np.max(np.imag(u))
    margin = max(right - left, up - down) * 0.1
    w, h = int(right - left + 2 * margin), int(up - down + 2 * margin)
    line_w = max(1, min(w, h) // 100)
    vec = - left - 1j * down + margin * (1 + 1j)
    new_u = 1j * h + np.conjugate(u + vec)
    cnt = Contour(new_u).Contour_cv
    img = np.zeros((h, w))
    cv2.drawContours(img, [cnt], 0, 255, line_w)
    for line in axis_list:
        axis_point, axis_vec = line.z1, line.Vec.z
        new_v = np.conjugate(axis_vec)
        new_p = 1j * h + np.conjugate(axis_point + vec)
        new_line = Axis(Point(new_p), Point(new_p + new_v))
        s1, s2 = new_line.limit(Point(0 + 0j), Point(w + h * 1j))
        cv2.line(img, (int(s1.x), int(s1.y)), (int(s2.x), int(s2.y)),
                 255, line_w)
    return img


class SymContourDrawer:
    def __init__(
        self,
        cnt,
        save_folder='../results',
        save_format='eps',
        width=None,
        height=None,
        origin_point_marker='r.',
        origin_edge_color='darkgray',
        origin_fill_color='lightgray',
        signal_point_marker='bo',
        signal_edge_color=None,
        signal_fill_color=None,
        hull_point_marker='c.',
        hull_edge_color='c',
        hull_fill_color=None,
        hull_based_marker='ro',
        hull_neibs_marker='yo',
        hull_centroid_marker='co',
        hull_middles_marker='c.',
        hull_lines_marker='c--',
        fd_color='g',
        q_color='g',
        q_sym_marker='r.'
    ):
        self.cnt = cnt
        self.save_folder = save_folder
        self.save_format = save_format
        self.width = width
        self.height = height
        self.origin_point_marker = origin_point_marker
        self.origin_edge_color = origin_edge_color
        self.origin_fill_color = origin_fill_color
        self.signal_point_marker = signal_point_marker
        self.signal_edge_color = signal_edge_color
        self.signal_fill_color = signal_fill_color
        self.hull_point_marker = hull_point_marker
        self.hull_edge_color = hull_edge_color
        self.hull_fill_color = hull_fill_color
        self.hull_based_marker = hull_based_marker
        self.hull_neibs_marker = hull_neibs_marker
        self.hull_centroid_marker = hull_centroid_marker
        self.hull_middles_marker = hull_middles_marker
        self.hull_lines_marker = hull_lines_marker
        self.fd_color = fd_color
        self.q_color = q_color
        self.q_sym_marker = q_sym_marker
        self.xlim = None
        self.ylim = None

    def plot_setup(self):
        prepare_scene(self.width, self.height)
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.axis('equal')
        if self.xlim:
            plt.xlim(self.xlim)
            print('set')
        if self.ylim:
            plt.ylim(self.ylim)
        plt.grid()

    def draw(self, axis_list=[]):
        u = self.cnt.origin
        return draw_complex_contour(u, axis_list)

    def plot(self, axis_list=[], label='polygon', title='Polygon'):
        self.plot_setup()
        u = self.cnt.origin
        plot_complex_contour(
            u=u,
            axis_list=axis_list,
            point_marker=self.origin_point_marker,
            edge_color=self.origin_edge_color,
            fill_color=self.origin_fill_color,
            label=label
        )
        plt.title(title + f', N={len(u)}')

    def plot_signal(
        self,
        axis_list=[],
        label='signal',
        title='Signal',
        plot_origin=True
    ):
        if plot_origin:
            origin_point_marker = self.origin_point_marker
            self.origin_point_marker = None
            self.plot()
            self.origin_point_marker = origin_point_marker
        u = self.cnt.Signal
        plot_complex_contour(
            u=u,
            axis_list=axis_list,
            point_marker=self.signal_point_marker,
            edge_color=self.signal_edge_color,
            fill_color=self.signal_fill_color,
            label=label
        )
        plt.title(title + f', N={len(u)}')

    def plot_hull(self, centroid=True, middles=True, neibs=None,
                  label='', title='Points to find symmetry'):
        label = '' if label == '' else label + ' '
        if neibs is None:
            neibs = self.cnt.neibs_hull
        c = self.cnt.Centroid
        ch = self.cnt.Convex_hull
        d = SymContourDrawer(
            ch,
            width=self.width,
            height=self.height,
            origin_point_marker=self.hull_point_marker,
            origin_edge_color=self.hull_edge_color,
            origin_fill_color=self.hull_fill_color
        )
        d.plot(label='convex hull')
        if middles:
            m = ch.Edge_middles
            plt.plot(np.real(m), np.imag(m), self.hull_middles_marker)
        if centroid:
            plt.plot([np.real(c)], [np.imag(c)], self.hull_centroid_marker,
                     label=label + 'centroid')
        u = self.cnt.Signal
        ls = self.cnt.Hull_based
        inds = np.ravel([[line.vertexes_ind(u)] for line in ls])
        v = u[join_neibs(u, inds, 0)]
        plt.plot(np.real(v), np.imag(v), self.hull_based_marker,
                 label=label + 'hull based axis points')
        v_neibs = u[join_neibs(u, inds, neibs)]
        v_neibs = [p for p in v_neibs if p not in v]
        plt.plot(np.real(v_neibs), np.imag(v_neibs), self.hull_neibs_marker,
                 label=label + 'neibs of axis points')
        n1 = len(u)
        n2 = len(v) + len(v_neibs)
        plt.grid()
        plt.title(title + f', N(signal) = {n1}, N(axis points) = {n2}')

    def plot_filter(self, beta=1.0, plot_origin=True, single_axis=True):
        if plot_origin:
            origin_point_marker = self.origin_point_marker
            self.origin_point_marker = None
            self.plot()
            self.origin_point_marker = origin_point_marker
        fd = FourierDescriptor(self.cnt.Signal)
        u = fd.smoothed(beta)
        cnt = SymContour(u, mult_coef=1)
        d = SymContourDrawer(
            cnt,
            width=self.width,
            height=self.height,
        )
        lines = cnt.Axis_list.lines
        if single_axis:
            lines = [lines[0]]
        garms = np.sum(fd.criteria(beta, True))
        title = f'Filtered Signal, G = {garms}'
        d.plot_signal(lines, title=title, plot_origin=False)

    def plot_f(
        self,
        s, beta=1.0, ret_zero=False,
        scaling=False,
        set_limits=False
    ):
        self.plot_setup()
        scaler = None
        if not scaling:
            scaler = Scaler()
            scaler.scale = 1
            scaler.vec = 0 + 0j
        fd = FourierDescriptor(self.cnt.Signal, scaler=scaler)
        f = fd.f_s(s, beta, ret_zero)
        start = round_complex(fd.signal[s])
        plt.title(f'Fourier Descriptor, N = {len(f)}')
        size = np.max(np.abs(f)) // 250
        label = 'Fourier coefficients' + '\n' + f'start = {start}'
        plt.scatter(np.real(f), np.imag(f), color=self.fd_color, s=size,
                    label=label)
        if set_limits:
            self.xlim = plt.xlim()
            self.ylim = plt.ylim()
        else:
            self.xlim = None
            self.ylim = None
        return start

    def plot_q(
        self,
        beta=1.0,
        label='Q',
        title='Asymmetry measure',
        sym_ind=[]
    ):
        plt.xlabel('p, point number')
        plt.ylabel('Q, asymmetry measure')
        plt.grid()
        fd = FourierDescriptor(self.cnt.Signal)
        x = np.arange(len(fd))
        y = np.array([fd.symmetry_measure(s=s, beta=beta) for s in x])
        l1 = label + '(p)'
        plt.plot(x, y, color=self.q_color, label=l1)
        if len(sym_ind):
            l2 = 'min' + l1
            if len(sym_ind) > 1:
                l2 = 'local ' + l2
            plt.plot(x[sym_ind], y[sym_ind], self.q_sym_marker, label=l2)
        plt.ylim(bottom=0.0)
        plt.title(title + f', N = {len(x)}')

    def show(self, legend=True, ax=plt):
        if legend:
            ax.legend()
        ax.show()

    def save(self, name, legend=True, ax=plt, show=True):
        if legend:
            ax.legend()
        save_plot(name, format_=self.save_format, res_folder=self.save_folder)
        if show:
            self.show(legend=False, ax=ax)

    def __str__(self):
        return f'Drawer for {self.cnt}'


def draw_silhouettes(sym_image, save_path=None):
    origin_bw = imread_bw(sym_image.img_path)
    board = np.zeros(origin_bw.shape)
    for cnt in sym_image:
        cnt_cv = cnt.Contour_cv
        board = cv2.fillPoly(board, pts=[cnt_cv], color=(255, 255, 255))
    if save_path is not None:
        im = Image.fromarray(board[::-1].astype(np.uint8))
        im.save(save_path)
    return board
