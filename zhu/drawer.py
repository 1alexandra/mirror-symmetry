import numpy as np
from matplotlib import pyplot as plt
import cv2

from zhu import Point, Axis
from zhu import Contour

from zhu.draw_tools import get_box
from other.rgb import MplColorHelper


class ContourDrawer:
    def __init__(self, cnt):
        self.cnt = cnt

    def plot_setup(self):
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.axis('equal')
        plt.grid()

    def plot(
        self,
        axis_list=[],
        label='U',
        point_marker='go',
        edge_color='gray',
        axis_marker='-',
        ax=plt
    ):
        u = self.cnt.origin
        if edge_color:
            x = np.concatenate((np.real(u), [np.real(u[0])]))
            y = np.concatenate((np.imag(u), [np.imag(u[0])]))
            ax.plot(x, y, color=edge_color)
        if point_marker:
            ax.plot(np.real(u), np.imag(u), point_marker, label=label)
        if len(axis_list):
            color_helper = MplColorHelper('hsv', 0, len(axis_list))
            colors = color_helper.get_rgb_index()
            small, big = get_box(u)
            for i, line in enumerate(axis_list):
                s1, s2 = line.limit(small, big)
                c = colors[i]
                ax.plot([s1.x, s2.x], [s1.y, s2.y],
                        axis_marker, c=c, label=f'Symmetry axis {i+1}')
        if label:
            ax.legend()

    def draw(self, axis_list=[]):
        u = self.cnt.origin
        left, right = np.min(np.real(u)), np.max(np.real(u))
        down, up = np.min(np.imag(u)), np.max(np.imag(u))
        margin = max(right - left, up - down) * 0.1
        w, h = int(right - left + 2 * margin), int(up - down + 2 * margin)
        line_w = max(1, min(w, h) // 50)
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

    def __str__(self):
        return f'Drawer for {self.cnt}'
