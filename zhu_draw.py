import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab

from zhu_contour import u_to_cnt

def prepare_scene(w=20, h=5):
    """
    input:
    w, h -- plt figsize
    output:
    None.
    Prettify diagrams for science reports.
    """
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (w, h),
              'axes.labelsize': 'x-large',
              'axes.titlesize':'x-large',
              'xtick.labelsize':'x-large',
              'ytick.labelsize':'x-large'}
    pylab.rcParams.update(params)

def contour(u, 
            method = 'plt',
            margin = 0.1,
            show = True,
            label = 'U',
            dot_style = 'go',
            draw_line = True, 
            line_color = 'gray',
            draw_zero = True, 
            zero_style = 'ro',
            line_w = 1,
            cmap = 'gray'
           ):
    """
    input:
    u -- complex array, contour points,
    method -- plt or cv2,
    margin -- 0..1, white space from the borders,
    show -- if True, plt show,
    only for plt mothod:
    label -- plt label,
    dot_style -- contour points plt marker style,
    draw_line -- bool, if True, contour edges draw,
    line_color -- plt colot of contour edges, if draw_line is True,
    draw_zero -- bool, if True, (0+0j) point draw,
    zero_style -- zero plt marker style, if draw_zero is True,
    only for cv2 method:
    line_w -- contour line width,
    cmap -- plt cmap to imshow.
    output:
    if plt method, None,
    if cv2 method, cv2 image
    """
    left, right = np.min(np.real(u)), np.max(np.real(u))
    down, up = np.min(np.imag(u)), np.max(np.imag(u))
    margin = (int)(max(right-left,up-down)*margin+1)
    x1 = min(left, down) - margin
    x2 = max(up, right) + margin
    if method == 'plt':
        plt.xlim(x1,x2)
        plt.ylim(x1,x2)
        plt.axis('equal')
        if draw_line:
            x = np.zeros(len(u)+1)
            x[:-1] = np.real(u)
            x[-1] = np.real(u[0])
            y = np.zeros(len(u)+1)
            y[:-1] = np.imag(u)
            y[-1] = np.imag(u[0])
            plt.plot(x, y, color=line_color)
        plt.plot(np.real(u), np.imag(u), dot_style, label = label)
        plt.grid()
        if draw_zero:
            plt.plot([0], [0], zero_style, label = 'zero')
        if label!='' or draw_zero:
            plt.legend()
        if show:
            plt.show()
    elif method == 'cv2':
        w = (int)(right-left+2*margin+1)
        h = (int)(up-down+2*margin+1)
        tmp_img = np.zeros((h,w))
        cnt = u_to_cnt(u - left - 1j * down + margin*(1+1j))
        cv2.drawContours(tmp_img, [cnt], 0, 255, line_w)
        if show:
            imshow_bw(tmp_img, cmap=cmap)
        return tmp_img

def imshow_bw(img, title = '', cmap = 'gray', ax = None, show = True):
    """
    input:
    img -- cv2 image,
    title -- plt title,
    cmap -- plt cmap,
    ax -- None of plt ax,
    show -- if True, plt show,
    output:
    None.
    Imshow img to plt or ax.
    """
    if ax is None:
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(255-img, cmap=cmap)
    else:
        ax.set_title(title,fontsize= 30)
        ax.imshow(255-img, cmap=cmap)
    if show:
        plt.show()