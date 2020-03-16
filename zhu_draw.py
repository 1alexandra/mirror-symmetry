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
            label = 'U',
            dot_style = 'go',
            draw_line = True,
            line_color = 'gray',
            draw_zero = True, 
            line_w = 1,
            cmap = 'gray'
           ):
    """
    input:
    u -- complex array, contour points,
    method -- plt or cv2,
    margin -- 0..1, white space from the borders,
    only for plt mothod:
    label -- plt label,
    dot_style -- contour points plt marker style,
    draw_line -- bool, if True, gray edges draw,
    line_color -- plt color, if draw_line is True,
    draw_zero -- bool, if True, (0+0j) point draw,
    only for cv2 method:
    line_w -- contour line width,
    cmap -- plt cmap to imshow.
    output:
    if plt method, None,
    if cv2 method, cv2 image
    """
    left, right = np.min(np.real(u)), np.max(np.real(u))
    down, up = np.min(np.imag(u)), np.max(np.imag(u))
    margin = (int)(max(right-left,up-down) * margin) + 1
    x1 = min(left, down) - margin
    x2 = max(up, right) + margin
    if method == 'plt':
        plt.xlabel('Re')
        plt.ylabel('Im')
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
            plt.plot(x, y, color = line_color)
        plt.plot(np.real(u), np.imag(u), dot_style, label = label)
        plt.grid()
        if draw_zero:
            plt.plot([0], [0], label = 'Zero(0,0)')
        if label!='' or draw_zero:
            plt.legend()
    elif method == 'cv2':
        w = (int)(right-left+2*margin+1)
        h = (int)(up-down+2*margin+1)
        tmp_img = np.zeros((h,w))
        cnt = u_to_cnt(u - left - 1j * down + margin*(1+1j))
        cv2.drawContours(tmp_img, [cnt], 0, 255, line_w)
        return tmp_img

def imshow_bw(img, title = '', cmap = 'gray', ax = None):
    """
    input:
    img -- cv2 image,
    title -- plt title,
    cmap -- plt cmap,
    ax -- None or plt ax,
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
        
def plot_measure(q):
    ind = np.argmin(q)
    plt.title('Asymmetry measure for contour points')
    plt.xlabel('p - point index')
    plt.ylabel('Q - asymmetry measure')
    plt.plot(q, label = 'Q(p)')
    plt.plot([ind], [q[ind]], 'ro', 
             label = 'min(Q) = ' + str(round(q[ind],3)))
    plt.grid()
    plt.legend()
    
def plot_hull(u, u_h, h, m):
    plt.title('Hull based search')
    contour(u, 'plt', label = 'contour points', dot_style = 'go', draw_zero = False)
    contour(m, 'plt', label = 'nearest to centroid-CH edge center line', 
            dot_style = 'co', draw_line = False, draw_zero = False)
    contour(h, 'plt', label = 'convex hull points', dot_style = 'ro', 
            draw_line=False, draw_zero = False)   
    contour(h, 'plt', label = 'convex hull', dot_style = 'y-', 
            line_color = 'yellow', draw_zero = False)     
    contour(u_h, 'plt', label = 'points to search axis', dot_style = 'b+', 
            draw_line = False, draw_zero = False)  
    centroid = u.mean()
    plt.plot([np.real(centroid)],[np.imag(centroid)],'yo',label='centroid')
    plt.grid()
    plt.legend()
    
def plot_f_abs(f, eps, garms, ind):
    plt.title("FD coefficients' importance")
    plt.xlabel('l - coefficien index (l>0)')
    plt.ylabel('importance')
    g = np.log(np.abs(f[1:]))
    left, right = -1, len(f)
    up, down = np.max(g) + 1, np.min(g) - 1
    plt.plot(np.arange(1,len(f)), g, label='log|f|')
    plt.plot([left, right],[np.log(eps), np.log(eps)], 'c--')
    plt.plot([garms//2, garms//2],[down, up], 'c-.')
    plt.plot([len(f)-garms//2, len(f)-garms//2],[down, up], 'c-.')
    plt.plot(ind, np.log(f[ind]), 'go')
    plt.grid()
    plt.legend()
    
    