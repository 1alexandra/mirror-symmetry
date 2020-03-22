import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab

from zhu_contour import u_to_cnt

def prepare_scene(w=9, h=6):
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

def line_in_rect(p, v, x1, x2, y1, y2):
    EPS = 1e-15
    v /= abs(v)
    if abs(np.real(v)) < EPS:
        return p.re + x1 * 1j, p.re + x2 * 1j 
    if abs(np.imag(v)) < EPS:
        return x1 + np.imag(p) * 1j, x2 + np.imag(p) * 1j
    points = []
    for x in [x1, x2]:    
        s = p + (x - np.real(p)) / np.real(v) * v
        if y1 - EPS <= np.imag(s) <= y2 + EPS:
            points.append(s)
    for y in [y1, y2]:
        s = p + (y - np.imag(p)) / np.imag(v) * v
        if x1 - EPS <= np.real(s) <= x2 + EPS:
            points.append(s)
    while len(points) != 2:
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if abs(points[i]-points[j]) < 2 * EPS:
                    points = points[:j]+points[j+1:]
                    break
                else:
                    print(abs(points[i]-points[j]))
            else:
                continue
            break
        else:
            print('line_in_rect error:', p, v, x1, x2, y1, y2)
            break
    return points
        

def draw_contour(
            method,
            u, 
            axis_point = None,
            axis_vec = None,
            label = 'U',
            point_marker = 'go',
            edge_color = 'gray', 
            axis_marker = 'r-',
            scale = 800,
            cmap = 'gray'
           ):
    """
    input:
    method -- plt or cv2,
    u -- complex array, contour points,
    axis_point -- complex, a point on axis,
    axis_vec -- complex, an axis direction,
    only for plt mothod:
    label -- plt label,
    point_marker -- contour points plt marker style,
    edge_color -- contour edges plt color,
    axis_marker -- axis line plt marker style,
    only for cv2 method:
    scale -- double: draw int(u*scale) contour
    line_w -- contour line width,
    cmap -- plt cmap to imshow.
    output:
    if plt method, None,
    if cv2 method, cv2 image
    """
    left, right = np.min(np.real(u)), np.max(np.real(u))
    down, up = np.min(np.imag(u)), np.max(np.imag(u))
    margin = max(right-left,up-down) * 0.1
    x1 = left - margin
    x2 = right + margin
    y1 = down - margin
    y2 = up + margin
    if method == 'plt':
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.xlim(x1,x2)
        plt.ylim(y1,y2)
        plt.axis('equal')
        if edge_color:
            x = np.zeros(len(u)+1)
            x[:-1] = np.real(u)
            x[-1] = np.real(u[0])
            y = np.zeros(len(u)+1)
            y[:-1] = np.imag(u)
            y[-1] = np.imag(u[0])
            plt.plot(x, y, color = edge_color)
        if point_marker:
            plt.plot(np.real(u), np.imag(u), point_marker, label = label)
        if axis_marker and axis_point and axis_vec:
            s1, s2 = line_in_rect(axis_point, axis_vec, x1, x2, y1, y2)
            plt.plot([np.real(s1),np.real(s2)],[np.imag(s1),np.imag(s2)], 
                     axis_marker, label = 'Symmetry axis')
        if label:
            plt.legend()
        plt.grid()
    elif method == 'cv2':
        if right > 10 or up > 10 or down < -10 or left < -10:
            scale = 1
        w = scale * (x2 - x1)
        h = scale * (y2 - y1)
        w, h = int(w), int(h)
        line_w = max(1, min(w,h)//100)
        img = np.zeros((h,w))
        vec = - left - 1j * down + margin*(1+1j)
        cnt = u_to_cnt(1j * h + np.conjugate((u + vec) * scale))
        cv2.drawContours(img, [cnt], 0, 255, line_w)
        if axis_point and axis_vec:
            s1, s2 = line_in_rect(1j * h + np.conjugate((axis_point + vec) * scale), 
                                  np.conjugate(axis_vec), 0, w, 0, h)
            x1, y1 = int(np.real(s1)), int(np.imag(s1))
            x2, y2 = int(np.real(s2)), int(np.imag(s2))
            cv2.line(img, (x1, y1), (x2, y2), 255, line_w)
        return img

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
        ax.set_title(title, fontsize= 30)
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
    
def draw_hull(u, u_h, h, m):
    plt.title('Hull based search')
    plt.grid()
    draw_contour('plt', u, label = 'contour points', 
                 point_marker = 'go')
    draw_contour('plt', m, label = 'centroid-CH edge', 
                 point_marker = 'co', edge_color = None)
    draw_contour('plt', h, label = 'convex hull points', 
                 point_marker = 'ro', edge_color = None)   
    draw_contour('plt', h, label = 'convex hull', point_marker = 'y-', 
                 edge_color = 'yellow')     
    draw_contour('plt', u_h, label = 'points to search axis', 
                 point_marker = 'b+', edge_color = None)  
    centroid = u.mean()
    plt.plot([np.real(centroid)],[np.imag(centroid)], 'yo', label='centroid')
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
    plt.plot(ind, np.log(np.abs(f[ind])), 'go')
    plt.grid()
    plt.legend()
    
def savefig(name, fmt = 'eps'):
    plt.savefig(name + '.' + fmt, format = fmt, bbox_inches = 'tight')
    
def bw_to_rgb(img):
    w, h = img.shape
    rgb = np.zeros((w, h, 3), dtype = int)
    for i in range(3):
        rgb[:,:,i] = img.copy()
    return rgb

def subploter(drawing_args, cols=5, figscale=5):
    """
    input:
    drawing_args -- list of [u, p, v, q], where 
        u -- complex array, contour points,
        p -- complex point on symmetry axis,
        v -- complex vector of symmetry axis direction,
        q -- double, measure of symmetry;
    cols -- int, number of columns in subplot,
    figscale -- double, scale factor to figsize;
    output:
    None.
    It draws by plt subplots with cv2 images 
    with contours and symmetry axises, titled by "Q=...".
    """
    rows = (len(drawing_args) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, 
                            figsize=(figscale*cols, figscale*rows))
    plt.setp(axs, xticks=[], yticks=[])
    for i, (u, p, v, q) in enumerate(drawing_args):
        row = i // cols
        col = i % cols
        imshow_bw(draw_contour('cv2', u, p, v),
                  title=f'Q = {round(q,3)}',
                  ax=axs[row][col])
