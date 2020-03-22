import os
import numpy as np
import cv2
from scipy.spatial import ConvexHull

from zhu_contour import preprocess, preprocess_inverse, index_neighbors 
from zhu_contour import u_to_cnt, cnt_to_u, get_contours

def nearest_to_line(u, z1, z2):
    """
    input:
    x -- complex point or numpy array of complex points,
    z1, z2 -- complex points
    output:
    a distance between point x and a line passing throuth z1 and z2 points 
    """
    
    re_z1, im_z1 = np.real(z1), np.imag(z1)
    re_z2, im_z2 = np.real(z2), np.imag(z2)
    dists = np.abs((im_z2 - im_z1) * np.real(u)
                   - (re_z2 - re_z1) * np.imag(u)
                   + im_z1 * re_z2 - im_z2 * re_z1)
    return np.argmin(dists)


def hull_based_index(u, delta = None):
    """
    input:
    u -- complex array, contour points,
    delta -- scale of neighboorhood,
    output:
    by_hull -- points to search symmetry axis,
    hull_mid -- hull points and points on line passing through centroid and 
            convex hull edge centres, without neighboors,
    hull_ind -- convex hull points.
    """
    if delta is None:
        return np.arange(len(u)), np.array([],dtype=int), np.array([],dtype=int)
    dots = np.array([np.real(u),np.imag(u)]).T
    hull_ind = ConvexHull(dots).vertices
    hull = u[hull_ind]
    centroid = np.mean(u)
    middles = []
    for i in range(len(hull)):
        z = (hull[i] + hull[(i+1)%len(hull)]) / 2
        middles.append(nearest_to_line(u, z, centroid))
    middles = np.array(middles)
    hull_mid = np.unique(np.ravel([hull_ind, middles]))
    by_hull = []
    for ind in hull_mid:
        by_hull += list(index_neighbors(u, u[ind], delta))
    by_hull = np.unique(np.ravel(by_hull))
    return by_hull, hull_ind, middles
    
def f_abs_based_index(f, alpha = 0, beta = 1, ret_zero = False):
    """
    input:
    f -- complex array, Fourier Descriptor (FD),
    alpha -- double, 0..1,
    beta -- double, 0..1,
    ret_zero -- bool, if False, 0 is excluded from result 
    output:
    array of indexes 'ind' of FD, where abs(f) >= 'eps' and
        'ind' in the first or the last len(f)*beta/2 coefficients,
        alpha = 0 => eps = min(abs(f[1:])),
        alpha = 1 => eps = max(abs(f[1:])),
        eps(alpha) is linear.
    """
    N = len(f)
    a1 = np.min(np.abs(f[1:]))
    a2 = np.max(np.abs(f[1:]))
    eps = a1+(a2-a1)*alpha
    garms = N*beta   
    ind = np.arange(N)
    crit_1 = np.abs(f) >= eps
    crit_1[0] = ret_zero
    crit_2 = np.logical_or(ind <= garms/2, ind >= N-garms/2)
    return ind[crit_1 * crit_2], eps, garms
    
def find_theta(dots):
    """
    input:
    dots -- array of complex points;
    output:
    optimal angle of line passing through all dots in radians.
    """
    a = np.real(dots)
    b = np.imag(dots)
    k1 = np.sum(b * b)
    k2 = np.sum(a * a)
    k3 = 2 * np.sum(a * b)
    f = lambda t: k1*np.cos(t)**2 + k2*np.sin(t)**2 + 2*k3*np.sin(t)*np.cos(t)
    if k1 == k2:
        if k3 == 0:
            print('find_theta error')
            return 0
        t1 = np.pi/4
    elif k3 == 0:
        t1 = 0
    else:
        t1 = 0.5 * np.arctan(k3/(k1-k2))
    t2 = t1 + 0.5 * np.pi
    if f(t1) < f(t2):
        if f(0) > f(t1):
            return - t1 % np.pi
    else:
        if f(0) > f(t2):
            return - t2 % np.pi
    return 0
    
def measure_axis(dots, N): 
    """
    input:
    dots -- array of complex, Fourier Descriptor (FD) 
        subsequence of coefficients, 
    N -- contour points number (in order to normalize),
    output:
    double measure,
    double angle theta in radians
    """
    theta = find_theta(dots)
    b = np.imag(dots * np.exp(-1j*theta))
    # было (np.sum(b*b)/N)**0.5
    ### TODO может быть нормировать на количество точек?
    ### TODO может быть нормировать на max(abs(dot))?
    ### TODO может быть смотреть не на квадрат мнимой части, а на модуль?
    return np.sum(b*b)**0.5 / N, theta

def new_start_point(f, s, ind = None):
    """
    input:
    f -- complex array, Fourier descriptor (FD),
    s -- new index of starting point,
    ind -- None or array of int, indexes where to calculate new FD.
    output:
    fd_new if ind is None, else fd_new[ind] -- complex array
    """
    N = len(f)
    ind = ind if (ind is not None) else np.arange(N)
    return f[ind] * np.exp(-1j * 2*np.pi/N * ind * (N-s)) ### было s   

def find_sym(u_, 
             delta_hull = None, 
             alpha = 0, 
             beta = 1, 
             delta_neib = None, 
             q_th = np.inf
            ):
    u, vec, scale = preprocess(u_)
    f = np.fft.fft(u)
    N = len(u)
    by_hull, *_ = hull_based_index(u, delta_hull)
    f_ind, *_ = f_abs_based_index(f, alpha, beta)
    qs1 = [measure_axis(new_start_point(f, s, f_ind), N)[0] 
           for s in by_hull]
    approx_ind = by_hull[np.argmin(qs1)]
    neibs = index_neighbors(u, u[approx_ind], delta_neib)
    all_f_ind = np.arange(1,N)
    qs2 = [measure_axis(new_start_point(f, s, all_f_ind), N)[0] 
           for s in neibs]
    sym_ind = neibs[np.argmin(qs2)]
    q, theta = measure_axis(new_start_point(f, sym_ind, all_f_ind), N)
    sym_point, sym_vec = None, None
    if q <= q_th:
        sym_point = preprocess_inverse(u[sym_ind], vec, scale)
        sym_vec = np.exp(1j*theta)
    return q, (sym_point, sym_vec) 
    
def get_drawing_args(folder):
    drawing_args = []
    for name in os.listdir(path='./'+folder):
        try:
            u_list = get_contours(folder+'/'+name)
        except BaseException:
            print(folder+'/'+name, 'error')
            continue
        for u in u_list:
            q, (p, v) = find_sym(u)
            drawing_args.append([u, p, v, q])
    drawing_args.sort(key=lambda x: x[-1])
    return drawing_args
