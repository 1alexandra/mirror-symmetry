import numpy as np
import cv2

from zhu_contour import preprocess, index_neighbors, u_to_cnt
    
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
    ind = ind if ind is not None else np.arange(N)
    return f[ind] * np.exp(-1j * 2*np.pi/N * ind * s)    
    
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

def dot_to_line(x, z1, z2):
    """
    input:
    x -- complex point or numpy array of complex points,
    z1, z2 -- complex points
    output:
    a distance between point x and a line passing throuth z1 and z2 points 
    """
    re_z1, im_z1 = np.real(z1), np.imag(z1)
    re_z2, im_z2 = np.real(z2), np.imag(z2)
    return np.abs((im_z2 - im_z1) * np.real(x)
                   - (re_z2 - re_z1) * np.imag(x)
                   + im_z1 * re_z2 - im_z2 * re_z1)


def hull_based_index(u, delta = None, mid_iters = 1):
    """
    input:
    u -- numpy array of complex contour points,
    delta -- scale of neighboorhood,
    middle_iters -- how many times edges was split in half;
    output:
    by_hull -- points to search symmetry axis,
    hull -- hull points and points on line passing through centroid and 
            convex hull edge centres, without neighboors.
    """
    if delta is None:
        return np.arange(len(u)), np.array([], dtype=int)
    scale = 2 ** mid_iters
    hull0 = cv2.convexHull(u_to_cnt(u * scale), returnPoints = False) 
    centroid = np.mean(u)
    hull0 = np.ravel(hull0)
    hull_u = u[hull0]
    middles0 = []
    for i in range(len(hull0)):
        next_i = (i+1)%len(hull_u)
        bet_z = np.mean([hull_u[i],hull_u[next_i]])
        end = hull0[i]
        start = hull0[next_i]
        if start > end:
            ind = np.concatenate((np.arange(start,len(u)),np.arange(0,end)))
        else:
            ind = np.arange(start,end)
        middle_ind = ind[np.argmin(dot_to_line(u[ind], bet_z, centroid))]
        middles0.append(middle_ind)
    middles0 = np.array(middles0)
    hull1 = np.unique(np.ravel(np.array([hull0,middles0],dtype = int).T))
    hull_delta = []
    for ind in hull1:
        hull_delta += list(index_neighbors(u, u[ind], delta))
    hull2 = np.unique(np.ravel(np.array(hull_delta,dtype = int).T))
    return np.arange(len(u))[hull2], np.arange(len(u))[hull1]
    
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
    return ind[crit_1 * crit_2]
    
def measure_axis(theta, dots, N): ### здесь не было theta!
    """
    input:
    theta -- angle of measuring axis to x-axis, radians,
    dots -- array of complex, Fourier Descriptor (FD) 
        subsequence of coefficients, 
    N -- contour points number (in order to normalize),
    output:
    double, measure
    """
    b = np.imag(dots * np.exp(-1j*theta))
    # было (np.sum(b*b)/N)**0.5
    ### TODO может быть нормировать на количество точек?
    ### TODO может быть нормировать на max(abs(dot))?
    ### TODO может быть смотреть не на квадрат мнимой части, а на модуль?
    return np.sum(b*b)**0.5 / N