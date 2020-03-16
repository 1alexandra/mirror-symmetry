import numpy as np
import cv2
from scipy.spatial import ConvexHull

from zhu_contour import preprocess, index_neighbors, u_to_cnt, cnt_to_u
    
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
        return np.arange(len(u), np.array([]), dtype=int)
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
    
def find_sym(u, plot_q = False, delta = None, q_thresh = 100,
                     name=None, gauss=False, alpha=0, beta=1):
    u = preprocess(u)
    f = np.fft.fft(u)
    N = len(u)
    tmp_img = draw_ifft(f,False,line)
    by_hull, hull = hull_based_index(u, delta)
    f_ind = filter_f(f,alpha,beta)
    thrs = [rotate_calc(new_start_point(f,s,f_ind),N) for s in by_hull]
    first_true_index = by_hull[np.argmin(thrs)]
    x, y = np.real(u[first_true_index]), np.imag(u[first_true_index])
    ind = u_delta(u,x,y,delta)
    thrs = [rotate_calc(new_start_point(f, s), N) for s in ind]
    true_index = ind[np.argmin(thrs)]
    thr, theta = rotate_calc(new_start_point(f,true_index),N,True)
    x = -1
    y = -1
    if thr<thresh:
        x, y = np.real(u[-true_index]), np.imag(u[-true_index])
        v_x, v_y = np.cos(theta), np.sin(theta)
        c = (int)(800/tmp_img.shape[0])
        tmp_img = draw_ifft(f*c,False,line)   
        w, h = tmp_img.shape
        if draw_points:
            tmp = np.zeros((w,h,3),dtype=int)*255
            for inddd in range(3):
                tmp[:,:,inddd] = tmp_img.copy()
            tmp_img = tmp
        k = max(tmp_img.shape)
        cv2.line(tmp_img,(c*int(x),c*int(y)),(c*int(x-k*v_x),c*int(y-k*v_y)),(255,255,255),line)
        cv2.line(tmp_img,(c*int(x),c*int(y)),(c*int(x+k*v_x),c*int(y+k*v_y)),(255,255,255),line)
        if draw_points:
            for el in by_hull:
                x, y = np.real(u[-el]), np.imag(u[-el])
                cv2.circle(tmp_img,(c*int(x),c*int(y)),line,(0,255,255),line)
            for el in hull:
                x, y = np.real(u[-el]), np.imag(u[-el])
                cv2.circle(tmp_img,(c*int(x),c*int(y)),line*2,(255,255,0),line)
    if plot_q:

        
#         thrs_abs_sum = [rotate_calc(new_start_point(f, s), N, ret_abs_sum = True) for s in range(N)]
#         plt.plot(thrs_abs_sum,label = 'abs(sum)')
#         ind_min = np.argmin(thrs_abs_sum)
#         plt.plot([ind_min],[thrs_abs_sum[ind_min]],'bo', label = 'min abs(sum)')
#         thrs_sum_abs = [rotate_calc(new_start_point(f, s), N, ret_sum_abs = True) for s in range(N)]
#         plt.plot(thrs_sum_abs,label = 'sum(abs)')
#         ind_min = np.argmin(thrs_sum_abs)
#         plt.plot([ind_min],[thrs_sum_abs[ind_min]],'bo', label = 'min sum(abs)')
        
        plt.legend()
        if not name is None:
            plt.savefig(name[:-4]+'_q.png',format='png',bbox_inches='tight')
        plt.show()
#         plt.xlabel('coefficients')
#         plt.ylabel('importance')
#         plt.title('Fourier descriptor importance')
#         plt.plot(np.log(np.abs(f))[1:], label='log|f|')
#         plt.legend()
#         plt.show()
    if show:
        if i>0:
            draw_complex(new_start_point(f,true_index),True,False,'F, l>0',False)
            plt.savefig('plane_f'+str(i)+'.png',format='png')
            plt.show()
            if gauss:
                tmp_img = cv2.GaussianBlur(tmp_img, (3,3), 0)
            plt.imsave('plane_line'+str(i)+'.png',255-tmp_img,cmap='gray',format = 'png')
        plt.imshow(255-tmp_img,cmap='gray')
        plt.show()
    return tmp_img, x-np.real(vec), y-np.imag(vec), theta, thr, -true_index
