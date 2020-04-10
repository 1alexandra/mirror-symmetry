import numpy as np
from scipy.spatial import ConvexHull
from time import time
import zhu_contour as zc
import zhu_draw as zd
from matplotlib import pyplot as plt


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


def axis_points(u, p, vec):
    margin = len(u) // 4
    ind1 = nearest_to_line(u, p, p + vec)
    mask = np.ones(len(u), dtype=bool)
    mask[np.arange(ind1 - margin, ind1 + margin + 1) % len(u)] = False
    u_part = u[mask]
    ind_part2 = nearest_to_line(u_part, p, p + vec)
    p1, p2 = u[ind1], u_part[ind_part2]
    if (np.imag(p1) > np.imag(p2)
            or np.imag(p1) == np.imag(p2) and np.real(p1) > np.real(p2)):
        p1, p2 = p2, p1
    return p1, p2


def join_index(*inds):
    if len(inds) == 1:
        return np.unique(inds[0])
    return np.unique(np.concatenate(inds))


def hull_based_index(u, delta=None):
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
        by_hull = np.arange(len(u))
        hull_ind = np.array([], dtype=int)
        middles = np.array([], dtype=int)
        return by_hull, hull_ind, middles
    dots = np.array([np.real(u), np.imag(u)]).T
    hull_ind = ConvexHull(dots).vertices
    hull = u[hull_ind]
    centroid = np.mean(u)
    middles = []
    for i in range(len(hull)):
        z = (hull[i] + hull[(i + 1) % len(hull)]) / 2
        middles.append(nearest_to_line(u, z, centroid))
    middles = np.unique(middles)
    hull_mid = join_index(hull_ind, middles)
    by_hull = []
    for ind in hull_mid:
        by_hull += list(zc.index_neighbors(u, ind, delta))
    by_hull = np.unique(by_hull)
    return by_hull, hull_ind, middles


def f_abs_based_index(f, beta=1, ret_zero=False):
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
    garms = max(2, N * beta)
    ind = np.arange(N)
    crit = np.logical_or(ind <= garms/2, ind >= N - garms/2)
    crit[0] = ret_zero
    return ind[crit], garms


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
    if k1 == k2:
        if k3 == 0:
            print('find_theta error', dots)
            return 0
        t1 = np.pi / 4
    elif k3 == 0:
        t1 = 0
    else:
        t1 = 0.5 * np.arctan(k3 / (k1 - k2))
    t2 = t1 + 0.5 * np.pi

    def f(t):
        return (k1 * np.cos(t)**2
                + k2 * np.sin(t)**2
                + 2 * k3 * np.sin(t) * np.cos(t))

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
    b = np.imag(dots * np.exp(-1j * theta))
    return np.sum(b * b) ** 0.5 / N, theta


def new_start_point(f, s, ind=None):
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
    return f[ind] * np.exp(-1j * 2 * np.pi / N * ind * (N - s))


def find_local_minimas(q, q_th=np.inf):
    q = np.array(q, dtype=float)
    N = len(q)
    crit1 = q <= q[(np.arange(N) - 1) % N]
    crit2 = q <= q[(np.arange(N) + 1) % N]
    crit3 = q <= q_th
    return np.arange(N)[crit1 * crit2 * crit3]


def find_sym(
    u_,
    delta_hull=3,
    beta=0.2,
    delta_neib=10,
    n_mult=2,
    q_th=0.025,# np.inf,
    theta_eps=0.1,
    find_all=True
):
    u, vec, scale = zc.preprocess(zc.fix_period(u_, n_mult))
    f = np.fft.fft(u)
    N = len(u)
    by_hull, *_ = hull_based_index(u, delta_hull)
    f_ind, *_ = f_abs_based_index(f, beta)
    qs1 = [measure_axis(new_start_point(f, s, f_ind), N)[0]
           for s in by_hull]
    if find_all:
        ind_min_qs1 = by_hull[find_local_minimas(qs1)]
    else:
        ind_min_qs1 = [by_hull[np.argmin(qs1)]]
    neibs = []
    for approx_ind in ind_min_qs1: 
        neibs.append(zc.index_neighbors(u, approx_ind, delta_neib))
    neibs = join_index(*tuple(neibs))
    all_f_ind = np.arange(1, N)
    qs2 = [measure_axis(new_start_point(f, s, all_f_ind), N)[0]
           for s in neibs]
    if find_all:
        ind_min_qs2 = neibs[find_local_minimas(qs2)]
    else:
        ind_min_qs2 = [neibs[np.argmin(qs2)]]
    axs = []
    bad_axs = []
    for sym_ind in ind_min_qs2:
        q, theta = measure_axis(new_start_point(f, sym_ind, all_f_ind), N)
        sym_point = zc.preprocess_inverse(u[sym_ind], vec, scale)
        sym_vec = np.exp(1j * theta)
        if q <= q_th:
            axs.append((q, sym_point, sym_vec))
        else:
            bad_axs.append((q, sym_point, sym_vec))
    if not len(axs):
        bad_axs.sort(key=lambda x: x[0])
        axs = [bad_axs[0]]
    axs.sort(key=lambda x: np.angle(x[2]))
    last_theta = np.angle(axs[0][2])
    cur_ind = 1
    while cur_ind < len(axs):
        cur_theta = np.angle(axs[cur_ind][2])
        if cur_theta - last_theta < theta_eps:
            axs = axs[:cur_ind] + axs[cur_ind+1:]
        else:
            last_theta = cur_theta
            cur_ind += 1
    axs.sort(key=lambda x: x[0])
    return np.min(qs2), axs


def get_drawing_args(folder, get_all=True, from_txt=False):
    drawing_args = []
    for name, u_list in zc.from_folder(folder, get_all, from_txt).items():
        for u in u_list:
            Q, axs = find_sym(u)
            for (q, p, v) in axs:
                drawing_args.append([u, p, v, q, Q])
    drawing_args.sort(key=lambda x: x[-1])
    return drawing_args


def write_axis_points(
    img_folder,
    res_filename=None,
    get_all=False,
    from_txt=False
):
    if res_filename is None:
        rimg_folder = img_folder.rstrip('/').rstrip('\\')
        right_sep = img_folder.rfind('/')
        if right_sep == -1:
            right_sep = img_folder.rfind('\\')
        res_filename = img_folder[right_sep+1:] + '_results.txt'
    time_start = time()
    with open('results/' + res_filename, 'w') as file:
        for name, u_list in zc.from_folder(img_folder, get_all, from_txt).items():
            for u in u_list:
                Q, axs = find_sym(u)
                for (q, p, v) in axs:
                    p1, p2 = axis_points(u, p, v)
                    file.write(' '.join([name, str(p1), str(p2), str(q)]) + '\n')
        file.write('Total time: '+str(round(time() - time_start, 3)))
