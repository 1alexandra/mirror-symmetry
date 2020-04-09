import numpy as np

from time import time
from matplotlib import pyplot as plt

import zhu_draw as zd
import zhu_contour as zc
import zhu_symmetry as zs


def vs_jaccard(
        my_result_file='result.txt',
        sonya_result_file='result_sonya.txt'
):
    with open('result.txt') as file1:
        my_res = file1.readlines()
    with open('result_sonya.txt') as file2:
        so_res = file2.readlines()
    so_res = {i+1: float(el[:-1]) for i, el in enumerate(so_res)}
    my_res = [el[:-1].split(' ') for el in my_res]
    my_res = {int(name[:-4]): float(q) for name, p1, p2, q in my_res[:-1]}
    index = np.argsort(list(my_res.values()))
    my_x = np.array(list(my_res.keys()))[index]
    my_y = np.array(list(my_res.values()))[index]
    so_x = np.array(list(so_res.keys()))[index]
    so_y = np.array(list(so_res.values()))[index]
    zd.prepare_scene()
    plt.xlabel('Q - asymmetry measure')
    plt.ylabel('J - Jaccard index')
    plt.scatter(my_y, so_y)
    plt.grid()
    zd.savefig('jaccard')
    plt.show()


def change_find_sym_args(u, u_img, mult=2, beta=1):
    start = time()
    q, (p, v) = zs.find_sym(u, n_mult=mult, beta=beta)
    t = time() - start
    theta = np.angle(v)
    p1, p2 = zs.axis_points(u_img, p, v)
    u_new, vec, scale = zc.preprocess(zc.fix_period(u, mult))
    f_new = np.fft.fft(u_new)
    p_new = (p - vec) / scale
    sym_ind = zs.nearest_to_line(u_new, p_new, p_new + v)
    f_sym = zs.new_start_point(f_new, sym_ind)
    f_ind, *_ = zs.f_abs_based_index(f_sym, beta)
    return q, p1, p2, theta, t, f_sym[f_ind]


def construct(folder, beta_list, mult_list, get_all=False):
    u_lists = zc.from_folder(folder, get_all=get_all, from_txt=True)
    true_img = {}
    beta_txt = {}
    mult_txt = {}
    for name_txt in u_lists:
        name = name_txt[:-4]
        print(name)
        name_img = folder + '/' + name + '.bmp'
        u_img = zc.get_contours(name_img)[0]
        true_img[name] = change_find_sym_args(u_img, u_img)
        u_txt = u_lists[name_txt][0]
        beta_txt[name] = [change_find_sym_args(u_txt, u_img, beta=beta)
                          for beta in beta_list]
        mult_txt[name] = [change_find_sym_args(u_txt, u_img, mult=mult)
                          for mult in mult_list]
    ans = [true_img]
    if len(beta_list):
        ans.append(beta_txt)
    if len(mult_list):
        ans.append(mult_txt)
    return tuple(ans)


def plot_results(arg_name, arg_list, true_img, arg_txt):

    def dummy(x): return x

    args = [
        ('Theta', 3, dummy),
        ('Q', 0, dummy),
        ('Re(p1)', 1, np.real),
        ('Im(p1)', 1, np.imag),
        ('Re(p2)', 2, np.real),
        ('Im(p2)', 2, np.imag),
        ('new N', 5, len),
        ('Time (sec.)', 4, dummy)
    ]
    all_keys = np.array(list(arg_txt.keys()))
    keys = all_keys.copy()
    keys = all_keys[np.sort(np.random.permutation(len(all_keys))[:9])]
    cols = 2
    rows = (len(args) + cols - 1) // cols
    zd.prepare_scene()
    fig, axs = plt.subplots(rows, cols,
                            figsize=(18*cols, 15*rows))
    for i, (title, ind, func) in enumerate(args):
        row = i // cols
        col = i % cols
        ax = axs[row][col]
        ax.set_xlabel(arg_name)
        ax.set_ylabel('value')
        for i, name in enumerate(keys):
            values = [func(el[ind]) for el in arg_txt[name]]
            ax.plot(arg_list, values, color='C'+str(i+1), label=name)
            if title not in ['new N', 'Time (sec.)']:
                ax.plot([arg_list[-1]], [func(true_img[name][ind])],
                        'o', color='C'+str(i+1))
        ax.set_title(title)
        ax.legend()
        ax.grid()
    zd.savefig(arg_name + '_changing')
    plt.show()

def example():
    folder = 'data/to_visapp/planes'
    beta_list = np.arange(0.01, 1.01, 0.01)
    mult_list = np.arange(0.1, 5.1, 0.1)

    true_img, beta_txt, mult_txt = construct(folder, beta_list, mult_list)
    for arg_name, arg_list, arg_txt in [('beta', beta_list, beta_txt),
                                        ('mult coefficient', mult_list, mult_txt)]:
        plot_results(arg_name, arg_list, true_img, arg_txt)