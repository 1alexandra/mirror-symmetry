import os
from time import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import zhu.contour as zc
import zhu.symmetry as zs
from zhu.draw import prepare_scene, save_plot
from other.rgb import MplColorHelper


class ChangeParameters:
    def __init__(
        self,
        data_folder='data/to_visapp/planes',
        beta_list=np.arange(0.01, 1.01, 0.01),
        mult_list=np.arange(0.1, 5.1, 0.1),
        res_folder='results',
        n_graphs=None
    ):
        self.data_folder = data_folder
        self.beta_list = beta_list
        self.mult_list = mult_list
        self.res_folder = res_folder
        self.n_graphs = n_graphs
        self.t_title = 'Time (sec.)'
        self.n_title = 'New N'
        self.q_title = 'Asymmetry measure'
        self.theta_title = 'Axis angle'

    def _get_new_n(self, u, p, v, mult, beta):
        u_new, vec, scale = zc.preprocess(zc.fix_period(u, mult))
        f_new = np.fft.fft(u_new)
        p_new = (p - vec) / scale
        sym_ind = zs.nearest_to_line(u_new, p_new, p_new + v)
        f_sym = zs.new_start_point(f_new, sym_ind)
        f_ind, *_ = zs.f_abs_based_index(f_sym, beta)
        return len(f_ind)

    def _get_args(self, u, u_pix, mult=2, beta=1):
        start = time()
        q, axs = zs.find_sym(u, mult=mult, beta=beta)
        q, p, v = axs[0]
        t = time() - start
        theta = np.angle(v)
        p1, p2 = zs.axis_points(u_pix, p, v)
        new_n = self._get_new_n(u, p, v, mult, beta)
        return {
            self.theta_title: theta,
            self.q_title: q,
            'Re(p1)': np.real(p1),
            'Im(p1)': np.imag(p1),
            'Re(p2)': np.real(p2),
            'Im(p2)': np.imag(p2),
            self.n_title: new_n,
            self.t_title: t
        }

    def _construct_arg_dicts(self):
        true_img, beta_txt, mult_txt = {}, {}, {}
        u_dict = zc.from_folder(self.data_folder, get_all=False, from_txt=True)
        for name_txt in tqdm(u_dict):
            name, ext = os.path.splitext(name_txt)
            name_img = self.data_folder + '/' + name + '.bmp'
            u_img = zc.get_contours(name_img, get_all=False)[0]
            true_img[name] = self._get_args(u_img, u_img)
            u_txt = u_dict[name_txt][0]
            if len(self.beta_list):
                beta_txt[name] = [self._get_args(u_txt, u_img, beta=beta)
                                  for beta in self.beta_list]
            if len(self.mult_list):
                mult_txt[name] = [self._get_args(u_txt, u_img, mult=mult)
                                  for mult in self.mult_list]
        return true_img, beta_txt, mult_txt

    def _plot_arg_changing(
        self, titles, keys,
        arg_name, arg_list,
        img_args, txt_args
    ):
        color_helper = MplColorHelper('hsv', 0, self.n_graphs or len(keys))
        colors = color_helper.get_rgb_index()
        cols, rows = 2, (len(titles) + 1) // 2
        prepare_scene()
        fig, axs = plt.subplots(rows, cols, figsize=(18*cols, 15*rows))
        for i, title in enumerate(titles):
            row, col = i // cols, i % cols
            ax = axs[row][col]
            ax.set_title(title)
            ax.set_xlabel(arg_name)
            ax.set_ylabel('value')
            for j, name in enumerate(keys):
                c = colors[j]
                ys = [args[title] for args in txt_args[name]]
                ax.plot(arg_list, ys, c=c, label=name)
                ax.plot([arg_list[-1]], [img_args[name][title]], 'o', c=c)
            ax.grid()
            ax.legend()

    def run(self):
        titles = [
            self.theta_title, self.q_title,
            'Re(p1)', 'Im(p1)',
            'Re(p2)', 'Im(p2)',
            self.n_title, self.t_title
        ]
        true_img, beta_txt, mult_txt = self._construct_arg_dicts()
        beta_args = ('Beta', self.beta_list, beta_txt)
        mult_args = ('N multiply coefficient', self.mult_list, mult_txt)
        for arg_name, arg_list, arg_txt in [beta_args, mult_args]:
            keys = np.array(list(arg_txt.keys()))
            index = np.random.permutation(len(keys))
            if self.n_graphs is not None:
                index = np.sort(index[:self.n_graphs])
            keys = keys[index]
            self._plot_arg_changing(
                titles, keys,
                arg_name, arg_list,
                true_img, arg_txt)
            save_plot(arg_name+'_changing')
            plt.show()
