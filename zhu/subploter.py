import os
from time import time

import numpy as np
from matplotlib import pyplot as plt

from zhu import ContourDrawer
from zhu import DataFolder

from zhu.draw_tools import imshow_bw
from zhu.draw_tools import choose_cmap
from zhu.draw_tools import save_plot

from zhu import CMAP_OTHER
from zhu import Q_MAX_MULT


class Subploter:
    def __init__(
        self,
        cols=5,
        figscale=5,
        allow_multiple_lines=True,
        force_single_axis=True
    ):
        self.cols = cols
        self.figscale = figscale
        self.multiple = allow_multiple_lines
        self.force_single_axis = force_single_axis

    def contour(self, sym_cnt):
        d = ContourDrawer(sym_cnt)
        if self.multiple:
            q = sym_cnt.Sym_measure
            cmap = choose_cmap(sym_cnt.symmetrical())
            return [(d.draw(sym_cnt.Axis_list), q, cmap)]
        else:
            ans = []
            for sym_axis in sym_cnt.Axis_list:
                img = d.draw([sym_axis])
                q = sym_axis.q
                cmap = choose_cmap(q < sym_cnt.q_max_pix, None)
                ans.append((img, q, cmap))
            return ans

    def image(self, sym_image, last_cmap=None, q_max=np.inf):
        ans = []
        last_cmap = last_cmap if not self.multiple else None
        cmap = last_cmap
        for cnt in sym_image:
            cnt_res = self.contour(cnt)
            tail = len(cnt_res)
            for i in range(tail):
                img, q, cmap = cnt_res[i]
                if i == 0 or q < q_max or not self.force_single_axis:
                    if cmap != CMAP_OTHER:
                        cmap = choose_cmap(True, last_cmap)
                    cnt_res[i] = img, q, cmap
                else:
                    tail = i
                    break
            if not self.multiple:
                last_cmap = cmap
            ans += cnt_res[:tail]
        return ans

    def folder(self, data_folder, force_sort=False, q_max_mult=Q_MAX_MULT):
        ans = []
        last_cmap = None
        data = [cs for cs in data_folder if cs.Sym_measure is not None]
        q_max = max([cs.Sym_measure for cs in data]) * q_max_mult
        if data_folder.single or force_sort:
            data.sort(key=lambda x: x.Sym_measure)
        for sym_image in data:
            cur = self.image(sym_image, last_cmap, q_max)
            if len(cur):
                last_cmap = cur[-1][-1]
                ans += cur
        return ans

    def parent_folder(
        self,
        main_folder,
        subfolders=None,
        res_folder='../subploter_results',
        format_='png',
        single=True,
        force_sort=True,
        log=True
    ):
        if subfolders is None:
            subfolders = os.listdir(path='./' + main_folder)
        for folder in subfolders:
            if log:
                print(folder)
                start = time()
            df = DataFolder(main_folder + '/' + folder, single=single)
            self.plot(self.folder(df, force_sort=force_sort))
            save_plot(
                folder.split('/')[-1],
                format_=format_,
                res_folder=res_folder)
            if log:
                print(f'Time = {round(time()-start, 3)} s.')
                plt.show()
            else:
                plt.clf()

    def plot(self, img_q_cmap_list):
        if not len(img_q_cmap_list):
            return
        if len(img_q_cmap_list) == 1:
            imshow_bw(img_q_cmap_list[0])
            return
        cols = self.cols
        rows = (len(img_q_cmap_list) + cols - 1) // cols
        size = (self.figscale * cols, self.figscale * rows)
        fig, axs = plt.subplots(rows, cols, figsize=size)
        plt.setp(axs, xticks=[], yticks=[])
        for i, (img, q, cmap) in enumerate(img_q_cmap_list):
            row = i // cols
            col = i % cols
            ax = axs[row][col] if rows != 1 else axs[col]
            imshow_bw(img, q, cmap, ax)

    def __str__(self):
        return 'Subploter with cols={self.cols}, figscale={self.figscale}, \
                allow_multiple_lines={self.multiple}'
