from matplotlib import pyplot as plt

from zhu.draw_tools import prepare_scene, save_plot


class VS_Jaccard:
    def __init__(
        self,
        my_filename='result.txt',
        so_filename='result_sonya.txt',
        res_folder='results',
        dataset_name='Butterfly'
    ):
        self.res_folder = res_folder
        self.dataset_name = dataset_name
        self.my_res = self.read_my_res(my_filename)
        self.so_res = self.read_so_res(so_filename)

    def path(self, filename):
        return filename

    def read_my_res(self, filename):
        with open(self.path(filename)) as f:
            lines = f.readlines()
        words = [line[:-1].split(' ') for line in lines]
        res = {}
        for (name, p1, p2, q, t) in words:
            i = int(name)
            if i in res:
                res[i] = min(res[i], float(q))
            else:
                res[i] = float(q)
        return res

    def read_so_res(self, filename):
        with open(self.path(filename)) as f:
            lines = f.readlines()
        words = [line[:-1].split(' ') for line in lines]
        res = {}
        for (name, name_i, x1, y1, x2, y2, q) in words:
            i = int(name)
            if i in res:
                res[i] = max(res[i], float(q))
            else:
                res[i] = float(q)
        return res

    def scatter_plot(self, filename='q_vs_jaccard', fmt='png', figsize=10,
                     xlim=None, ylim=None, color=None):
        keys = list(self.my_res.keys())
        keys.sort()
        my = [self.my_res[key] for key in keys]
        so = [self.so_res[key] for key in keys]
        prepare_scene(figsize, figsize)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        plt.title(self.dataset_name + ' dataset')
        # plt.axis('equal')
        plt.xlabel('Q - asymmetry measure')
        # plt.xlim(-0.1, 1.1)
        plt.ylabel('J - Jaccard index')
        # plt.ylim(-0.1, 1.1)
        plt.scatter(my, so, label=self.dataset_name + ' objects',
                    color=color, alpha=0.7)
        plt.grid()
        plt.legend()
        if filename:
            save_plot(filename, fmt, self.res_folder)
