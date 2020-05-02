from matplotlib import pyplot as plt

from zhu.draw import prepare_scene, save_plot


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
        return self.res_folder + '/' + filename

    def read_my_res(self, filename):
        with open(self.path(filename)) as f:
            lines = f.readlines()
        words = [line[:-1].split(' ') for line in lines[:-1]]
        res = {int(name[:-4]): float(q) for (name, p1, p2, q) in words}
        return res

    def read_so_res(self, filename):
        with open(self.path(filename)) as f:
            lines = f.readlines()
        res = {i+1: float(line[:-1]) for i, line in enumerate(lines)}
        return res

    def scatter_plot(self, filename='q_vs_jaccard', fmt='png'):
        keys = list(self.my_res.keys())
        keys.sort()
        my = [self.my_res[key] for key in keys]
        so = [self.so_res[key] for key in keys]
        prepare_scene(10, 10)
        plt.title(self.dataset_name + ' dataset')
        # plt.axis('equal')
        plt.xlabel('Q - asymmetry measure')
        # plt.xlim(-0.1, 1.1)
        plt.ylabel('J - Jaccard index')
        # plt.ylim(-0.1, 1.1)
        plt.scatter(my, so, label=self.dataset_name + ' objects', alpha=0.7)
        plt.grid()
        plt.legend()
        if filename:
            save_plot(filename, fmt, self.res_folder)
