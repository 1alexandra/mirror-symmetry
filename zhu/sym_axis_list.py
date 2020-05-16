import numpy as np

from zhu import Point
from zhu import FourierDescriptor
from zhu import SymAxis

from zhu.tools import join_neibs
from zhu.tools import local_minimas

from matplotlib import pyplot as plt


class SymAxisList:
    def __init__(self, lines, scaler=None):
        self.lines = lines
        self.scaler = scaler
        self.neibs_trained = None

    def select(self, q_max=np.inf, n_max=np.inf):
        if self.lines is None:
            return self
        goods = [line for line in self.lines if line.q <= q_max]
        if len(goods):
            goods.sort(key=lambda x: x.Vec.angle)
            i = 1
            while i < len(goods):
                if i > 0 and goods[i].Vec.collinear(goods[i - 1].Vec):
                    rm_i = i if goods[i].q > goods[i - 1].q else i - 1
                    goods[:] = goods[:rm_i] + goods[rm_i + 1:]
                else:
                    i += 1
            goods.sort(key=lambda x: x.q)
            if len(goods) > n_max:
                goods[:] = goods[:n_max]
            return SymAxisList(goods, self.scaler)
        best_bad = min(self.lines, key=lambda x: x.q)
        return SymAxisList([best_bad], self.scaler)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        return self.lines[i]

    def _inds_old(self, u):
        if self.lines is None:
            return np.array([], dtype=int)
        ans = []
        for line in self.lines:
            ans += line.on_line_ind(u)
        return ans

    def _inds_new(self, q_list):
        if not len(q_list):
            return np.array([], dtype=int)
        return local_minimas(q_list)

    def _refined(self, u, q_list, thetas, inds_new):
        if not len(inds_new):
            return SymAxisList(None, self.scaler)
        lines = []
        for i in inds_new:
            p, q, theta = u[i], q_list[i], thetas[i]
            p1, p2 = Point(p), Point(p + 10*np.exp(1j * theta))
            lines.append(SymAxis(p1, p2, q, u))
        return SymAxisList(lines, self.scaler)

    def refinement(self, u, neibs, beta, train=False, log=False):
        if not len(self.lines) or not len(u):
            return []
        d, n = FourierDescriptor(u, self.scaler), len(u)
        q_list, thetas = np.ones(n) * np.nan, np.ones(n) * np.nan
        inds = np.arange(len(u))
        if self.lines is not None:
            inds_old = self._inds_old(u)
            inds = join_neibs(u, inds_old, neibs_coef=neibs, min_neibs=3)
        for i in inds:
            f = d.f_s(i, beta)
            thetas[i] = d.angle(i, f=f)
            q_list[i] = d.symmetry_measure(i, f=f, theta=thetas[i])
        inds_new = self._inds_new(q_list)
        if train:
            self.train_neibs(inds_old, inds_new, n)
        if log:
            if self.neibs_trained is not None:
                nt = self.neibs_trained
                print(f"neibs = {int(round(nt * n))} / {n} = {nt}")
            else:
                print('no neibs trained')
            self.plot_refinement(n, q_list, inds_old, inds_new)
        return self._refined(u, q_list, thetas, inds_new)

    def train_neibs(self, inds_old, inds_new, n):
        i0 = np.array(inds_old).reshape((-1, 1))
        i1 = np.array(inds_new).reshape((1, -1))
        d0 = np.abs(i1-i0)
        d1 = n - d0
        d0[d0 > d1] = d1[d0 > d1]
        self.neibs_trained = np.max(np.min(d0, axis=0)) / n

    def plot_refinement(self, n, q_list, inds_old, inds_new):
        plt.xlim(0, n)
        plt.plot(q_list, label='Q')
        plt.plot(inds_old, q_list[inds_old], 'ro', label='Last Index')
        plt.plot(inds_new, q_list[inds_new], 'g.', label='New Index')
        plt.legend()
        plt.show()

    def __str__(self):
        return '\n'.join([
            f'SymAxisList with lines=',
            *[str(line) for line in self.lines]
        ])
