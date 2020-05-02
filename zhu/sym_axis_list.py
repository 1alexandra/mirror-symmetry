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

    def select(self, q_max=np.inf):
        goods = [line for line in self.lines if line.q <= q_max]
        if len(goods):
            goods.sort(key=lambda x: x.Vec.angle)
            i = 1
            while i < len(goods):
                if i > 0 and goods[i].Vec.collinear(goods[i-1].Vec):
                    rm_i = i if goods[i].q > goods[i-1].q else i-1
                    goods[:] = goods[:rm_i] + goods[rm_i+1:]
                else:
                    i += 1
            goods.sort(key=lambda x: x.q)
            return SymAxisList(goods, self.scaler)
        best_bad = min(self.lines, key=lambda x: x.q)
        return SymAxisList([best_bad], self.scaler)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        return self.lines[i]

    def refinement(self, u, neibs, beta, log=False):
        if not len(self.lines) or not len(u):
            return []
        d, n = FourierDescriptor(u, self.scaler), len(u)
        q_list, thetas = np.ones(n) * np.nan, np.ones(n) * np.nan
        inds = np.ravel([[line.vertexes_ind(u)] for line in self.lines])
        for i in join_neibs(u, inds, neibs):
            f = d.f_s(i, beta)
            thetas[i] = d.angle(i, f=f)
            q_list[i] = d.symmetry_measure(i, f=f, theta=thetas[i])
        inds_new = local_minimas(q_list)
        if not len(inds_new):
            return []
        lines = []
        for i in inds_new:
            p, q, theta = u[i], q_list[i], thetas[i]
            p1, p2 = Point(p), Point(p + np.exp(1j * theta))
            lines.append(SymAxis(p1, p2, q))
        if log:
            plt.xlim(0, n)
            plt.plot(q_list, label='Q')
            plt.plot(inds, q_list[inds], 'ro', label='Last Index')
            plt.plot(inds_new, q_list[inds_new], 'g.', label='New Index')
            plt.legend()
            plt.show()
        return SymAxisList(lines, self.scaler)

    def __str__(self):
        return '\n'.join([
            f'SymAxisList with lines=',
            *[str(line) for line in self.lines]
        ])
