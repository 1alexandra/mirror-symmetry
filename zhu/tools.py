import numpy as np

from zhu import EPS


def close(a, b, eps=EPS):
    return abs(a - b) < eps


def close_np(a, b, eps=EPS):
    a_ = np.array(a)
    b_ = np.array(b)
    return np.all(np.abs(a_ - b_) < eps)


def between(a, x, b, eps=EPS):
    return a - eps < x < b + eps


def unique(x):
    exclude = []
    for i in range(len(x)):
        if i in exclude:
            continue
        for j in range(i + 1, len(x)):
            if j in exclude:
                continue
            if len(x) - len(exclude) <= 2:
                break
            if x[i] == x[j]:
                exclude.append(j)
    return [p for i, p in enumerate(x) if i not in exclude]


def round_complex(u):
    re = np.round(np.real(u)).astype(np.int32)
    im = np.round(np.imag(u)).astype(np.int32)
    return re + 1j * im


def index_neibs(u, i, neibs_coef, min_neibs=0):
    neibs = int(round(len(u) * neibs_coef))
    neibs = max(neibs, min_neibs)
    if neibs >= len(u):
        return np.arange(len(u))
    return np.arange(i - neibs // 2, i + neibs - neibs // 2 + 1) % len(u)


def join_index(*inds):
    if len(inds) == 1:
        return np.unique(inds[0])
    return np.unique(np.concatenate(inds))


def join_neibs(u, index, neibs_coef, min_neibs=0):
    inds = [index_neibs(u, i, neibs_coef, min_neibs) for i in index]
    return join_index(*tuple(inds))


def local_minimas(q, q_th=np.inf):
    q = np.array(q, dtype=float)
    N = len(q)
    q_next = q[(np.arange(N) + 1) % N]
    q_last = q[(np.arange(N) - 1) % N]
    with np.errstate(invalid='ignore'):
        crit1 = q <= q_last
        crit2 = q <= q_next
        crit3 = q <= q_th
    return np.arange(N)[crit1 & crit2 & crit3]
