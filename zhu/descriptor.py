import numpy as np

from zhu import Scaler


class FourierDescriptor:
    def __init__(self, signal, scaler=None):
        self.scaler = scaler or Scaler(signal)
        self.signal = self.scaler.predict(signal)
        self.n = len(signal)
        self.f_0 = np.fft.fft(self.signal)

    def criteria(self, beta, ret_zero):
        N = self.n
        index = np.arange(N)
        beta = min(beta, 1.0)
        if beta != 1:
            garms = round(max(2, N * beta))
            left_part = garms // 2
            right_part = garms - left_part
            crit = np.logical_or(index <= left_part, index >= N - right_part)
        else:
            crit = np.ones(N, dtype=bool)
        crit[0] = ret_zero
        return crit

    def f_s(self, s, beta=1.0, ret_zero=False):
        """FD coefficients with starting point cnt[s].

        Return only max(2, N*beta) main coefficients, excluding zero.
        Return zero coefficient based on a ret_zero parameter.
        """
        N = self.n
        index = np.arange(N)
        crit = self.criteria(beta, ret_zero)
        index = index[crit]
        coefs = np.exp(-1j * (2 * np.pi / N) * (N - s) * index)
        return coefs * self.f_0[index]

    def smoothed(self, beta):
        f = self.f_0.copy()
        crit = self.criteria(beta, True)
        f[~crit] = 0.0 + 0.0j
        sm = np.fft.ifft(f)
        return self.scaler.inverse(sm)

    def angle(self, s, beta=1.0, f=None):
        f = f if f is not None else self.f_s(s, beta)
        a, b = np.real(f), np.imag(f)
        k1, k2, k3 = np.sum(b * b), np.sum(a * a), 2 * np.sum(a * b)
        if k1 == k2:
            if k3 == 0:
                return 0
            t1 = np.pi / 4
        elif k3 == 0:
            t1 = 0
        else:
            t1 = np.arctan(k3 / (k1 - k2)) / 2
        ts = [0, t1, t1 + np.pi / 2]
        t_min = min(ts, key=lambda t: (
            k1 * np.cos(t)**2
            + k2 * np.sin(t) ** 2
            + 2 * k3 * np.sin(t) * np.cos(t)
        ))
        return -t_min % np.pi

    def symmetry_measure(self, s, beta=1.0, f=None, theta=None):
        f = f if f is not None else self.f_s(s, beta)
        theta = theta or self.angle(s, beta=beta, f=f)
        f_rotated = f * np.exp(-1j * theta)
        t = np.imag(f_rotated)
        N = self.n
        return np.sum(t * t) ** 0.5 / N

    def __str__(self):
        return f'Fourier Descriptor: signal={self.signal}, f_0={self.f_0}'

    def __len__(self):
        return len(self.signal)
