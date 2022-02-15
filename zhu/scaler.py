import numpy as np


class Scaler:
    def __init__(self, u=None):
        self.scale = None
        self.vec = None
        if u is not None:
            self.fit(u)

    def fit(self, u):
        self.vec = np.min(np.real(u)) + np.min(np.imag(u)) * 1j
        self.scale = np.max(np.abs(u - self.vec))

    def predict(self, u):
        u_ = np.array(u, dtype=complex)
        return (u_ - self.vec) / self.scale

    def inverse(self, v):
        v_ = np.array(v, dtype=complex)
        return v_ * self.scale + self.vec

    def __str__(self):
        return f'Scaler with scale={self.scale} and vec={self.vec}'
