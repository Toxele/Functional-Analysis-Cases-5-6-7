import numpy as np
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from shared.kernels import KERNELS
from shared.utils import pairwise_distances


class NadarayaWatson:
    def __init__(self, kernel='gaussian', h=None, k=None):
        self.kernel_name = kernel
        self.kernel_fn = KERNELS[kernel]
        self.h = h
        self.k = k
        self.X_train = None
        self.y_train = None
        self.sample_weights = None

    def fit(self, X, y, sample_weights=None):
        self.X_train = np.asarray(X)
        if self.X_train.ndim == 1: self.X_train = self.X_train.reshape(-1, 1)
        self.y_train = np.asarray(y)
        self.sample_weights = np.ones(len(y)) if sample_weights is None else np.asarray(sample_weights)
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1: X = X.reshape(-1, 1)

        dists = pairwise_distances(X, self.X_train)
        preds = np.zeros(len(X))

        for i, d in enumerate(dists):
            if self.h is not None:
                h_val = self.h
            elif self.k is not None:
                k_actual = min(self.k, len(d) - 1)
                h_val = np.sort(d)[k_actual]
            else:
                raise ValueError("Укажите h (фикс. окно) или k (перем. окно)")

            if h_val == 0:
                preds[i] = np.mean(self.y_train[d == 0]) if np.any(d == 0) else 0.0
                continue

            u = d / h_val
            weights = self.kernel_fn(u) * self.sample_weights

            sum_w = np.sum(weights)
            preds[i] = np.sum(weights * self.y_train) / sum_w if sum_w > 0 else np.mean(self.y_train)

        return preds


class LOWESS:
    def __init__(self, kernel='gaussian', h=None, k=None, n_iter=3):
        self.kernel = kernel
        self.h = h
        self.k = k
        self.n_iter = n_iter
        self.nw_model = None
        self.robust_weights = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(X)
        self.robust_weights = np.ones(n)
        quartic = KERNELS['quartic']

        for _ in range(self.n_iter):
            a_i = np.zeros(n)
            # LOO оценки для робастных весов
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                nw = NadarayaWatson(kernel=self.kernel, h=self.h, k=self.k)
                nw.fit(X[mask], y[mask], sample_weights=self.robust_weights[mask])
                a_i[i] = nw.predict(X[i:i + 1])[0]

            eps = np.abs(a_i - y)
            median_eps = np.median(eps)
            if median_eps == 0: break

            m = 6.0 * median_eps
            self.robust_weights = quartic(eps / m)

        self.nw_model = NadarayaWatson(kernel=self.kernel, h=self.h, k=self.k)
        self.nw_model.fit(X, y, sample_weights=self.robust_weights)
        return self

    def predict(self, X):
        return self.nw_model.predict(X)