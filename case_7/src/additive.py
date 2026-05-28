import numpy as np
from itertools import combinations_with_replacement


class AdditiveModel:
    def __init__(self, M=3, basis='poly', alpha=0.0):
        self.M = M
        self.basis = basis
        self.alpha = alpha
        self.beta_0 = 0.0
        self.theta = None
        self.p = 0
        self.feature_means = None

    def _get_basis(self, x_col):
        """Базисное разложение для одного признака (M функций)"""
        evals = []
        for m in range(1, self.M + 1):
            if self.basis == 'poly':
                evals.append((x_col ** m).reshape(-1, 1))
            elif self.basis == 'trig':
                if m % 2 == 1:
                    evals.append(np.sin(2 * np.pi * ((m + 1) // 2) * x_col).reshape(-1, 1))
                else:
                    evals.append(np.cos(2 * np.pi * (m // 2) * x_col).reshape(-1, 1))
        return np.hstack(evals)

    def _build_Z(self, X):
        """Блочная матрица признаков Z"""
        N, self.p = X.shape
        Z_blocks = [np.ones((N, 1))]
        for j in range(self.p):
            Z_blocks.append(self._get_basis(X[:, j]))
        return np.hstack(Z_blocks)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        Z = self._build_Z(X)

        # OLS или Ridge
        if self.alpha == 0:
            self.theta = np.linalg.pinv(Z.T @ Z) @ Z.T @ y
        else:
            I = np.eye(Z.shape[1])
            I[0, 0] = 0  # Не штрафуем свободный член
            self.theta = np.linalg.inv(Z.T @ Z + self.alpha * I) @ Z.T @ y

        self.beta_0 = self.theta[0]
        self.feature_means = np.zeros(self.p)

        # Центрирование компонент (Теорема Колмогорова / Идентификация)
        new_beta_0 = self.beta_0
        for j in range(self.p):
            start = 1 + j * self.M
            end = 1 + (j + 1) * self.M
            g_vals = Z[:, start:end] @ self.theta[start:end]
            mean_val = np.mean(g_vals)

            self.feature_means[j] = mean_val
            new_beta_0 += mean_val

        self.beta_0 = new_beta_0
        return self

    def predict(self, X):
        return self._build_Z(np.asarray(X)) @ self.theta

    def predict_components(self, X):
        """Возвращает центрированные компоненты g_j(x_j) для интерпретации"""
        Z = self._build_Z(np.asarray(X))
        components = []
        for j in range(self.p):
            start = 1 + j * self.M
            end = 1 + (j + 1) * self.M
            g_vals = Z[:, start:end] @ self.theta[start:end] - self.feature_means[j]
            components.append(g_vals)
        return self.beta_0, components


class FullNonLinearModel:
    """Полная полиномиальная регрессия с попарными взаимодействиями для сравнения"""

    def __init__(self, degree=2, alpha=0.0):
        self.degree = degree
        self.alpha = alpha

    def _build_Z(self, X):
        N, p = X.shape
        cols = [np.ones((N, 1))]
        for d in range(1, self.degree + 1):
            for combs in combinations_with_replacement(range(p), d):
                col = np.ones(N)
                for idx in combs:
                    col *= X[:, idx]
                cols.append(col.reshape(-1, 1))
        return np.hstack(cols)

    def fit(self, X, y):
        Z = self._build_Z(np.asarray(X))
        if self.alpha == 0:
            self.theta = np.linalg.pinv(Z.T @ Z) @ Z.T @ y
        else:
            I = np.eye(Z.shape[1])
            I[0, 0] = 0
            self.theta = np.linalg.inv(Z.T @ Z + self.alpha * I) @ Z.T @ y
        return self

    def predict(self, X):
        return self._build_Z(np.asarray(X)) @ self.theta