import numpy as np


def pairwise_distances(X1, X2):
    """Евклидовы расстояния между парами объектов."""
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    if X1.ndim == 1: X1 = X1.reshape(-1, 1)
    if X2.ndim == 1: X2 = X2.reshape(-1, 1)
    return np.linalg.norm(X1[:, np.newaxis, :] - X2[np.newaxis, :, :], axis=2)


def pairwise_minkowski(X1, X2, p=2.0, w=None):
    """Попарные расстояния с использованием взвешенной метрики Минковского."""
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    if w is None:
        w = np.ones(X1.shape[1])
    else:
        w = np.asarray(w)

    diff = np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :])
    return np.sum(w * (diff ** p), axis=2) ** (1.0 / p)


def min_max_scale(X, feature_range=(0, 1)):
    """Нормирование признаков в заданный интервал"""
    X = np.asarray(X)
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    scale = max_val - min_val
    scale[scale == 0] = 1.0
    X_scaled = (X - min_val) / scale
    return X_scaled * (feature_range[1] - feature_range[0]) + feature_range[0]