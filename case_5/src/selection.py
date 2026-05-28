import sys
import pathlib
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from shared.utils import pairwise_minkowski
from case_5.src.classifiers import KNNClassifier


def compactness_profile(X, y):
    """Вычисляет профиль компактности П(m)"""
    X = np.asarray(X)
    y = np.asarray(y)
    L = len(y)
    profile = []

    dists = pairwise_minkowski(X, X)
    np.fill_diagonal(dists, np.inf)
    sorted_neighbors_idx = np.argsort(dists, axis=1)

    for m in range(L - 1):
        m_th_neighbors_classes = y[sorted_neighbors_idx[:, m]]
        errors = np.sum(y != m_th_neighbors_classes)
        profile.append(errors / L)

    return profile


def select_prototypes_addition(X, y):
    """Алгоритм последовательного добавления эталонов"""
    X = np.asarray(X)
    y = np.asarray(y)
    classes, indices = np.unique(y, return_index=True)
    current_idx = list(indices)

    while True:
        model = KNNClassifier(k=1).fit(X[current_idx], y[current_idx])
        preds = model.predict(X)
        errors = (preds != y)
        if not np.any(errors): break
        error_indices = np.where(errors)[0]
        current_idx.append(error_indices[0])

    return current_idx