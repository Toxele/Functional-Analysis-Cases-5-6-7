import sys
import pathlib
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from shared.utils import pairwise_minkowski
from shared.validation import leave_one_out_classification
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


def select_prototypes_deletion(X, y):
    """Алгоритм последовательного удаления эталонов"""
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)

    current_idx = list(range(n))
    base_errors = 0
    min_size = len(np.unique(y))

    while len(current_idx) > min_size:
        best_removal_idx = -1
        best_errors = float('inf')

        for idx in current_idx:
            test_idx = [i for i in current_idx if i != idx]

            model = KNNClassifier(k=1).fit(X[test_idx], y[test_idx])

            errors = np.sum(model.predict(X) != y)

            if errors < best_errors:
                best_errors = errors
                best_removal_idx = idx

        if best_errors <= base_errors:
            current_idx.remove(best_removal_idx)
            base_errors = best_errors
        else:
            break

    return current_idx


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