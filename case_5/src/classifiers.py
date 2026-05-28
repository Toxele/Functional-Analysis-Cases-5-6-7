import sys
import pathlib
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from shared.utils import pairwise_minkowski
from shared.kernels import KERNELS


class BaseMetricClassifier:
    def __init__(self, p=2.0, feature_weights=None):
        self.p = p
        self.feature_weights = feature_weights

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(y)
        return self

    def _get_weights(self, sorted_distances):
        raise NotImplementedError

    def predict(self, X):
        X = np.asarray(X)
        dists = pairwise_minkowski(X, self.X_train, self.p, self.feature_weights)

        predictions = []
        for row_dist in dists:
            sorted_idx = np.argsort(row_dist)
            sorted_distances = row_dist[sorted_idx]

            weights = self._get_weights(sorted_distances)

            class_scores = {c: 0.0 for c in self.classes_}
            for idx, w in zip(sorted_idx, weights):
                if w > 0:
                    class_scores[self.y_train[idx]] += w

            predictions.append(max(class_scores, key=class_scores.get))

        return np.array(predictions)


class KNNClassifier(BaseMetricClassifier):
    def __init__(self, k=1, p=2.0, feature_weights=None):
        super().__init__(p, feature_weights)
        self.k = k

    def _get_weights(self, sorted_distances):
        weights = np.zeros_like(sorted_distances)
        weights[:self.k] = 1.0
        return weights


class WeightedKNNClassifier(BaseMetricClassifier):
    def __init__(self, k=5, p=2.0, feature_weights=None):
        super().__init__(p, feature_weights)
        self.k = k

    def _get_weights(self, sorted_distances):
        weights = np.zeros_like(sorted_distances)
        w_vals = np.arange(self.k, 0, -1)
        k_actual = min(self.k, len(sorted_distances))
        weights[:k_actual] = w_vals[:k_actual]
        return weights


class ParzenFixedClassifier(BaseMetricClassifier):
    def __init__(self, h=1.0, kernel='gaussian', p=2.0, feature_weights=None):
        super().__init__(p, feature_weights)
        self.h = h
        self.kernel = KERNELS[kernel]

    def _get_weights(self, sorted_distances):
        if self.h == 0: return np.zeros_like(sorted_distances)
        u = sorted_distances / self.h
        return self.kernel(u)


class ParzenVariableClassifier(BaseMetricClassifier):
    def __init__(self, k=5, kernel='gaussian', p=2.0, feature_weights=None):
        super().__init__(p, feature_weights)
        self.k = k
        self.kernel = KERNELS[kernel]

    def _get_weights(self, sorted_distances):
        k_actual = min(self.k, len(sorted_distances) - 1)
        h = sorted_distances[k_actual]
        if h == 0:
            weights = np.zeros_like(sorted_distances)
            weights[sorted_distances == 0] = 1.0
            return weights
        u = sorted_distances / h
        return self.kernel(u)