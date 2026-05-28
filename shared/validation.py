import numpy as np


def leave_one_out_regression(model_cls, model_kwargs, X, y):
    """Возвращает MSE по Leave-One-Out для регрессии (пересоздает модель)."""
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    errors = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model = model_cls(**model_kwargs)
        model.fit(X[mask], y[mask])
        pred = model.predict(X[i:i + 1])[0]
        errors[i] = (pred - y[i]) ** 2
    return np.mean(errors)


def leave_one_out_classification(model, X, y):
    """Возвращает долю ошибок (Error Rate) по Leave-One-Out для классификации."""
    X = np.asarray(X)
    y = np.asarray(y)
    errors = 0
    n = len(X)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        model.fit(X[mask], y[mask])
        pred = model.predict(X[i:i + 1])[0]

        if pred != y[i]:
            errors += 1

    return errors / n