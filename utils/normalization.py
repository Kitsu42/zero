import numpy as np


def max_normalize(X, y):
    """
    Normaliza dados dividindo pelo maior valor
    """

    max_val = max(np.max(X), np.max(y))

    X_norm = X / max_val
    y_norm = y / max_val

    return X_norm, y_norm, max_val


def max_denormalize(value, max_val):
    """
    Retorna valor para escala original
    """

    return value * max_val