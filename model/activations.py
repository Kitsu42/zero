import numpy as np


def sigmoid(x):
    """
    Função sigmoid
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivada da sigmoid
    x deve ser a saída da sigmoid
    """
    return x * (1 - x)


def relu(x):
    """
    ReLU activation
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivada da ReLU
    """
    return (x > 0).astype(float)


def linear(x):
    """
    Função linear (usada na saída de regressão)
    """
    return x


def linear_derivative(x):
    """
    Derivada da função linear
    """
    return np.ones_like(x)