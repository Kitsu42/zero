import numpy as np


def fibonacci_sequence(n):
    """
    Gera uma sequência de Fibonacci com n elementos
    """
    seq = [0, 1]

    for i in range(2, n):
        seq.append(seq[i-1] + seq[i-2])

    return seq


def create_training_data(sequence, window_size=2):
    """
    Converte a sequência em dados de treino

    Exemplo:
    [0,1,1,2,3]

    X = [0,1]
    y = 1

    X = [1,1]
    y = 2
    """

    X = []
    y = []

    for i in range(len(sequence) - window_size):
        X.append(sequence[i:i+window_size])
        y.append(sequence[i+window_size])

    return np.array(X, dtype=float), np.array(y, dtype=float).reshape(-1,1)


def normalize_data(X, y):
    """
    Normaliza os dados para facilitar o treinamento
    """
    max_val = max(np.max(X), np.max(y))

    X = X / max_val
    y = y / max_val

    return X, y, max_val


def generate_fibonacci_dataset(n=20, window_size=2):
    """
    Pipeline completo de geração de dataset
    """

    sequence = fibonacci_sequence(n)

    X, y = create_training_data(sequence, window_size)

    X, y, scale = normalize_data(X, y)

    return X, y, scale, sequence