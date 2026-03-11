import numpy as np

from data.generate_sequence import generate_fibonacci_dataset
from model.neural_network import NeuralNetwork


def train_model():

    # gerar dataset
    X, y, scale, sequence = generate_fibonacci_dataset(n=20)

    print("Dataset gerado:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # criar rede neural
    nn = NeuralNetwork(
        input_size=2,
        hidden_size=8,
        output_size=1
    )

    # treinar rede
    nn.train(
        X,
        y,
        epochs=5000,
        lr=0.1
    )

    return nn, scale, sequence


if __name__ == "__main__":

    model, scale, sequence = train_model()

    print("\nTreinamento finalizado.")