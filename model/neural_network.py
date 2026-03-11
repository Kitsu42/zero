import numpy as np

from model.layers import DenseLayer
from model.activations import sigmoid, sigmoid_derivative
from model.activations import linear, linear_derivative


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):

        # criar camadas
        self.layer1 = DenseLayer(
            input_size,
            hidden_size,
            sigmoid,
            sigmoid_derivative
        )

        self.layer2 = DenseLayer(
            hidden_size,
            output_size,
            linear,
            linear_derivative
        )

    def forward(self, X):
        """
        Propagação para frente
        """

        out = self.layer1.forward(X)
        out = self.layer2.forward(out)

        return out

    def backward(self, y_true, y_pred, learning_rate):
        """
        Backpropagation
        """

        error = y_pred - y_true

        grad = self.layer2.backward(error, learning_rate)
        grad = self.layer1.backward(grad, learning_rate)

    def train(self, X, y, epochs=1000, lr=0.01):
        """
        Treinamento da rede
        """

        for epoch in range(epochs):

            y_pred = self.forward(X)

            loss = np.mean((y - y_pred) ** 2)

            self.backward(y, y_pred, lr)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {loss}")

    def predict(self, X):
        """
        Fazer previsão
        """

        return self.forward(X)