import numpy as np


class DenseLayer:

    def __init__(self, input_size, output_size, activation, activation_derivative):
        """
        Camada densa (fully connected)
        """

        self.W = np.random.randn(input_size, output_size) * 0.1
        self.b = np.zeros((1, output_size))

        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, X):
        """
        Forward propagation
        """

        self.input = X

        self.z = np.dot(X, self.W) + self.b

        self.output = self.activation(self.z)

        return self.output

    def backward(self, grad_output, learning_rate):
        """
        Backpropagation
        """

        activation_grad = grad_output * self.activation_derivative(self.output)

        dW = np.dot(self.input.T, activation_grad)
        db = np.sum(activation_grad, axis=0, keepdims=True)

        grad_input = np.dot(activation_grad, self.W.T)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return grad_input