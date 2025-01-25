import numpy as np

from lib.linear_algebra import Vector, Matrix


class Layer:
    def __call__(self, X):
        return self.forward(X)

    def params(self):
        return []


class Linear(Layer):
    def __init__(self, fan_in, fan_out):
        self.W = Matrix(np.random.normal(size=(fan_in, fan_out)))
        self.b = Vector(np.zeros(fan_out))

    def forward(self, X):
        out = Matrix.matmul(X, self.W) + self.b
        return out

    def params(self):
        return [self.W, self.b]


class ReLU(Layer):
    def forward(self, X):
        return Matrix.max(X, 0)


class Sigmoid(Layer):
    def forward(self, X):
        return 1 / (1 + (-X).exp())


class NN:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        for l in self.layers:
            X = l(X)
        return X

    def params(self):
        params = []
        for l in self.layers:
            params.extend(l.params())
        return params
