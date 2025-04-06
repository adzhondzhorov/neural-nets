import numpy as np

from lib.np_backend.linear_algebra import Vector, Matrix, Tensor3D

EPSILON = 1e-5


class Layer:
    def __call__(self, X):
        return self.forward(X)

    def params(self):
        return []


class Linear(Layer):
    def __init__(self, fan_in, fan_out, initialization="standard"):
        inner_w = np.random.normal(size=(fan_in, fan_out))
        match initialization:
            case "normal_glorot":
                inner_w = np.random.normal(size=(fan_in, fan_out), scale=np.sqrt(2/(fan_in + fan_out)))
            case "uniform_glorot":
                limit = np.sqrt(6/(fan_in + fan_out))
                inner_w = np.random.uniform(size=(fan_in, fan_out), low=-limit, high=limit)
        self.W = Matrix(inner_w)
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


class Softmax(Layer):
    def forward(self, X):
        exp_X = X.exp()
        return exp_X / (Matrix.broadcast(exp_X.row_sum(), X.dims()[1]) + EPSILON)


class Embedding(Layer):
    def __init__(self, vocab_size, emb_dim):
        self.embedding = Matrix(np.random.normal(size=(vocab_size, emb_dim)))

    def forward(self, X):
        out = Tensor3D([self.embedding.rows(seq) for seq in X])
        return out


class Flatten(Layer):
    def forward(self, X):
        out = Matrix([[channel for channels in sequence for channel in channels] for sequence in X.matrices])
        return out


class MaxPooling(Layer):
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def _maxpool(self, channels):
        pooled_channels = []
        step = 0
        while True:
            pooled_channels.append(max(channels[step:step + self.size]))
            step += self.stride
            if step >= len(channels):
                break
        return pooled_channels

    def forward(self, X):
        out = Tensor3D([[self._maxpool(channels) for channels in sequence] for sequence in X.matrices])
        return out


class LayerNorm(Layer):
    def __init__(self, fan_in):
        self.scale = Vector([1 for _ in range(fan_in)])
        self.bias = Vector([0 for _ in range(fan_in)])

    def forward(self, X):
        norm_rows = []
        for r in range(X.dims()[0]):
            row = X.row(r)
            norm_row = ((row - row.mean()) / row.std())
            norm_rows.append(norm_row)
        norm_X = Matrix(norm_rows)
        out = norm_X * Matrix.broadcast(self.scale, norm_X.dims()[0], axis=1) + Matrix.broadcast(self.bias, norm_X.dims()[0],  axis=1)
        return out

    def params(self):
        return [self.scale, self.bias]


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
