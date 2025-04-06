import numpy as np
from lib.np_backend.value import Value


class Vector(np.ndarray):
    def __new__(cls, values):
        are_value_types = isinstance(values[0], Value)
        if are_value_types:
            return np.asarray(values, dtype=object).view(cls)
        else:
            return np.asarray([Value(v) for v in values], dtype=object).view(cls)

    def dim(self):
        return len(self)

    def sum(self):
        return sum(self)

    def mean(self):
        return self.sum() / self.dim()

    def var(self):
        return sum([(v - self.mean()) ** 2 for v in self]) / self.dim()

    def std(self):
        return self.var() ** 0.5

    def dotprod(self, other):
        return sum(self * other)

    def all_values(self):
        return self


class Matrix(np.ndarray):
    def __new__(cls, values):
        are_value_types = isinstance(values[0][0], Value)
        if are_value_types:
            return np.asarray(values, dtype=object).view(cls)
        else:
            return np.asarray([[Value(v) for v in row] for row in values], dtype=object).view(cls)

    def dims(self):
        return self.shape

    def _apply(self, value_func, *args):
        self = np.vectorize(lambda x: value_func(x, *args), otypes=[object])(self)
        return self
    
    def max(self, num):
        return self._apply(Value.max, num)
        
    def min(self, num):
        return self._apply(Value.min, num)

    def exp(self):
        return self._apply(Value.exp)

    def ln(self):
        return self._apply(Value.ln)

    def all_values(self):
        return self.flatten().tolist()
    
    def row(self, key):
        return Vector(self[key])

    def rows(self, keys):
        return Matrix(self[[int(key.data) if isinstance(key, Value) else key for key in keys], :])
        
    def col(self, key):
        return Vector(self[:, key])

    def cols(self, key):
        return Matrix(self[:, [int(key.data) if isinstance(key, Value) else key for key in keys]])

    def row_sum(self):
        return Vector(self.sum(1))
     
    def col_sum(self):
        return Vector(self.sum(0))

    def matmul(self, other):
        return Matrix(self @ other)

    @staticmethod
    def broadcast(vector, n, axis=0):
        if axis == 0:
            return Matrix(np.tile(vector, (n, 1)).T)
        elif axis == 1:
            return Matrix(np.tile(vector, (n, 1)))


class Tensor3D:
    def __init__(self, matrices):
        self.matrices = matrices

    def dims(self):
        return self.shape

    def __str__(self):
        return f"Tensor([{[str(m) for m in self.matrices]}])"

    def __repr__(self):
        return str(self)
