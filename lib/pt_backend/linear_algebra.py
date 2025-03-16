import numpy as np
from torch import Tensor, log


class Vector(Tensor):
    def dim(self):
        return self.shape

    # def sum(self):
    #     return sum(self)

    # def mean(self):
    #     return self.sum() / self.dim()

    # def var(self):
    #     return sum([(v - self.mean()) ** 2 for v in self]) / self.dim()

    # def std(self):
    #     return self.var() ** 0.5

    # def dotprod(self, other):
    #     return sum(self * other)

    # def all_values(self):
    #     return self


class Matrix(Tensor):
    def dims(self):
        return self.shape

    # def _apply(self, value_func, *args):
    #     self = np.vectorize(lambda x: value_func(x, *args), otypes=[object])(self)
    #     return self
    
    # def max(self, num):
    #     return self._apply(Value.max, num)
        
    # def min(self, num):
    #     return self._apply(Value.min, num)

    # def exp(self):
    #     return self._apply(Value.exp)

    def ln(self):
        return log(self)

    # def all_values(self):
    #     return self.flatten().tolist()
    
    def row(self, key):
        return Tensor(self[key])

    def rows(self, keys):
        return Matrix(self[[key for key in keys], :])
        
    def col(self, key):
        return Tensor(self[:, key])

    # def cols(self, key):
    #     return Matrix(self[:, [int(key.data) if isinstance(key, Value) else key for key in keys]])

    def row_sum(self):
        return Tensor(self.sum(1))
     
    def col_sum(self):
        return Tensor(self.sum(0))

    def __round__(self, precision=0):
        if precision == 0:
            return self.round()
        return self.round() / 10 ** precision

    def __format__(self, format_spec):
        return format(self.item(), format_spec)
    # def matmul(self, other):
    #     return Matrix(self @ other)

    # @staticmethod
    # def broadcast(vector, n):
    #     return Matrix(np.tile(vector, (n, 1)).T)


class Tensor3D:
    def __init__(self, matrices):
        self.matrices = matrices

    def dims(self):
        return self.shape

    def __str__(self):
        return f"Tensor([{[str(m) for m in self.matrices]}])"

    def __repr__(self):
        return str(self)
