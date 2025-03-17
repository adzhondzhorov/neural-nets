import numpy as np
from torch import Tensor, log


class Matrix(Tensor):
    def dims(self):
        return self.shape

    def ln(self):
        return log(self)

    def row(self, key):
        return Tensor(self[key])

    def rows(self, keys):
        return Matrix(self[[key for key in keys], :])
        
    def col(self, key):
        return Tensor(self[:, key])

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


class Tensor3D:
    def __init__(self, matrices):
        self.matrices = matrices

    def dims(self):
        return self.shape

    def __str__(self):
        return f"Tensor([{[str(m) for m in self.matrices]}])"

    def __repr__(self):
        return str(self)
