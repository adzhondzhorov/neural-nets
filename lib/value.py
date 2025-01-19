from numbers import Number
from uuid import uuid4

import numpy as np

from lib.calculus import Derivative, Op


class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0
        self.children = set()
        self._backward = lambda: 0
        self._id = uuid4()

    def __str__(self):
        return str(f"{{{str(self._id)[:8]}, {round(self.data, 2)}, {round(self.grad, 2)}}}")

    def __repr__(self):
        return str(self)

    def __mul__(self, other):
        out = Value(self.data * other.data)
        out.children.update([self, other])
        out._backward = Value._get_backward_func(Op.MUL, [self, other], out)
        return out

    def __truediv__(self, other):
        if isinstance(other, Number):
            return self.__truediv__(Value(other))
        return self.__mul__(other**(-1))

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return Value(other).__truediv__(self)

    def __add__(self, other):
        out = Value(self.data + other.data)
        out.children.update([self, other])
        out._backward = Value._get_backward_func(Op.ADD, [self, other], out)
        return out
    
    def __radd__(self, other):
        if isinstance(other, Number):
            return self.__add__(Value(other))
        else:
            return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)
          
    def __neg__(self):
        out = Value(-self.data)
        out.children.update([self])
        out._backward = Value._get_backward_func(Op.NEG, [self], out)
        return out

    def __pow__(self, power):
        out = Value(self.data ** power)
        out.children.update([self])
        out._backward = Value._get_backward_func(Op.POW, [self, power], out)
        return out
  
    def exp(self):
        out = Value(np.exp(self.data))
        out.children.update([self])
        out._backward = Value._get_backward_func(Op.EXP, [self], out)
        return out

    def ln(self):
        out = Value(np.log(self.data))
        out.children.update([self])
        out._backward = Value._get_backward_func(Op.LN, [self], out)
        return out
    
    def max(self, num):
        if isinstance(num, Number):
            out = Value(self.data if self.data >= num else num)
            out.children.update([self])
            out._backward = Value._get_backward_func(Op.MAX, [self, num], out)
        return out
    
    def __lt__(self, num):
        if isinstance(num, Number):
            return self.data < num
            
    def __gt__(self, num):
        if isinstance(num, Number):
            return self.data > num

    def __eq__(self, other):
        return isinstance(other, Value) and self._id == other._id

    def __hash__(self):
        return hash(self._id)

    def zero_grad(self):
        self.grad = 0

    @staticmethod
    def _get_backward_func(op, in_vars, out_var):
        derivative = Derivative(op, in_vars, out_var)
        def _backward():
            for in_var in in_vars:
                if isinstance(in_var, Value):
                    in_var.grad += derivative(in_var)
        return _backward

    def backward(self):
        self._backward()
        for child in self.children:
            child.backward()
