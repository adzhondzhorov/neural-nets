from numbers import Number
from uuid import uuid4

import numpy as np

from lib.calculus import Derivative, Op


class Value:
    __slots__ = ("data", "grad", "children", "_backward", "_id")

    def __init__(self, data):
        self.data = data
        self._id = uuid4()

    def __getattr__(self, name):
        lazy_loads = {
            "grad": 0,
            "children": set(),
            "_backward": lambda: 0,
        }
        setattr(self, name, lazy_loads[name])
        return lazy_loads[name]

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
    
    def min(self, num):
        if isinstance(num, Number):
            out = Value(self.data if self.data <= num else num)
            out.children.update([self])
            out._backward = Value._get_backward_func(Op.MIN, [self, num], out)
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

    @staticmethod
    def _order_topologically(items):
        descendants_map = {}
        for item in items:
            item_descendats = set()
            item._get_all_descendats(item_descendats)
            item_descendats.remove(item)
            descendants_map[item] = item_descendats
        reverse_order = []
        while items:
            new_to_order = set()
            for item in items:
                if not descendants_map[item]:
                    new_to_order.add(item)
            items = [i for i in items if i not in new_to_order]
            for item in descendants_map:
                descendants_map[item] -= new_to_order
            reverse_order.extend(new_to_order)
        
        return reversed(reverse_order)
    
    def _get_all_descendats(self, all_descendats):
        all_descendats.add(self) 
        for child in self.children:
            child._get_all_descendats(all_descendats)

    def _get_topologically_ordered_all_descendats(self):
        all_decsendants = set()
        self._get_all_descendats(all_decsendants)
        return self._order_topologically(list(all_decsendants))
       
    def backward(self):
        all_decsendants = self._get_topologically_ordered_all_descendats()
        for decsendant in all_decsendants:
            decsendant._backward()
