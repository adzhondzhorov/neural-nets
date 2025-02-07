import numpy as np


class Op:
    ADD = "add"
    MUL = "multiply"
    NEG = "negate"
    POW = "power"
    EXP = "exponent"
    LN = "natural_logarithm"
    MAX = "maximum"
    MIN = "minimum"


class Derivative:
    EPSILON = 1e-10
    
    def __init__(self, op, in_vars, out_var):
        self.op = op
        self.in_vars = in_vars
        self.out_var = out_var
    
    def _other_var(self, wrt_var):
        return [var for var in self.in_vars if var != wrt_var][0]
    
    def __call__(self, wrt_var):
        part_deriv = None
        match self.op:
            case Op.ADD:
                part_deriv =  1
            case Op.MUL:
                other = self._other_var(wrt_var)
                part_deriv = other.data
            case Op.NEG:
                part_deriv = -1
            case Op.POW:
                power = self._other_var(wrt_var)
                part_deriv = power * wrt_var.data ** (power-1)
            case Op.EXP:
                part_deriv = np.exp(wrt_var.data)
            case Op.LN:
                part_deriv = 1 / (wrt_var.data + self.EPSILON)
            case Op.MAX:
                compare_num = self._other_var(wrt_var)
                part_deriv = 1 if wrt_var.data >= compare_num else 0
            case Op.MIN:
                compare_num = self._other_var(wrt_var)
                part_deriv = 1 if wrt_var.data <= compare_num else 0
    
        return self.out_var.grad * part_deriv
       