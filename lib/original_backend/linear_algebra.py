from numbers import Number

from lib.original_backend.value import Value


class Vector:
    def __init__(self, values):
        are_value_types = isinstance(values[0], Value)
        if are_value_types:
            self.values = values.copy()
        else:
            self.values = [Value(v) for v in values]

    def __str__(self):
        return f"Vector({self.values})"

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        return self.values.__getitem__(key)
        
    def __iter__(self):
        return self.values.__iter__()

    def __next__(self):
        return self.values.__next__()

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector([sv + ov for sv, ov in zip(self.values, other.values)])
        elif isinstance(other, Value):
            return Vector([sv + other for sv in self.values])

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector([sv * ov for sv, ov in zip(self.values, other.values)])
        elif isinstance(other, Value):
            return Vector([sv * other for sv in self.values])

    def __truediv__(self, other):
        return self.__mul__(other**(-1))

    def dim(self):
        return len(self.values)

    def sum(self):
        return sum(self.values)

    def mean(self):
        return (sum(self.values)/self.dim())

    def __neg__(self):
        return Vector([-v for v in self.values])

    def var(self):
        mean = self.mean()
        return sum([(v - mean)**2 for v in self.values]) / self.dim()

    def std(self):
        return self.var() ** (1/2)
        
    def dotprod(self, other):
        return sum(self * other)

    def all_values(self):
        return self.values


class Matrix:
    def __init__(self, values):
        are_value_types = isinstance(values[0][0], Value)
        if are_value_types:
            self.values = values.copy()
        else:
            result = []
            for row in values:
                result.append([Value(v) for v in row])
            self.values = result

    def __str__(self):
        values_str = ",".join(["\n" + str(r) for r in self.values])
        return f"Matrix([{values_str}\n])"

    def dims(self):
        return (len(self.values), len(self.values[0]))
    
    def __repr__(self):
        return str(self)

    def __iter__(self):
        return self.values.__iter__()

    def __next__(self):
        return self.values.__next__()
  
    def __radd__(self, other):
        if isinstance(other, Number):
            return self.__add__(Value(other))
        else:
            return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, Matrix):
            result = []
            for rs, ro in zip(self.values, other.values):
                result.append([vr + vo for vr, vo in zip(rs, ro)])
            return Matrix(result)
        elif isinstance(other, Vector):
            result = []
            for row in self.values:
                result.append([vr + vv for vr, vv in zip(row, other.values)]) 
            return Matrix(result)
        elif isinstance(other, Value):
            result = []
            for row in self.values:
                result.append([vr + other for vr in row]) 
            return Matrix(result)
        elif isinstance(other, Number):
            return self.__add__(Value(other))

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        if isinstance(other, Number):
            d1, d2 = self.dims()
            return Matrix([[other] * d2] * d1).__add__(-self)

    def __neg__(self):
        result = [] 
        for row in self.values:
            result.append([-v for v in row])
        return Matrix(result)

    def __pow__(self, power):
        result = [] 
        for row in self.values:
            result.append([v**power for v in row])
        return Matrix(result)

    def __mul__(self, other):
        result = []
        for rs, ro in zip(self.values, other.values):
            result.append([vr * vo for vr, vo in zip(rs, ro)])
        return Matrix(result)
    
    def __truediv__(self, other):
        return self.__mul__(other**(-1))

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return Matrix([[other]*self.dims()[1]]*self.dims()[0]).__truediv__(self)

    def max(self, num):
        for row in self.values:
            for i in range(len(row)):
                row[i] = Value.max(row[i], num)
        return self

    def min(self, num):
        for row in self.values:
            for i in range(len(row)):
                row[i] = Value.min(row[i], num)
        return self

    def exp(self):
        result = []
        for row in self.values:
            result_row = []
            for v in row:
                result_row.append(Value.exp(v))
            result.append(result_row)
        return Matrix(result)
    
    def ln(self):
        result = []
        for row in self.values:
            result_row = []
            for v in row:
                result_row.append(Value.ln(v))
            result.append(result_row)
        return Matrix(result)
    
    def all_values(self):
        all_values = []
        for row in self.values:
            for v in row:
                all_values.append(v)
        return all_values
    
    def row(self, key):
        return Vector(self.values[int(key.data) if isinstance(key, Value) else key])

    def rows(self, keys):
        return Matrix([self.values[int(key.data) if isinstance(key, Value) else key] for key in keys])
        
    def col(self, key):
        col = []
        for r in self.values:
            col.append(r[key])
        return Vector(col)

    def row_sum(self):
        row_sum = []
        for row in self.values:
            row_sum.append(sum(row))
        return Vector(row_sum)
     
    def col_sum(self):
        col_sum = [0] * self.dims()[1]
        for i in range(self.dims()[1]):
            for row in self.values:
                col_sum[i]+=row[i]
        return Vector(col_sum)

    @staticmethod
    def broadcast(vector, n, axis=0):
        result = []
        if axis == 0:
            for v in vector:
                result.append([v] * n)
        elif axis == 1:
            result = [vector.values] * n
        return Matrix(result)

    def matmul(self, other):
        assert self.dims()[1] == other.dims()[0], f"Trying to multiply matrices of dims {self.dims()} and {other.dims()}"
        result = []
        for sri in range(len(self.values)):
            row = []
            for oci in range(len(other.values[0])):
                self_row = self.values[sri]
                other_col = [r[oci] for r in other.values]
                row.append(sum([vr * vo for vr, vo in zip(self_row, other_col)]))
            result.append(row)
        return Matrix(result)


class Tensor3D:
    def __init__(self, matrices):
        self.matrices = matrices

    def __str__(self):
        return f"Tensor([{[str(m) for m in self.matrices]}])"

    def __repr__(self):
        return str(self)

    def dims(self):
        return tuple([len(self.matrices), *self.matrices[0].dims()])
