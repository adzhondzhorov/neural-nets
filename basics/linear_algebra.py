from basics.print_utils import pprint


class Scalar:
    def __init__(self, value):
        self.value = value
        
    def __str__(self):
        return f"Sclar({self.value})"
        
    def __repr__(self):
        return str(self)
        
    def dim(self):
        return 0

    def __add__(self, other):
        assert isinstance(other, Scalar)
        return Scalar(self.value + other.value)
    
    def __mul__(self, other):
        assert isinstance(other, Scalar)
        return Scalar(self.value * other.value)


class Vector:
    def __init__(self, values):
        self.values = values
        
    def __str__(self):
        return f"Vector({pprint(self.values)})"
        
    def __repr__(self):
        return str(self)
    
    def __add__(self, other):  
        assert isinstance(other, Vector)
        assert self.dim() == other.dim()
        return Vector([v1 + v2 for v1, v2 in zip(self.values, other.values)])

    def dim(self):
        return len(self.values)

    def dotprod(self, other):  
        assert isinstance(other, Vector)
        assert self.dim() == other.dim()
        return Scalar(sum([v1 * v2 for v1, v2 in zip(self.values, other.values)]))
        

class Matrix:
    def __init__(self, values):
        self.values = values

    def __str__(self):
        return f"Matrix({pprint(self.values)})"

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        assert isinstance(other, Matrix)
        assert self.dims() == other.dims()
        rows = []
        for r1 in range(self.dims()[0]):
            row = []
            for c1 in range(self.dims()[1]):
                row.append(self.values[r1][c1] + other.values[r1][c1])
            rows.append(row)
        return Matrix(rows)
    
    def __mul__(self, other):
        assert isinstance(other, Matrix)
        assert self.dims() == other.dims()
        rows = []
        for r1 in range(self.dims()[0]):
            row = []
            for c1 in range(self.dims()[1]):
                row.append(self.values[r1][c1] * other.values[r1][c1])
            rows.append(row)
        return Matrix(rows)
        
    def dims(self):
        return (len(self.values), len(self.values[0]))

    def t(self):
        new_rows = []
        for c in range(self.dims()[1]):
            new_row = []
            for r in range(self.dims()[0]):
                new_row.append(self.values[r][c])
            new_rows.append(new_row) 
        return Matrix(new_rows)

    def matmul(self, other):
        assert isinstance(other, Matrix)
        assert self.dims()[1] == other.dims()[0]
        rows = []
        for r1 in range(self.dims()[0]):
            row = []
            for c2 in range(other.dims()[1]):
                val = 0
                for i in range(self.dims()[1]):
                    val += self.values[r1][i] * other.values[i][c2]
                row.append(val)
            rows.append(row)
        return Matrix(rows)

    
class Tensor:
    def __init__(self, values):
        self.values = values

    def __str__(self):
        return f"Tensor({pprint(self.values)})"

    def __repr__(self):
        return str(self)

    def __mul__(self, other):
        def _inner_mul(t1, t2):
            if not isinstance(t1, list):
                return t1*t2
            else:
                return [_inner_mul(t1[i],t2[i]) for i in range(len(t1))]
        assert isinstance(other, Tensor)
        assert self.dims() == other.dims()
        return Tensor(_inner_mul(self.values, other.values))
            
    def __add__(self, other):
        def _inner_add(t1, t2):
            if not isinstance(t1, list):
                return t1 + t2
            else:
                return [_inner_add(t1[i], t2[i]) for i in range(len(t1))]
        assert isinstance(other, Tensor)
        assert self.dims() == other.dims()
        return Tensor(_inner_add(self.values, other.values))

    def dims(self):
        l = self.values
        dims = []
        while isinstance(l, list):
            dims.append(len(l))
            l = l[0]
        return tuple(dims)
 
    def matmul(self, other):
        def inner_matmul(t1, t2, d=0):
            if d == len(self.dims()) - 2:
                return Matrix(t1).matmul(Matrix(t2)).values
            else:
                return [inner_matmul(t1[i], t2[i], d + 1) for i in range(len(t1))]
        assert isinstance(other, Tensor)
        assert self.dims()[:-2] == other.dims()[:-2]
        assert self.dims()[-1] == other.dims()[-2]
        return Tensor(inner_matmul(self.values, other.values))
