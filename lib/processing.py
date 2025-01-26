from lib.value import Value
from lib.linear_algebra import Matrix


class OneHotEncoder:
    def __init__(self):
        self.categories = []

    def fit(self, vector):
        for v in vector:
            if v not in self.categories:
                self.categories.append(v)

    def transform(self, vector):
        result = []
        for v in vector:
            result.append([int(v == c) for c in self.categories])

        return Matrix(result)


class ColumnNormalizer:
    def __init__(self):
        self.means = []
        self.stds = []

    def fit(self, matrix):     
        for col in range(matrix.dims()[1]):
            self.means.append(matrix.col(col).mean())
            self.stds.append(matrix.col(col).std())
            
    def transform(self, matrix):
        result = []
        for col in range(matrix.dims()[1]):
            for idx, v in enumerate(matrix.col(col)):
                if len(result) <= idx:
                    result.append([])
                result[idx].append(Value((v.data - self.means[col].data) / self.stds[col].data))
        return Matrix(result)