from lib.np_backend.value import Value
from lib.np_backend.linear_algebra import Matrix, Vector


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
        return Matrix((matrix - self.means) / self.stds)


class LabelEncoder:
    def __init__(self):
        self.categories = []

    def fit(self, vector):
        for v in vector:
            if v not in self.categories:
                self.categories.append(v)

    def transform(self, vector):
        return Vector([self.categories.index(v) for v in vector])
