import random


class DataLoader:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def get_batch(self):
        pass


class BatchDataLoader(DataLoader):
    def get_batch(self):
        return self.X, self.y


class StochasticDataLoader(DataLoader):
    def __init__(self, X, y):
        super().__init__(X, y)
        self.current_step = 0

    def get_batch(self):
        indices = [self.current_step]
        if self.current_step < self.X.dims()[0] - 1:
            self.current_step += 1
        else:
            self.current_step = 0
        return self.X.rows(indices), self.y.rows(indices)


class MiniBatchDataLoader(DataLoader):
    def __init__(self, X, y, batch_size=16):
        super().__init__(X, y)
        self.current_step = 0
        self.batch_size = batch_size
        self.indexes = self._regenerate_indexes()

    def _regenerate_indexes(self):
        rows_len = self.X.dims()[0]
        return random.sample(range(rows_len), rows_len)

    def get_batch(self):
        next_step = self.current_step + self.batch_size
        if next_step >= self.X.dims()[0]:
            rest_len = next_step - self.X.dims()[0]
            batch_indexes = self.indexes[self.current_step:]
            self.indexes = self._regenerate_indexes()
            batch_indexes.extend(self.indexes[:rest_len])
            self.current_step = rest_len
        else:
            batch_indexes = self.indexes[self.current_step:next_step]
            self.current_step = next_step
            
        return self.X.rows(batch_indexes), self.y.rows(batch_indexes)
