EPSILON = 1e-8

class SgdOptimizer:
    def __init__(self, nn, learning_rate):
        self.all_values = [v for p in nn.params() for v in p.all_values()]
        self.learning_rate = learning_rate

    def step(self, loss):
        for v in self.all_values:
            v.zero_grad()

        loss.grad = 1
        loss.backward()
        self._update_grads()

    def _update_grads(self):
        for v in self.all_values:
            v.data -= self.learning_rate * v.grad


class SgdWithMomentumOptimizer(SgdOptimizer):
    def __init__(self, nn, learning_rate, momentum_coef):
        super().__init__(nn, learning_rate)
        self.momentum_coef = momentum_coef
        self.grad_accums = [0 for _ in range(len(self.all_values))]

    def _update_grads(self):
        for idx, v in enumerate(self.all_values):
            self.grad_accums[idx] = self.momentum_coef * self.grad_accums[idx] + (1 - self.momentum_coef) * v.grad
            v.data -= self.learning_rate * self.grad_accums[idx]


class AdaGradOptimizer(SgdOptimizer):
    def __init__(self, nn, learning_rate):
        super().__init__(nn, learning_rate)
        self.grad_accums = [0 for _ in range(len(self.all_values))]

    def _update_grads(self):
        for idx, v in enumerate(self.all_values):
            self.grad_accums[idx] +=  v.grad ** 2
            v.data -= self.learning_rate * v.grad / (self.grad_accums[idx] + EPSILON) ** 0.5


class RmsPropOptimizer(SgdWithMomentumOptimizer):
    def _update_grads(self):
        for idx, v in enumerate(self.all_values):
            self.grad_accums[idx] = self.momentum_coef * self.grad_accums[idx] + (1 - self.momentum_coef) * v.grad ** 2
            v.data -= self.learning_rate * v.grad / (self.grad_accums[idx] + EPSILON) ** 0.5


class AdamOptimizer(SgdOptimizer):    
    def __init__(self, nn, learning_rate, momentum_coef1, momentum_coef2):
        super().__init__(nn, learning_rate)
        self.grad_accums1 = [0 for _ in range(len(self.all_values))]
        self.grad_accums2 = [0 for _ in range(len(self.all_values))]
        self.momentum_coef1 = momentum_coef1
        self.momentum_coef2 = momentum_coef2
        self.time = 1

    def _update_grads(self):
        for idx, v in enumerate(self.all_values):
            self.grad_accums1[idx] = self.momentum_coef1 * self.grad_accums1[idx] + (1 - self.momentum_coef1) * v.grad
            self.grad_accums2[idx] = self.momentum_coef2 * self.grad_accums2[idx] + (1 - self.momentum_coef2) * v.grad**2
            m_norm = self.grad_accums1[idx] / (1 - self.momentum_coef1 ** self.time)
            v_norm = self.grad_accums2[idx] / (1 - self.momentum_coef2 ** self.time)
            v.data -= self.learning_rate * m_norm / (v_norm ** 0.5 + EPSILON)
        self.time += 1
