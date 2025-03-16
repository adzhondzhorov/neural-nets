import torch

EPSILON = 1e-8

class SgdOptimizer:
    def __init__(self, nn, learning_rate):
        self.params = list(nn.params())
        self.learning_rate = learning_rate

    def step(self, loss):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        self._update_grads()

    def _update_grads(self):
        for p in self.params:
            p.data -= self.learning_rate * p.grad

class SgdWithMomentumOptimizer(SgdOptimizer):
    def __init__(self, nn, learning_rate, momentum_coef):
        super().__init__(nn, learning_rate)
        self.momentum_coef = momentum_coef
        self.grad_accums = [torch.zeros(p.shape) for p in self.params]

    def _update_grads(self):
        for idx, p in enumerate(self.params):
            self.grad_accums[idx] = self.momentum_coef * self.grad_accums[idx] + (1 - self.momentum_coef) * p.grad
            p.data -= self.learning_rate * self.grad_accums[idx]


class AdaGradOptimizer(SgdOptimizer):
    def __init__(self, nn, learning_rate):
        super().__init__(nn, learning_rate)
        self.grad_accums = [torch.zeros(p.shape) for p in self.params]

    def _update_grads(self):
        for idx, p in enumerate(self.params):
            self.grad_accums[idx] += p.grad ** 2
            p.data -= self.learning_rate * p.grad / (self.grad_accums[idx] + EPSILON) ** 0.5


class RmsPropOptimizer(SgdWithMomentumOptimizer):
    def _update_grads(self):
        for idx, p in enumerate(self.params):
            self.grad_accums[idx] = self.momentum_coef * self.grad_accums[idx] + (1 - self.momentum_coef) * p.grad ** 2
            p.data -= self.learning_rate * p.grad / (self.grad_accums[idx] + EPSILON) ** 0.5


class AdamOptimizer(SgdOptimizer):    
    def __init__(self, nn, learning_rate, momentum_coef1, momentum_coef2):
        super().__init__(nn, learning_rate)
        self.grad_accums1 = [torch.zeros(p.shape) for p in self.params]
        self.grad_accums2 = [torch.zeros(p.shape) for p in self.params]
        self.momentum_coef1 = momentum_coef1
        self.momentum_coef2 = momentum_coef2
        self.time = 1

    def _update_grads(self):
        for idx, p in enumerate(self.params):
            self.grad_accums1[idx] = self.momentum_coef1 * self.grad_accums1[idx] + (1 - self.momentum_coef1) * p.grad
            self.grad_accums2[idx] = self.momentum_coef2 * self.grad_accums2[idx] + (1 - self.momentum_coef2) * p.grad**2
            m_norm = self.grad_accums1[idx] / (1 - self.momentum_coef1 ** self.time)
            p_norm = self.grad_accums2[idx] / (1 - self.momentum_coef2 ** self.time)
            p.data -= self.learning_rate * m_norm / (p_norm ** 0.5 + EPSILON)
        self.time += 1
