import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-5

class Linear(nn.Module):
    def __init__(self, fan_in, fan_out, initialization="standard"):
        super().__init__()
        self.linear = nn.Linear(fan_in, fan_out)
        if initialization == "normal_glorot":
            nn.init.normal_(self.linear.weight, mean=0, std=np.sqrt(2 / (fan_in + fan_out)))
        elif initialization == "uniform_glorot":
            limit = np.sqrt(6/(fan_in+fan_out))
            nn.init.uniform_(self.linear.weight, -limit, limit)
        
    def forward(self, X):
        return self.linear(X)


class ReLU(nn.Module):
    def forward(self, X):
        return F.relu(X)


class Sigmoid(nn.Module):
    def forward(self, X):
        return torch.sigmoid(X)


class Softmax(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, X):
        exp_X = torch.exp(X)
        return exp_X / (torch.sum(exp_X, dim=self.dim, keepdim=True) + EPSILON)


class Embedding(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, X):
        return self.embedding(X.long())


class Flatten(nn.Module):
    def forward(self, X):
        return X.view(X.size(0), -1)


class MaxPooling(nn.Module):
    def __init__(self, size, stride):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=size, stride=stride)

    def forward(self, X):
        return self.pool(X)


class LayerNorm(nn.Module):
    def __init__(self, fan_in):
        super().__init__()
        self.layer_norm = nn.LayerNorm([fan_in,])

    def forward(self, X):
        return self.layer_norm(X)


class NN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def params(self):
        return self.parameters()
