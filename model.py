import tensorflow as tf
import numpy as np
from layer import Layer

class Model():
    def __init__(self, n_layers, eps, approx_n=100):
        self.layers = []
        self.approx_n = approx_n
        for i in range(n_layers):
            self.layers.append(Layer(eps=eps[i], distribution_approx_n=approx_n))
            # self.layers.append(Layer(ksize=1, stride=1, eps=eps[i]+1, distribution_approx_n=approx_n))

    def fit(self, dataset, batch_size=1):
        x = dataset

        xs, ys = Layer.build_distribution(x, self.approx_n)
        self.xs0 = xs
        self.ys0 = ys

        x = Layer.remap_distribution(x, self.xs0, self.ys0)

        for layer in self.layers:
            layer.fit(x, batch_size)
            x = layer.forward(x)
            print(x.shape)

    def forward(self, x, batch_size=1):
        x = Layer.remap_distribution(x, self.xs0, self.ys0)

        for layer in self.layers:
            x = layer.forward(x, batch_size)
        return x

    def backward(self, x, batch_size=1):
        for layer in self.layers[::-1]:
            x = layer.backward(x, batch_size)

        x = Layer.remap_distribution(x, self.ys0, self.xs0)
        return x