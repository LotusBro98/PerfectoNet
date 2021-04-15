import tensorflow as tf
import numpy as np

from distribution import remap_distribution, build_distribution, common_distribution, show_common_distributions, \
    get_independent_channels
from layer import Layer

class Model():
    def __init__(self, n_layers, eps, approx_n=100):
        self.layers = []
        self.approx_n = approx_n

        # eps = np.exp(np.linspace(np.log(eps_start), np.log(eps_end), n_layers * 2))

        for i in range(n_layers):
            # eps1 = eps[2*i]
            # eps2 = eps[2*i+1]
            eps1 = eps[i]
            eps2 = eps1

            self.layers.append(Layer(ksize=3, stride=2, orient="both", eps=eps1, distribution_approx_n=approx_n))

    def fit(self, dataset, batch_size=1, skip_layers=0):
        x = dataset

        for i, layer in enumerate(self.layers):
            if i < skip_layers:
                layer.load(i)
                x = layer.forward(x, batch_size)
            else:
                x = layer.fit(x, batch_size)
                # centers = layer.m2[:, :6]
                # stdf = layer.stdf[:, :6]
                # show_common_distributions(x[:, :, :, :9])
            print(x.shape)

        return x

    def forward(self, x, batch_size=1):

        for layer in self.layers:
            x = layer.forward(x, batch_size)

        return x

    def backward(self, x, batch_size=1):
        print(x.shape)
        for layer in self.layers[::-1]:
            x = layer.backward(x, batch_size)
            print(x.shape)

        return x

    def save(self):
        for i in range(len(self.layers)):
            self.layers[i].save(i)

    def load(self):
        for i in range(len(self.layers)):
            self.layers[i].load(i)