import tensorflow as tf
import numpy as np

from distribution import remap_distribution, build_distribution, common_distribution, show_common_distributions, \
    get_independent_channels
from layer import Layer

class Model():
    def __init__(self, n_layers, eps):
        self.layers = []
        # self.layers.append(Layer(do_conv=False))
        for i in range(n_layers):
            if hasattr(eps[i], "__len__"):
                eps1 = eps[i][0]
                eps2 = eps[i][1]
            else:
                eps1 = eps[i]
                eps2 = eps[i]

            # self.layers.append(Layer(ksize=2, stride=1, eps=eps1))
            # self.layers.append(Layer(ksize=2, stride=2, eps=eps2))
            # epsL_i = epsL if i > 0 else -8

            # self.layers.append(Layer(ksize=3, stride=2, orient="ver", eps=eps1))
            # self.layers.append(Layer(ksize=3, stride=2, orient="hor", eps=eps2))
            self.layers.append(Layer(ksize=3, stride=2, orient="both", eps=eps2))


        # self.layers.append(Layer(ksize=1, stride=1, eps=eps[-1]))

    def fit(self, dataset, batch_size=1):
        x = dataset

        # xs, ys = build_distribution(x, self.approx_n)
        # self.xs0 = xs
        # self.ys0 = ys
        #
        # x = remap_distribution(x, self.xs0, self.ys0)

        for layer in self.layers:
            layer.fit(x, batch_size)
            x = layer.forward(x)
            # independent = get_independent_channels(x)
            # print(len(independent), independent)
            # show_common_distributions(x)
            print(x.shape)
        return x

    def forward(self, x, batch_size=1):
        # x = remap_distribution(x, self.xs0, self.ys0)

        for layer in self.layers:
            x = layer.forward(x, batch_size)
        return x

    def backward(self, x, batch_size=1):
        for layer in self.layers[::-1]:
            x = layer.backward(x, batch_size)

        # x = remap_distribution(x, self.ys0, self.xs0)
        return x