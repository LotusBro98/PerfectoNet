import tensorflow as tf
import numpy as np

from distribution import remap_distribution, build_distribution, common_distribution, show_common_distributions, \
    get_independent_channels
from layer import Layer

class Model():
    def __init__(self, n_layers, epsL, epsD, approx_n=100):
        self.layers = []
        self.approx_n = approx_n
        # self.layers.append(Layer(do_conv=False))
        for i in range(n_layers):
            if hasattr(epsL[i], "__len__"):
                epsL1 = epsL[i][0]
                epsL2 = epsL[i][1]
            else:
                epsL1 = epsL[i]
                epsL2 = epsL[i]

            if hasattr(epsD[i], "__len__"):
                epsD1 = epsD[i][0]
                epsD2 = epsD[i][1]
            else:
                epsD1 = epsD[i]
                epsD2 = epsD[i]

            # self.layers.append(Layer(ksize=2, stride=1, eps=eps1, distribution_approx_n=approx_n))
            # self.layers.append(Layer(ksize=2, stride=2, eps=eps2, distribution_approx_n=approx_n))

            self.layers.append(Layer(ksize=3, stride=2, orient="ver", epsL=epsL1, epsD=epsD1, distribution_approx_n=approx_n))
            self.layers.append(Layer(ksize=3, stride=2, orient="hor", epsL=epsL2, epsD=epsD2, distribution_approx_n=approx_n))
            # self.layers.append(Layer(ksize=3, stride=2, orient="both", eps=eps2, distribution_approx_n=approx_n))


        # self.layers.append(Layer(ksize=1, stride=1, eps=eps[-1], distribution_approx_n=approx_n))

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