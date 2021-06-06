import math

import tensorflow as tf
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from distribution import *

class Layer():
    def __init__(self, ksize=3, stride=2, orient="both", eps=1e-2, distribution_approx_n=100, channels=None, equalize=False, clip_loss=0.01):
        self.ksize = ksize
        self.stride = stride
        self.eps = eps
        self.channels = channels
        self.distribution_approx_n = distribution_approx_n
        self.orient = orient
        self.equalize = equalize
        self.clip_loss = clip_loss
        pass

    def get_mask(self):
        if self.orient == "both":
            ksize = (self.ksize, self.ksize)
            stride = (self.stride, self.stride)
        elif self.orient == "hor":
            ksize = (1, self.ksize)
            stride = (1, self.stride)
        elif self.orient == "ver":
            ksize = (self.ksize, 1)
            stride = (self.stride, 1)

        if self.stride == 1:
            mask = np.ones(ksize) * (ksize[0] * ksize[1])
        elif self.stride == 2:
            if self.ksize == 1:
                mask = np.float32([[1]])
            elif self.ksize == 2:
                mask = np.float32([
                    [1, 1],
                    [1, 1]
                ])
            elif self.ksize == 3:
                if self.orient == "both":
                    mask = np.float32([
                        [4, 2, 4],
                        [2, 1, 2],
                        [4, 2, 4]
                    ])
                elif self.orient == "hor":
                    mask = np.float32([
                        [2, 1, 2]
                    ])
                elif self.orient == "ver":
                    mask = np.float32([
                        [2],
                        [1],
                        [2]
                    ])
        return mask

    def fit(self, dataset, batch_size=1):
        x = dataset

        y = self.convolve(x)

        self.input_shape = x.shape

        K, K_T, bias_forward, bias_backward = self._get_optimal_conv_kernel(y)
        self.kernel_forward = K
        self.kernel_backward = K_T
        self.bias_forward = bias_forward
        self.bias_backward = bias_backward

        x = self.forward(x, batch_size)

        return x

    def get_F_kernel(self):
        k = np.float32([[0, 0], [0, 1], [1, 0], [1, 1], [1, -1]])
        X = np.stack(np.mgrid[-1:2, -1:2], axis=-1)
        kx = np.sum(k * np.expand_dims(X, axis=-2), axis=-1)

        F = np.exp(2j * np.pi / 3 * kx)
        F = np.concatenate([np.real(F), np.imag(F)[:, :, 1:]], axis=-1)
        F = np.float32(F)
        F[:,:,1:] *= np.sqrt(2)

        return F

    def convolve(self, x):
        Cin = x.shape[-1]

        F = self.get_F_kernel()
        F = tf.math.conj(F) / 9

        F = np.stack([F] * Cin, axis=-2)

        y = tf.nn.depthwise_conv2d(x, F, strides=(1, 2, 2, 1), padding='VALID')
        y = tf.reshape(y, y.shape[:-1] + (Cin, 9))
        y0 = y[:,:,:,:,:1]
        y1 = y[:,:,:,:,1:5]
        y2 = y[:,:,:,:,5:]

        y_abs = tf.sqrt(tf.square(y1) + tf.square(y2)) * tf.sign(y2)
        y_ang = tf.math.atan2(y1, tf.abs(y2)) / (np.pi / 2)

        y = tf.concat([y0, y_abs, y_ang], axis=-1)

        y = tf.reshape(y, y.shape[:-2] + (Cin * 9))

        return y

    def deconvolve(self, y):
        Cin = y.shape[-1] // 9

        y = tf.reshape(y, y.shape[:-1] + (Cin, 9))
        y0 = y[:, :, :, :, :1]
        y_abs = y[:, :, :, :, 1:5]
        y_ang = y[:, :, :, :, 5:] * (np.pi / 2)

        y1 = tf.abs(y_abs) * tf.sin(y_ang)
        y2 = y_abs * tf.cos(y_ang)

        y = tf.concat([y0, y1, y2], axis=-1)

        F = self.get_F_kernel()

        y = tf.reduce_sum(tf.expand_dims(tf.expand_dims(y, axis=-2), axis=-2) * F, axis=-1)
        y = tf.transpose(y, (0, 1, 2, 4, 5, 3))
        y = tf.reshape(y, y.shape[:-3] + (Cin * 9))

        return y

    def forward(self, x, batch_size=1, do_2=True):
        x = self.convolve(x)

        x = tf.squeeze(tf.matmul(self.kernel_forward, tf.expand_dims(x, axis=-1))) + self.bias_forward

        return x

    def backward(self, x, batch_size=1):
        x = tf.squeeze(tf.matmul(self.kernel_backward, tf.expand_dims(x, axis=-1))) + self.bias_backward

        x = self.deconvolve(x)

        if self.orient == "both":
            stride = (self.stride, self.stride)
        elif self.orient == "hor":
            stride = (1, self.stride)
        elif self.orient == "ver":
            stride = (self.stride, 1)

        K_T = np.eye(x.shape[-1])
        K_T = np.reshape(K_T, (3, 3, K_T.shape[0] // 9, x.shape[-1]))
        K_T /= np.expand_dims(self.get_mask(), axis=(-1, -2))

        b, h, w, c = self.input_shape
        x_conv = []
        for i in range(0, x.shape[0], batch_size):
            batch0 = x[i: i + batch_size]
            batch = tf.nn.conv2d_transpose(
                batch0, K_T, (batch0.shape[0], h, w, c),
                (1, stride[0], stride[1], 1), padding="VALID")
            # bias = tf.nn.conv2d_transpose(
            #     tf.ones_like(batch0[:,:,:,:1]), self.bias_backward, (batch0.shape[0], h, w, c),
            #     (1, stride[0], stride[1], 1), padding="VALID")
            # batch += bias
            # batch = batch.numpy()
            x_conv.append(batch)
        x = np.concatenate(x_conv, axis=0)

        if self.ksize == 3 and self.stride == 2:
            if self.orient in ["both", "ver"]:
                x[:, 0, :, :] *= 2
                x[:, -1, :, :] *= 2
            if self.orient in ["both", "hor"]:
                x[:, :, 0, :] *= 2
                x[:, :, -1, :] *= 2
        if self.ksize == 2 and self.stride == 1:
            x[:, 0, :, :] *= 2
            x[:, -1, :, :] *= 2
            x[:, :, 0, :] *= 2
            x[:, :, -1, :] *= 2
        return x

    def save(self, n):
        np.save("saved/K{}".format(n), self.kernel_forward)
        np.save("saved/b{}".format(n), self.bias_forward)
        np.save("saved/K_T{}".format(n), self.kernel_backward)
        np.save("saved/b_T{}".format(n), self.bias_backward)
        # np.save("saved/Km{}".format(n), self.Km)
        # np.save("saved/K2{}".format(n), self.K2)
        # np.save("saved/K2_T{}".format(n), self.K2_T)
        # np.save("saved/m2{}".format(n), self.m2)
        # np.save("saved/std{}".format(n), self.channel_std)
        # np.save("saved/stdf{}".format(n), self.stdf)
        np.save("saved/in{}".format(n), self.input_shape)
        # np.save("saved/nc{}".format(n), self.n_features)
        # np.save("saved/xs{}".format(n), self.xs)
        # np.save("saved/ys{}".format(n), self.ys)

    def load(self, n):
        self.kernel_forward = np.load("saved/K{}.npy".format(n))
        self.bias_forward = np.load("saved/b{}.npy".format(n))
        self.kernel_backward = np.load("saved/K_T{}.npy".format(n))
        self.bias_backward = np.load("saved/b_T{}.npy".format(n))
        # self.Km = np.load("saved/Km{}.npy".format(n))
        # self.K2 = np.load("saved/K2{}.npy".format(n))
        # self.K2_T = np.load("saved/K2_T{}.npy".format(n))
        # self.m2 = np.load("saved/m2{}.npy".format(n))
        # self.channel_std = np.load("saved/std{}.npy".format(n))
        # self.stdf = np.load("saved/stdf{}.npy".format(n))
        self.input_shape = np.load("saved/in{}.npy".format(n))
        # self.n_features = np.load("saved/nc{}.npy".format(n))
        # self.xs = np.load("saved/xs{}.npy".format(n))
        # self.ys = np.load("saved/ys{}.npy".format(n))

    def _get_optimal_conv_kernel(self, dataset, batch_size_images=16):
        mean = np.average(dataset, axis=(0, 1, 2))

        N = 0
        M = 0
        for i in range(0, dataset.shape[0], batch_size_images):
            batch = dataset[i: i + batch_size_images]
            batch = tf.reshape(batch, [batch.shape[0] * batch.shape[1] * batch.shape[2], batch.shape[3]])
            cov = tf.matmul(tf.expand_dims(batch, axis=-1), tf.expand_dims(batch, axis=-2))
            cov = tf.reduce_sum(cov, axis=0)

            N += batch.shape[0]
            M += cov

            print("\rProcessing sample {} / {}".format(i, dataset.shape[0]), end='')
        print()
        M = M / N

        print("SV Decomposition running...")
        U, S, V = np.linalg.svd(M, full_matrices=False)

        n_samples = np.sum(np.sqrt(S) > self.eps)

        plt.plot(np.sqrt(S))
        plt.plot(np.ones_like(S) * self.eps)
        plt.show()

        V = V[:n_samples]

        K = V
        bias_forward = -tf.squeeze(tf.matmul(K, tf.expand_dims(mean, axis=-1)))

        K_T = V.T
        bias_backward = mean

        return K, K_T, bias_forward, bias_backward