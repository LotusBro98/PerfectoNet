import math

import tensorflow as tf
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from distribution import *

class Layer():
    def __init__(self, ksize=3, stride=2, orient="both", eps=0.1, channels=None, allow_reduce=True, do_conv=True, equalize=False):
        self.ksize = ksize
        self.stride = stride
        self.eps = eps
        self.channels = channels
        self.allow_reduce=allow_reduce
        self.do_conv = do_conv
        self.orient = orient
        self.equalize = equalize
        pass

    def get_mask(self):
        if self.stride == 1:
            mask = np.ones((self.ksize, self.ksize)) * (self.ksize * self.ksize)
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
        if self.do_conv:
            K, K_T, bias_forward, bias_backward, n_features = self._get_optimal_conv_kernel(dataset)
            self.kernel_forward = K[:,:,:,:n_features]
            self.kernel_backward = K_T[:,:,:,:n_features]
            self.bias_forward = bias_forward[:n_features]
            self.bias_backward = bias_backward

        self.input_shape = dataset.shape

    @tf.function
    def forward(self, x, batch_size=1):
        if self.orient == "both":
            stride = (self.stride, self.stride)
        elif self.orient == "hor":
            stride = (1, self.stride)
        elif self.orient == "ver":
            stride = (self.stride, 1)

        if self.do_conv:
            x_conv = []
            for i in range(0, x.shape[0], batch_size):
                batch = x[i: i + batch_size]
                batch = tf.nn.conv2d(
                    batch, self.kernel_forward,
                    (1, stride[0], stride[1], 1),
                    padding="VALID") + self.bias_forward
                # batch = batch.numpy()
                x_conv.append(batch)
            x = tf.concat(x_conv, axis=0)
        return x

    @tf.function
    def backward(self, x, batch_size=1):
        if self.orient == "both":
            stride = (self.stride, self.stride)
        elif self.orient == "hor":
            stride = (1, self.stride)
        elif self.orient == "ver":
            stride = (self.stride, 1)

        b, h, w, c = self.input_shape
        if self.do_conv:
            x_conv = []
            for i in range(0, x.shape[0], batch_size):
                batch0 = x[i: i + batch_size]
                batch = tf.nn.conv2d_transpose(
                    batch0, self.kernel_backward, (batch0.shape[0], h, w, c),
                    (1, stride[0], stride[1], 1), padding="VALID")
                bias = tf.nn.conv2d_transpose(
                    tf.ones_like(batch0[:,:,:,:1]), self.bias_backward, (batch0.shape[0], h, w, c),
                    (1, stride[0], stride[1], 1), padding="VALID")
                batch += bias
                x_conv.append(batch)
            x = tf.concat(x_conv, axis=0)
            mask = np.ones(x.shape)
            if self.ksize == 3 and self.stride == 2:
                if self.orient in ["both", "ver"]:
                    mask[:, 0, :, :] *= 2
                    mask[:, -1, :, :] *= 2
                if self.orient in ["both", "hor"]:
                    mask[:, :, 0, :] *= 2
                    mask[:, :, -1, :] *= 2
            if self.ksize == 2 and self.stride == 1:
                mask[:, 0, :, :] *= 2
                mask[:, -1, :, :] *= 2
                mask[:, :, 0, :] *= 2
                mask[:, :, -1, :] *= 2
            x = x * mask
        return x

    def _get_optimal_conv_kernel(self, dataset, batch_size_cells=100000000):
        assert self.ksize in [1, 2, 3]
        assert self.stride in [1, 2]
        assert self.orient in ["both", "hor", "ver"]

        if self.orient == "both":
            ksize = (self.ksize, self.ksize)
            stride = (self.stride, self.stride)
        elif self.orient == "hor":
            ksize = (1, self.ksize)
            stride = (1, self.stride)
        elif self.orient == "ver":
            ksize = (self.ksize, 1)
            stride = (self.stride, 1)

        Cin = dataset.shape[-1]
        if self.orient == "both":
            cells = dataset.shape[1] * dataset.shape[2] * np.square(dataset.shape[3])
        else:
            cells = dataset.shape[1] * dataset.shape[2] * dataset.shape[3]
        batch_size_images = math.ceil(batch_size_cells / (cells))

        std = np.std(dataset)

        N = 0
        mean_p = 0
        for i in range(0, dataset.shape[0], batch_size_images):
            batch = dataset[i: i + batch_size_images]
            patches = tf.image.extract_patches(
                batch,
                [1, ksize[0], ksize[1], 1],
                [1, stride[0], stride[1], 1],
                [1, 1, 1, 1], padding='VALID')
            patches = tf.reshape(patches, [patches.shape[0] * patches.shape[1] * patches.shape[2], patches.shape[3]])

            mean_p += tf.reduce_sum(patches, axis=0)
            N += patches.shape[0]
        mean_p /= N

        N = 0
        M = 0
        for i in range(0, dataset.shape[0], batch_size_images):
            batch = dataset[i: i + batch_size_images]
            patches = tf.image.extract_patches(
                batch,
                [1, ksize[0], ksize[1], 1],
                [1, stride[0], stride[1], 1],
                [1, 1, 1, 1], padding='VALID')
            patches = tf.reshape(patches, [patches.shape[0] * patches.shape[1] * patches.shape[2], patches.shape[3]])
            patches = patches - mean_p

            batch_size_mat = math.ceil(batch_size_cells / (patches.shape[1] * patches.shape[1]))
            for j in range(0, len(patches), batch_size_mat):
                batch_mat = patches[j:j + batch_size_mat]
                cov = tf.matmul(tf.expand_dims(batch_mat, axis=-1), tf.expand_dims(batch_mat, axis=-2))
                cov = tf.reduce_sum(cov, axis=0)

                N += batch_mat.shape[0]
                M += cov

            print("\rProcessing sample {} / {}".format(i, dataset.shape[0]), end='')
        print()

        M = M / N
        print("SV Decomposition running...")
        U, S, V = np.linalg.svd(M)

        # Ss = np.square(np.cumsum(S))
        # Ss /= Ss[-1]
        # Ss = 1 - Ss
        Ss = np.log(S)
        # Ss = np.sqrt(S / np.max(S))
        # Ss = S / np.sum(S)
        # Ss = np.sqrt(np.cumsum(Ss))
        thresh = self.eps
        # plt.plot(Ss)
        # plt.plot(np.ones_like(Ss) * thresh)
        # plt.show()
        n_features = (tf.math.count_nonzero(Ss > thresh)) if self.channels is None else self.channels
        if (not self.allow_reduce) and n_features < Cin:
            n_features = Cin
        if n_features >= len(S):
            n_features = len(S)

        # V = V[:n_features]

        K = V
        if self.equalize:
            K = np.matmul(np.diag(1/np.sqrt(S / np.average(S)) / std), K)
        bias_forward = -tf.squeeze(tf.matmul(K, tf.expand_dims(mean_p, axis=-1)))
        K = tf.reshape(K, (V.shape[0], ksize[0], ksize[1], Cin))
        K = tf.transpose(K, (1, 2, 3, 0))

        K_T = tf.transpose(V)
        if self.equalize:
            K_T = np.matmul(K_T, np.diag(np.sqrt(S / np.average(S)) * std))
        bias_backward = mean_p
        bias_backward = tf.reshape(bias_backward, (ksize[0], ksize[1], Cin, 1))
        # bias_backward = tf.reduce_mean(bias_backward, axis=0)
        K_T = tf.reshape(K_T, (ksize[0], ksize[1], Cin, V.shape[0]))
        K_T = tf.transpose(K_T, (0, 1, 2, 3))
        mask = self.get_mask()
        K_T /= np.expand_dims(mask, axis=(-1, -2))
        bias_backward /= np.expand_dims(mask, axis=(-1, -2))

        return K, K_T, bias_forward, bias_backward, n_features