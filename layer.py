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

        self.input_shape = x.shape

        K, K_T, bias_forward, bias_backward = self._get_optimal_conv_kernel(x)
        self.kernel_forward = K
        self.kernel_backward = K_T
        self.bias_forward = bias_forward
        self.bias_backward = bias_backward

        print("Processing dataset ...")
        x = self.forward(x, batch_size, do_2=False)

        self.kernel_backward /= np.expand_dims(self.get_mask(), axis=(-1, -2))
        self.bias_backward /= np.expand_dims(self.get_mask(), axis=(-1, -2))

        return x

    def get_ABC_mat(self, dx=1):
        ksize = (3, 3)
        stride = (2, 2)

        xy_m = np.ones((3, 3, 2), dtype=np.float32)
        xy_m[:, :, 0] = np.expand_dims(np.linspace(-1, 1, 3), axis=1)
        xy_m[:, :, 1] = np.expand_dims(np.linspace(-1, 1, 3), axis=0)
        x = np.reshape(xy_m[:, :, 1], (3 * 3,)) * dx
        y = np.reshape(xy_m[:, :, 0], (3 * 3,)) * dx

        A = np.vstack([x * x, 2 * x * y, y * y, x, y, np.ones_like(x)]).T
        return A

    def get_ABC_inv_mat(self, F, sigma=1, dx=1):
        ksize = (3, 3)
        stride = (2, 2)

        xy_m = np.ones((3, 3, 2), dtype=np.float32)
        xy_m[:, :, 0] = np.expand_dims(np.linspace(-1, 1, 3), axis=1)
        xy_m[:, :, 1] = np.expand_dims(np.linspace(-1, 1, 3), axis=0)
        x = np.reshape(xy_m[:, :, 1], (3 * 3,)) * dx
        y = np.reshape(xy_m[:, :, 0], (3 * 3,)) * dx

        A = np.vstack([x * x, 2 * x * y, y * y, x, y, np.ones_like(x)]).T

        # w = np.expand_dims(np.exp(-0.5 * (x*x + y*y) / np.square(sigma)), axis=-1)
        w = np.reshape(1 / self.get_mask(), (9, 1))
        # w = 1
        A = w * A
        F = w * F

        F_1 = np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)
        ABC = np.matmul(F_1, F)
        return ABC

    def convolve(self, x):
        ksize = (3, 3)
        stride = (2, 2)

        patches = tf.image.extract_patches(
            x,
            [1, ksize[0], ksize[1], 1],
            [1, stride[0], stride[1], 1],
            [1, 1, 1, 1], padding='VALID')

        patches = np.reshape(patches, patches.shape[:3] + (ksize[0] * ksize[1], x.shape[-1]))

        ABC = self.get_ABC_inv_mat(patches)

        axx, axy, ayy, bx, by, c = np.split(ABC, 6, axis=-2)
        A = np.transpose(np.squeeze(np.asarray([[axx, axy], [axy, ayy]]), axis=-2), (2, 3, 4, 5, 0, 1))
        b = np.transpose(np.squeeze(np.asarray([bx, by]), axis=-2), (1, 2, 3, 4, 0))
        c = np.expand_dims(np.squeeze(c, axis=-2), axis=-1)

        U, S, V = np.linalg.svd(A)

        sU1 = np.expand_dims(np.float32(U[:, :, :, :, 0, 0] >= 0) * 2 - 1, axis=-1)
        sU2 = np.expand_dims(np.float32(U[:, :, :, :, 1, 1] >= 0) * 2 - 1, axis=-1)
        sV1 = np.expand_dims(np.float32(V[:, :, :, :, 0, 0] >= 0) * 2 - 1, axis=-1)
        sV2 = np.expand_dims(np.float32(V[:, :, :, :, 1, 1] >= 0) * 2 - 1, axis=-1)

        U[:, :, :, :, :, 0] *= sU1
        U[:, :, :, :, :, 1] *= sU2
        V[:, :, :, :, 0] *= sV1
        V[:, :, :, :, 1] *= sV2
        S[:, :, :, :, 0:1] *= sU1 * sV1
        S[:, :, :, :, 1:2] *= sU2 * sV2

        cosf, sinf = np.split(V[:, :, :, :, :, 0], 2, axis=-1)

        f = np.arctan(sinf / cosf)
        V_T = np.transpose(V, (0, 1, 2, 3, 5, 4))

        # b_ = b
        b_ = np.squeeze(np.matmul(V_T, np.expand_dims(b, axis=-1)), axis=-1)

        Sfbc = np.concatenate([S, f, b_, c], axis=-1)
        Sfbc = np.reshape(Sfbc, Sfbc.shape[:-2] + (Sfbc.shape[-2] * Sfbc.shape[-1],))
        return Sfbc

    def deconvolve(self, Sfbc):
        Sfbc = np.reshape(Sfbc, Sfbc.shape[:-1] + (Sfbc.shape[-1] // 6, 6))
        S, f, b_, c = np.split(Sfbc, [2, 3, 5], axis=-1)
        f = np.squeeze(f, axis=-1)
        sinf, cosf = np.sin(f), np.cos(f)
        c = np.squeeze(c, axis=-1)
        S = np.transpose(S, (4, 0, 1, 2, 3))

        V = np.transpose(np.asarray([[cosf, -sinf], [sinf, cosf]]), (2, 3, 4, 5, 0, 1))
        U = np.transpose(V, (0, 1, 2, 3, 5, 4))
        S = np.transpose(np.asarray([[S[0], np.zeros_like(S[0])], [np.zeros_like(S[0]), S[1]]]),
                         (2, 3, 4, 5, 0, 1))
        A = np.matmul(U, np.matmul(S, V))
        b = np.matmul(V, np.expand_dims(b_, axis=-1)).squeeze(axis=-1)
        # b = b_

        Abc = np.stack(
            [A[:, :, :, :, 0, 0], A[:, :, :, :, 0, 1], A[:, :, :, :, 1, 1], b[:, :, :, :, 0], b[:, :, :, :, 1], c],
            axis=-2)

        A = self.get_ABC_mat()

        x = np.matmul(A, Abc)
        print("sas21", x[0, 20, :5, 0])
        x = np.reshape(x, x.shape[:-2] + (x.shape[-2] * x.shape[-1],))
        return x

    def forward(self, x, batch_size=1, do_2=True):
        x = self.convolve(x)

        return x

    def backward(self, x, batch_size=1):
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
        std_p = 0
        for i in range(0, dataset.shape[0], batch_size_images):
            batch = dataset[i: i + batch_size_images]
            patches = tf.image.extract_patches(
                batch,
                [1, ksize[0], ksize[1], 1],
                [1, stride[0], stride[1], 1],
                [1, 1, 1, 1], padding='VALID')
            patches = tf.reshape(patches, [patches.shape[0] * patches.shape[1] * patches.shape[2], patches.shape[3]])
            patches = patches - mean_p
            std_p += tf.reduce_sum(tf.square(patches), axis=0)

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
        std_p = tf.sqrt(std_p / N)
        # M = M / np.outer(std_p, std_p)
        print("SV Decomposition running...")
        U, S, V = np.linalg.svd(M, full_matrices=False)

        mask = self.get_mask()

        K = V
        if self.equalize:
            K = np.matmul(np.diag(1/np.sqrt(S / np.average(S)) / std), K)
        else:
            K = K / ((np.count_nonzero(mask) / np.sum(1 / mask)))
        bias_forward = -tf.squeeze(tf.matmul(K, tf.expand_dims(mean_p, axis=-1)))
        K = tf.reshape(K, (V.shape[0], ksize[0], ksize[1], Cin))
        K = tf.transpose(K, (1, 2, 3, 0))

        K_T = U
        if self.equalize:
            K_T = np.matmul(K_T, np.diag(np.sqrt(S / np.average(S)) * std))
        else:
            K_T = K_T * ((np.count_nonzero(mask) / np.sum(1 / mask)))
        bias_backward = mean_p
        bias_backward = tf.reshape(bias_backward, (ksize[0], ksize[1], Cin, 1))
        # bias_backward /= np.expand_dims(mask, axis=(-1, -2))
        K_T = tf.reshape(K_T, (ksize[0], ksize[1], Cin, V.shape[0]))
        K_T = tf.transpose(K_T, (0, 1, 2, 3))
        # K_T /= np.expand_dims(mask, axis=(-1, -2))

        return K, K_T, bias_forward, bias_backward