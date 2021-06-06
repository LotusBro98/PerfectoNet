import math

import tensorflow as tf
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from distribution import *


class Layer():
    def __init__(self, ksize=3, stride=2, orient="both", eps=1e-2, distribution_approx_n=100, channels=None,
                 equalize=False, clip_loss=0.01):
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

    def fit(self, dataset, batch_size=1, kz_eps=0.2, prob_eps=0.1):
        x = dataset
        Cin = x.shape[-1]

        self.input_shape = x.shape

        x0, xf = self.convolve(x, train=True)

        K, K_T, bias_forward, bias_backward = self._get_optimal_conv_kernel(xf, reduce=False)
        Kz = np.concatenate([K[:, i::8] + 1j * K[:, i + 4::8] for i in range(4)], axis=-1)
        self.Kz = Kz

        groups = self.get_abs_groups(Kz)
        # lonely = [g[0] for g in groups if len(g) == 1]
        groups = [g for g in groups if len(g) > 1]

        # self.KL = K[lonely]
        # self.KL_T = K[lonely].T
        # x1 = np.squeeze(np.matmul(self.KL, np.expand_dims(xf, axis=-1)), axis=-1)

        print(groups)

        # z = np.concatenate([xf[:,:,:,i::8] + 1j*xf[:,:,:,i+4::8] for i in range(4)], axis=-1)
        # z = np.reshape(z, z.shape[:-1] + (4, Cin))
        # z = np.transpose(z, (0, 1, 2, 4, 3))
        # k = np.float32([[0, 1], [1, 1], [1, 0], [1, -1]])
        # k = np.arctan2(k[:,1], k[:,0])
        # print(k)
        # kf = np.sum(np.square(np.abs(z)) * k, -1) / np.sum(np.square(np.abs(z)), -1)
        # show_common_distributions(kf)
        # print(kf.shape)
        # np.random.shuffle(z)
        # plt.figure(figsize=(20, 5))
        # plt.imshow(np.abs(z)[:100].T)
        # plt.show()

        # F0, F = self.get_F_mat()
        # for group in groups:
        #     f, ax = plt.subplots(1, len(group))
        #     for i in range(len(group)):
        #         a = K[group[i]]
        #         a = np.reshape(a, (3, 8))
        #         a = np.matmul(np.expand_dims(F, axis=(-2, -3)), np.expand_dims(a, axis=-1)).squeeze()
        #         if len(group) > 1:
        #           ax[i].imshow(a * 0.5 + 0.5)
        #         else:
        #           ax.imshow(a * 0.5 + 0.5)
        #     plt.show()

        yr = []
        ya = []
        self.group_K = []
        for group in zip(groups):
            Kf = np.matmul(K[group].T, K[group])
            Kfz = np.concatenate([Kf[i::8] + 1j * Kf[i + 4::8] for i in range(4)], axis=0)
            kz = Kz[group][0]
            kz = kz / np.abs(kz)
            take = np.linalg.norm(Kfz, axis=-1) > kz_eps
            Kfz = Kfz[take] / np.expand_dims(kz[take], axis=-1)

            kzr, kzi = np.real(np.diag(kz)[:, take]), np.imag(np.diag(kz)[:, take])
            Kfz_T = np.concatenate([
                np.concatenate([kzr, -kzi], axis=-1),
                np.concatenate([kzi, kzr], axis=-1)
            ], axis=0)
            Kfz_T = np.reshape(Kfz_T, (2, 4, Cin, Kfz_T.shape[-1]))
            Kfz_T = np.transpose(Kfz_T, (2, 0, 1, 3))
            Kfz_T = np.reshape(Kfz_T, (8 * Cin, Kfz_T.shape[-1]))

            z = np.squeeze(np.matmul(Kfz, np.expand_dims(xf, axis=-1)), axis=-1)
            z0 = z
            if np.average(np.linalg.norm(np.abs(z), axis=-1) > self.eps) < prob_eps:
                continue
            z = z[np.linalg.norm(np.abs(z), axis=-1) > self.eps]

            # xr = np.expand_dims(np.average(np.abs(z), -1), -1))
            # xr = np.abs(z)
            xa = np.angle(z) / 2 / np.pi
            # xr = xr[(np.abs(xa) < (0.5 - self.eps)).all(axis=-1)]
            xa = xa[(np.abs(xa) < (0.5 - self.eps)).all(axis=-1)]

            # xg = np.concatenate([xr, xa], axis=-1)

            # show_common_distributions(xr)
            # show_common_distributions(xa)

            Kg, Kg_T, bg, bg_T = self._get_optimal_conv_kernel(xa, reduce=True)
            plt.imshow(np.abs(Kg))
            plt.show()

            self.group_K.append((Kfz, Kfz_T, Kg, Kg_T, bg, bg_T))

            xr0 = np.expand_dims(np.average(np.abs(z0), -1), -1)
            xa0 = np.angle(z0) / 2 / np.pi
            xa0[np.linalg.norm(xr0, axis=-1) <= self.eps, :] = 0
            # xg0 = np.concatenate([xr0, xa0], axis=-1)

            yg0 = np.squeeze(np.matmul(Kg, np.expand_dims(xa0, axis=-1)), axis=-1) + bg
            yr.append(xr0)
            ya.append(yg0)

        yr = np.concatenate(yr, axis=-1)
        # yr1 = np.expand_dims(np.sum(yr, -1), -1)
        # yr = np.concatenate([yr1, yr / yr1], axis=-1)

        y = np.concatenate([x0, yr], axis=-1)

        self.K1, self.K1_T, self.b1, self.b1_T = self._get_optimal_conv_kernel(y, reduce=True, eps=self.eps / 2)
        # show_common_distributions(y)
        y = np.squeeze(np.matmul(self.K1, np.expand_dims(y, axis=-1)), axis=-1) + self.b1

        # print(y.shape)

        return y

    def get_abs_groups(self, Kz, eps=0.1):
        import time
        Ka = np.abs(Kz)
        Ka1 = np.reshape(Ka, (Ka.shape[0], 4, Ka.shape[-1] // 4))
        Ka1 = np.sqrt(np.sum(np.square(Ka1), axis=1))
        plt.figure(figsize=(20, 10))
        plt.imshow(Ka1.T[::-1])
        plt.show()

        plt.figure(figsize=(20, 10))
        # dis = np.reshape(Ka, (Ka.shape[0], 4, Ka.shape[-1] // 4))
        # dis = np.transpose(dis, (0, 2, 1))
        # dis = np.reshape(dis, (Ka.shape[0], Ka.shape[-1]))
        plt.imshow(Ka.T[::-1])
        plt.show()
        D = np.dot(Ka, Ka.T)
        rest = np.int32(list(range(len(Ka))))
        groups = []
        while len(rest) > 0:
            # plt.imshow(D)
            # plt.show()
            idx = np.argwhere(D[0] > (1 - eps))[:, 0]
            keep = [i for i in range(len(D)) if i not in idx]
            D = D[keep][:, keep]
            groups.append(list(rest[idx]))
            rest = rest[keep]
            # print(idx)
            time.sleep(0.01)

        return groups

    def get_F_mat(self):

        x = np.float32(np.stack(np.mgrid[-1:2, -1:2], axis=-1))
        k = np.float32([[0, 1], [1, 1], [1, 0], [1, -1]])
        kx = np.inner(x, k)

        F_1 = np.ones((3, 3, 1), np.float32) / 3
        F_sin = np.sin(2 * np.pi / 3 * kx) / 3 * np.sqrt(2)
        F_cos = np.cos(2 * np.pi / 3 * kx) / 3 * np.sqrt(2)
        F = np.concatenate([F_cos, F_sin], axis=-1)

        return F_1, F

    def convolve(self, x, train=False):
        Cin = x.shape[-1]

        F0, F = self.get_F_mat()

        F0 /= 3
        F /= 3

        K0 = np.repeat(np.reshape(F0, (3, 3, 1, 1)), Cin, axis=-2)
        KF = np.repeat(np.reshape(F, (3, 3, 1, 8)), Cin, axis=-2)

        y0 = tf.nn.depthwise_conv2d(x, K0, (1, 2, 2, 1), 'VALID').numpy()
        yf = tf.nn.depthwise_conv2d(x, KF, (1, 2, 2, 1), 'VALID').numpy()

        return y0, yf

    def deconvolve(self, x0, xf, batch_size=1):
        Cin = self.input_shape[-1]

        F0, F = self.get_F_mat()

        F0 *= 3
        F *= 3

        xf = np.reshape(xf, xf.shape[:-1] + (Cin, 8))
        xf = np.inner(xf, F)
        xf = np.transpose(xf, (0, 1, 2, 4, 5, 3))
        xf = np.reshape(xf, xf.shape[:3] + (Cin * 9,))

        x0 = np.expand_dims(x0, axis=-1)
        x0 = np.inner(x0, F0)
        x0 = np.transpose(x0, (0, 1, 2, 4, 5, 3))
        x0 = np.reshape(x0, x0.shape[:3] + (Cin * 9,))

        x = x0 + xf

        x = self.conv_transpose(x, batch_size)

        return x

    def forward(self, x, batch_size=1, do_2=True):
        x0, xf = self.convolve(x, train=True)

        x1 = np.squeeze(np.matmul(self.KL, np.expand_dims(xf, axis=-1)), axis=-1)

        y = []
        for Kfz, Kfz_T, Kg, Kg_T, bg, bg_T in self.group_K:
            z = np.squeeze(np.matmul(Kfz, np.expand_dims(xf, axis=-1)), axis=-1)

            xr = np.abs(z)
            xa = np.angle(z) / 2 / np.pi
            xa[xr < self.eps] = 0

            xg = np.concatenate([xr, xa], axis=-1)

            yg = np.squeeze(np.matmul(Kg, np.expand_dims(xg, axis=-1)), axis=-1) + bg
            y.append(yg)
        y = np.concatenate([x0, x1] + y, axis=-1)

        y = np.squeeze(np.matmul(self.K1, np.expand_dims(y, axis=-1)), axis=-1) + self.b1

        return y

    def conv_transpose(self, x, batch_size=1):
        if self.orient == "both":
            stride = (self.stride, self.stride)
        elif self.orient == "hor":
            stride = (1, self.stride)
        elif self.orient == "ver":
            stride = (self.stride, 1)

        K_T = np.eye(x.shape[-1], dtype=np.float32)
        K_T = np.reshape(K_T, (3, 3, K_T.shape[0] // 9, x.shape[-1]))
        K_T /= np.expand_dims(self.get_mask(), axis=(-1, -2))

        b, h, w, c = self.input_shape
        c = K_T.shape[2]
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

    def backward(self, y, batch_size=1):

        y = np.squeeze(np.matmul(self.K1_T, np.expand_dims(y, axis=-1)), axis=-1) + self.b1_T

        Cin = self.input_shape[-1]
        lens = [Kg.shape[0] for Kfz, Kfz_T, Kg, Kg_T, bg, bg_T in self.group_K]
        x0 = y[:, :, :, :Cin]
        xr = y[:, :, :, Cin:Cin + len(self.group_K)]
        # x1 = y[:,:,:,Cin:Cin+self.KL.shape[0]]
        # y_groups = np.split(y[:,:,:,Cin+self.KL.shape[0]:], np.cumsum(lens)[:-1], axis=-1)
        y_groups = np.split(y[:, :, :, Cin + len(self.group_K):], np.cumsum(lens)[:-1], axis=-1)

        # xr1 = xr[:,:,:,0:1]
        # xr = xr[:,:,:,1:] * xr1

        xf = 0  # np.squeeze(np.matmul(self.KL_T, np.expand_dims(x1, axis=-1)), axis=-1)
        for xr, ya, (Kfz, Kfz_T, Kg, Kg_T, bg, bg_T) in zip(np.split(xr, xr.shape[-1], axis=-1), y_groups,
                                                            self.group_K):
            xa = np.squeeze(np.matmul(Kg_T, np.expand_dims(ya, axis=-1)), axis=-1) + bg_T
            # xr, xa = np.split(xg, 2, axis=-1)
            xr = np.clip(xr, 0, None)
            xa = xa * 2 * np.pi

            xg = np.concatenate([
                xr * np.cos(xa),
                xr * np.sin(xa)
            ], axis=-1)

            xg = np.squeeze(np.matmul(Kfz_T, np.expand_dims(xg, axis=-1)), axis=-1)

            xf += xg

        x = self.deconvolve(x0, xf, batch_size)

        return x

    def save(self, n):
        pass
        # np.save("saved/K{}".format(n), self.kernel_forward)
        # np.save("saved/b{}".format(n), self.bias_forward)
        # np.save("saved/K_T{}".format(n), self.kernel_backward)
        # np.save("saved/b_T{}".format(n), self.bias_backward)
        # np.save("saved/Km{}".format(n), self.Km)
        # np.save("saved/Kc_1_{}".format(n), self.Kc1)
        # np.save("saved/Kc_2_{}".format(n), self.Kc2)
        # np.save("saved/K2_T{}".format(n), self.K2_T)
        # np.save("saved/m2{}".format(n), self.m2)
        # np.save("saved/std{}".format(n), self.channel_std)
        # np.save("saved/stdf{}".format(n), self.stdf)
        # np.save("saved/in{}".format(n), self.input_shape)
        # np.save("saved/nc{}".format(n), self.n_features)
        # np.save("saved/xs{}".format(n), self.xs)
        # np.save("saved/ys{}".format(n), self.ys)

    def load(self, n):
        pass
        # self.kernel_forward = np.load("saved/K{}.npy".format(n))
        # self.bias_forward = np.load("saved/b{}.npy".format(n))
        # self.kernel_backward = np.load("saved/K_T{}.npy".format(n))
        # self.bias_backward = np.load("saved/b_T{}.npy".format(n))
        # self.Km = np.load("saved/Km{}.npy".format(n))
        # self.Kc1 = np.load("saved/Kc_1_{}.npy".format(n))
        # self.Kc2 = np.load("saved/Kc_2_{}.npy".format(n))
        # self.K2_T = np.load("saved/K2_T{}.npy".format(n))
        # self.m2 = np.load("saved/m2{}.npy".format(n))
        # self.channel_std = np.load("saved/std{}.npy".format(n))
        # self.stdf = np.load("saved/stdf{}.npy".format(n))
        # self.input_shape = np.load("saved/in{}.npy".format(n))
        # self.n_features = np.load("saved/nc{}.npy".format(n))
        # self.xs = np.load("saved/xs{}.npy".format(n))
        # self.ys = np.load("saved/ys{}.npy".format(n))

    def _get_optimal_conv_kernel(self, dataset, eps=None, reduce=True, batch_size_cells=100000000):
        if eps is None:
            eps = self.eps
        dataset = flatten(dataset)

        mean_p = np.average(dataset, axis=0)

        N = 0
        M = 0
        batch_size = int(np.ceil(batch_size_cells / np.square(dataset.shape[-1])))
        for i in range(0, dataset.shape[0], batch_size):
            batch = dataset[i: i + batch_size]
            nz = np.float32(np.abs(batch) > self.eps)
            nz = np.matmul(np.expand_dims(nz, axis=-1), np.expand_dims(nz, axis=-2))
            cov = tf.matmul(tf.expand_dims(batch, axis=-1), tf.expand_dims(batch, axis=-2))

            cov = tf.reduce_sum(cov, axis=0)

            N += batch.shape[0]
            M += cov

        M = M / N
        U, S, V = np.linalg.svd(M, full_matrices=False)

        # mask = self.get_mask()
        if reduce:
            n_features = np.count_nonzero(np.sqrt(S) > eps)
            V = V[:n_features]
            S = S[:n_features]

        K = V
        bias_forward = -np.squeeze(np.matmul(K, np.expand_dims(mean_p, axis=-1)))

        K_T = V.T
        bias_backward = mean_p

        return K, K_T, bias_forward, bias_backward