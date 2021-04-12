import math

import tensorflow as tf
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from distribution import *

class Layer():
    def __init__(self, ksize=3, stride=2, orient="both", eps=1e-2, distribution_approx_n=100, channels=None, equalize=False, clip_loss=0.001):
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

        self.get_2_order_kernel(x)
        print("Slice...")
        x = self._forward_2_order(x, train=True)

        return x

    def get_peak(self, x, n_features, min_len=1, eps=None):
        if eps is None:
            eps = self.eps * 3
        # n_splits = 10

        x = flatten(x)
        rest = x
        outside = []
        for i in range(n_features):
            ch = rest[:, i]
            m, M = np.min(ch), np.max(ch) + 1e-4
            n_splits = int(math.ceil((M - m) / eps)) + 1
            # n_splits = min(int(math.ceil((M - m) / eps)) + 1, 6)
            checks = np.linspace(m, M, n_splits)
            density = np.stack([np.sum((ch >= checks[i]) & (ch < checks[i + 1])) for i in range(len(checks) - 1)])
            peak_i = np.argmax(density)
            outside.append(rest[(ch < checks[peak_i]) | (ch >= checks[peak_i + 1])])
            rest = rest[(ch >= checks[peak_i]) & (ch < checks[peak_i + 1])]

        # if len(rest) < min_len:
        peak = np.average(rest, axis=0)
        dist = np.linalg.norm((x - peak), axis=-1)
        idxs = np.argsort(dist)
        dist = np.sort(dist)
        take_n = np.count_nonzero(dist <= 2 * eps)
        take_n = max(min_len, take_n)
        inside = np.take(x, idxs[:take_n], axis=0)
        outside = np.take(x, idxs[take_n:], axis=0)
        # else:
        #     inside = rest
        #     outside = np.concatenate(outside, axis=0)

        return inside, outside

    def get_centers(self):
        n_features = self.K2.shape[2]
        centers = self.m2[:,:n_features]
        return centers

    def get_2_order_kernel(self, x, odd=0.005, batch_size=1000):
        x = flatten(x)
        # self.channel_std = np.std(x, axis=0)

        min_len = int(len(x) * odd)

        # xp, xnp = self.get_peak(x, x.shape[-1], 0.9)
        std = np.std(x, axis=0)
        n_features = np.count_nonzero(std >= self.eps)
        # if (min_len < n_features):
        #     n_features = int(np.sqrt(min(n_features, min_len) * n_features))
        #     min_len = n_features
        take_channels = np.argsort(std)[::-1]
        print("{} main channels, {} secondary".format(n_features, x.shape[-1] - n_features))

        x = np.take(x, take_channels, axis=-1)
        self.kernel_forward = np.take(self.kernel_forward, take_channels, axis=-1)
        self.kernel_backward = np.take(self.kernel_backward, take_channels, axis=-1)
        self.bias_forward = np.take(self.bias_forward, take_channels, axis=-1)
        self.channel_std = np.take(std, take_channels, axis=-1)

        rest = x
        groups = []
        Ks = []
        ms = []
        stds = []
        # show_common_distributions(rest[:,:4])
        while len(rest) > min_len:
            inside, outside = self.get_peak(rest, n_features=n_features, min_len=min_len)

            mean = np.average(inside, axis=0)
            xi = inside - mean

            x_main = xi[:, :n_features]
            x_sec = xi[:, n_features:]

            K = np.linalg.lstsq(x_main, x_sec, rcond=None)[0].T

            rest = outside

            Ks.append(K)
            ms.append(mean)
            stds.append(np.std(inside[:, :n_features], axis=0))
            groups.append(inside)

            print(len(inside), len(rest), len(x), min_len)

            # show_common_distributions(rest[:,:4])

        self.K2 = np.stack(Ks, axis=0)
        self.m2 = np.stack(ms, axis=0)
        self.stdf = np.stack(stds, axis=0)

        sigma_near = scipy.special.erfinv(0.7) * np.sqrt(2)
        centers = self.get_centers()
        close_groups = []
        for i in range(centers.shape[0]-1):
            found = False
            for gr in close_groups:
                if i in gr:
                    found = True
                    break
            if found:
                continue
            dist = np.linalg.norm((centers - centers[i]) / (self.stdf + self.stdf[i]), axis=-1)[i:]
            near = (i + np.argwhere(dist <= sigma_near)[:, 0]).tolist()
            close_groups.append(near)
        print(close_groups)

        take_groups = []
        for i, gr in enumerate(close_groups):
            max_gr_i = gr[np.argmax([len(groups[g]) for g in gr])]
            take_groups.append(max_gr_i)

        self.K2 = np.take(self.K2, take_groups, axis=0)
        self.m2 = np.take(self.m2, take_groups, axis=0)
        self.stdf = np.take(self.stdf, take_groups, axis=0)

        print(self.K2.shape)



    def _forward_2_order(self, x, do_matmul=True, train=False, batch_size=8):
        n_features = self.K2.shape[2]
        centers = self.m2[:, :n_features]

        xf_a = x[:,:,:,:n_features]

        xf_all = []
        for i in range(0, len(x), batch_size):
            xf = xf_a[i:i + batch_size]
            xf = tf.expand_dims(xf, axis=-2)
            xf = xf - centers

            dist = tf.linalg.norm(xf / self.stdf, axis=-1)
            closest = tf.argmin(dist, axis=-1)
            xf = self.clip(xf)

            xf = xf + centers
            xf = tf.gather(xf, closest, axis=-2, batch_dims=3)
            xf_all.append(xf)
        xf = tf.concat(xf_all, axis=0)
        return xf

    def _backward_2_order(self, x, batch_size=1):
        n_features = x.shape[-1]
        centers = self.m2[:,:n_features]

        xb_all = []
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            xf = x_batch
            dist = tf.linalg.norm(((tf.expand_dims(xf, axis=-2) - centers) / self.stdf), axis=-1)
            # dist = tf.linalg.norm(((tf.expand_dims(xf, axis=-2) - centers)), axis=-1)
            closest = tf.argmin(dist, axis=-1)

            xf = tf.expand_dims(xf, axis=-2)
            xf = xf - centers
            xf = self.clip(xf)
            xf = tf.expand_dims(xf, axis=-1)
            xb = tf.matmul(self.K2, xf)
            xf = tf.squeeze(xf, axis=-1)
            xb = tf.squeeze(xb, axis=-1)
            xb = tf.concat([xf, xb], axis=-1)
            xb = xb + self.m2
            xb = tf.gather(xb, closest, axis=-2, batch_dims=3)
            # xb = tf.reduce_sum(xb * closest, axis=-2)

            xb_all.append(xb)
        xb = tf.concat(xb_all, axis=0)
        return xb

    def forward(self, x, batch_size=1, do_2=True):
        if self.orient == "both":
            stride = (self.stride, self.stride)
        elif self.orient == "hor":
            stride = (1, self.stride)
        elif self.orient == "ver":
            stride = (self.stride, 1)

        x_conv = []
        for i in range(0, x.shape[0], batch_size):
            batch = x[i: i + batch_size]
            batch = tf.nn.conv2d(
                batch, self.kernel_forward,
                (1, stride[0], stride[1], 1),
                padding="VALID") + self.bias_forward
            batch = batch.numpy()
            x_conv.append(batch)
        x = np.concatenate(x_conv, axis=0)

        if do_2:
            x = self._forward_2_order(x)

        return x

    def clip(self, x, reverse=False, train=False):
        sigma = scipy.special.erfinv(1 - self.clip_loss) * np.sqrt(2)
        # std = self.channel_std
        std = self.stdf
        # std = np.sqrt(np.sum(self.channel_std))

        x = x / (std * sigma)
        # x = x.numpy()

        if reverse:
            # x = tf.atanh(x / sigma) * sigma
            # mul = 10
            # x = tf.tanh(x)
            x = tf.clip_by_value(x, -1, 1)
            pass
        else:
            if train:
                # x = tf.tanh(x / sigma) * sigma
                pass
            else:
                # x = tf.tanh(x)
                # pass
                x = tf.clip_by_value(x, -1, 1)

        # x[np.abs(x) > sigma] = 0
        # x = tf.clip_by_value(x, -sigma, sigma)
        # x[:,:,:,-1] = tf.clip_by_value(x[:,:,:,-1], -sigma, sigma)
        # x[:,:,:,-1] = np.clip(x[:,:,:,-1], -sigma, sigma)
        x = x * (std * sigma)
        return x

    def backward(self, x, batch_size=1):

        x = self._backward_2_order(x)

        if self.orient == "both":
            stride = (self.stride, self.stride)
        elif self.orient == "hor":
            stride = (1, self.stride)
        elif self.orient == "ver":
            stride = (self.stride, 1)

        b, h, w, c = self.input_shape
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
        np.save("saved/K2{}".format(n), self.K2)
        # np.save("saved/K2_T{}".format(n), self.K2_T)
        np.save("saved/m2{}".format(n), self.m2)
        np.save("saved/std{}".format(n), self.channel_std)
        np.save("saved/stdf{}".format(n), self.stdf)
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
        self.K2 = np.load("saved/K2{}.npy".format(n))
        # self.K2_T = np.load("saved/K2_T{}.npy".format(n))
        self.m2 = np.load("saved/m2{}.npy".format(n))
        self.channel_std = np.load("saved/std{}.npy".format(n))
        self.stdf = np.load("saved/stdf{}.npy".format(n))
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
            K = K / np.sqrt((np.count_nonzero(mask) / np.sum(1 / mask)))
        bias_forward = -tf.squeeze(tf.matmul(K, tf.expand_dims(mean_p, axis=-1)))
        K = tf.reshape(K, (V.shape[0], ksize[0], ksize[1], Cin))
        K = tf.transpose(K, (1, 2, 3, 0))

        K_T = U
        if self.equalize:
            K_T = np.matmul(K_T, np.diag(np.sqrt(S / np.average(S)) * std))
        else:
            K_T = K_T * np.sqrt((np.count_nonzero(mask) / np.sum(1 / mask)))
        bias_backward = mean_p
        bias_backward = tf.reshape(bias_backward, (ksize[0], ksize[1], Cin, 1))
        # bias_backward /= np.expand_dims(mask, axis=(-1, -2))
        K_T = tf.reshape(K_T, (ksize[0], ksize[1], Cin, V.shape[0]))
        K_T = tf.transpose(K_T, (0, 1, 2, 3))
        # K_T /= np.expand_dims(mask, axis=(-1, -2))

        return K, K_T, bias_forward, bias_backward