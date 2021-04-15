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

        self.get_2_order_kernel(x)
        print("Slice...")
        x = self._forward_2_order(x, train=True)

        return x

    def _get_peaks(self, x, n_features, min_len, eps):
        if n_features == 0:
            return [np.int32(list(range(len(x))))]

        peaks = []
        ch = x[:, 0]
        m, M = np.min(ch), np.max(ch) + 2 * eps
        checks = np.arange(m, M, eps)
        for i in range(len(checks) - 1):
            where = np.argwhere((ch >= checks[i]) & (ch < checks[i+1]))[:,0]
            subset = x[where]
            if len(subset) <= min_len:
                continue
            subset_peaks = self._get_peaks(subset[:,1:], n_features-1, min_len, eps)
            subset_peaks = [np.take(where, peak) for peak in subset_peaks]
            peaks += subset_peaks
        return peaks

    def get_peaks(self, x, n_features, min_len, max_peaks=400):
        # eps = 0.2

        n_features = x.shape[-1]

        # min_len = int(math.ceil(len(x) / max_peaks))

        # min_dens = 1 / np.max(np.max(x, axis=0) - np.min(x, axis=0))
        # size = np.sqrt(np.sum(np.square(np.std(x, axis=0))))

        eps = 0.5
        # min_len_by_eps = int(len(x) * np.power(eps * min_dens, n_features))
        # if min_len < min_len_by_eps:
        #   min_len = min_len_by_eps
        # else:
        #   eps = np.power(min_len / (len(x)), 1 / n_features) / min_dens

        peaks = self._get_peaks(x, n_features, min_len, eps)

        # peaks = []
        # rest = x
        # while len(rest) > min_len:
        #     print(eps, min_len, len(rest), len(peaks))
        #     peaks_1 = self._get_peaks(np.asarray(rest), n_features, min_len, eps)
        #     if len(peaks_1) > 0:
        #         all_peaks_1 = np.sort(np.concatenate(peaks_1, axis=0))
        #         rest_idx = np.arange(0, len(rest), dtype=np.int32)
        #         rest_idx = list(set(rest_idx) - set(all_peaks_1))
        #         rest = np.take(rest, rest_idx, axis=0)
        #         peaks += peaks_1
        #     eps *= 1.5

        peaks = sorted(peaks, key=lambda x: len(x), reverse=True)
        peaks = [np.take(x, peak, axis=0).copy() for peak in peaks]

        # peaks = peaks[:max_peaks]
        # lens = np.cumsum([len(peak) for peak in peaks])
        # n_peaks = np.sum(lens < frac * len(x))
        # peaks = peaks[:n_peaks]

        return peaks

    def get_peak(self, x, eps=None):
        if eps is None:
            eps = 0.1
        # n_splits = 10

        x = flatten(x)
        rest = x
        outside = []
        for i in range(x.shape[-1]):
            ch = rest[:, i]
            m, M = np.min(ch), np.max(ch) + 2 * eps
            checks = np.arange(m, M, eps)
            density = np.stack([np.sum((ch >= checks[i]) & (ch < checks[i + 1])) for i in range(len(checks) - 1)])
            peak_i = np.argmax(density)
            outside.append(rest[(ch < checks[peak_i]) | (ch >= checks[peak_i + 1])])
            rest = rest[(ch >= checks[peak_i]) & (ch < checks[peak_i + 1])]

        # if len(rest) < min_len:
        peak = np.average(rest, axis=0)

        return peak


    def get_centers(self):
        n_features = self.K2.shape[2]
        centers = self.m2[:,:n_features]
        return centers

    def get_nonzero_cov(self, dataset):
        cov = np.zeros((dataset.shape[-1], dataset.shape[-1]), dtype=np.float32)
        nz = np.abs(dataset) > self.eps
        for i in range(dataset.shape[-1]):
            print(f"\r{i} / {dataset.shape[-1]}", flush=True, end="")
            for j in range(dataset.shape[-1]):
                nonzero = nz[:, i] & nz[:, j]

                if np.sum(nonzero) == 0:
                    cov[i, j] = 0
                    continue

                xy = np.product(dataset[:, [i, j]][nonzero], axis=-1)
                cov[i, j] = np.average(xy)

        print()

        return cov

    def get_2_order_kernel(self, x, odd=0.0001, batch_size=1000):
        x = flatten(x)
        # self.channel_std = np.std(x, axis=0)

        cov = self.get_nonzero_cov(x)
        disp = np.stack([cov[i, i] for i in range(x.shape[-1])])
        cor = cov / np.sqrt(np.outer(disp, disp))
        plt.imshow(np.abs(cor))
        plt.show()

        U, S, V = np.linalg.svd(cov)
        plt.imshow(np.abs(V))
        plt.show()

        # thresh = 0.3
        # rest = x2
        # take_channels = np.arange(x.shape[-1])
        # while len(rest) > (1 - thresh) * len(x):
        #     nz = np.abs(rest) >= self.eps
        #     nzc = np.count_nonzero(nz, axis=0)
        #     drop_channel = np.argmin(nzc)
        #     rest = rest[~nz[:, drop_channel]]
        #     rest = np.delete(rest, drop_channel, axis=-1)
        #     take_channels = np.delete(take_channels, drop_channel, axis=-1)
        #     # plt.plot(np.sort(nzc)[::-1])
        #     # plt.show()

        # thresh = 0.1
        # nz = np.abs(x2) >= self.eps
        # nzc = np.average(nz, axis=0)
        # n_features = np.count_nonzero(nzc >= thresh)
        # take_channels = np.argsort(nzc)[::-1][:n_features]
        #
        # plt.plot(np.sort(nzc)[::-1])
        # plt.plot(np.ones_like(nzc) * thresh)
        # plt.show()

        thresh = 2 * self.eps
        n_features = np.count_nonzero(np.sqrt(S) >= thresh)
        take_channels = np.argsort(np.sqrt(S))[::-1][:n_features]

        plt.plot(np.sqrt(S))
        plt.plot(np.ones_like(S) * thresh)
        plt.show()

        self.n_features = n_features
        self.K2 = np.take(V, take_channels, axis=0)
        x2 = np.matmul(x, self.K2.T)
        print("{} main channels, {} secondary".format(n_features, x.shape[-1] - n_features))

        # nz1 = (np.float32(np.abs(x) >= self.eps)) * 2 - 1
        nz1 = np.square(x)
        # nz2 = (np.float32(np.abs(x2) >= self.eps)) * 2 - 1
        x22 = np.matmul(x2, self.K2)
        nz2 = np.concatenate([np.square(x2), np.square(x22)], axis=-1)
        K0 = np.linalg.lstsq(nz2, nz1, rcond=None)[0].T
        self.K0 = K0

        plt.imshow(np.abs(K0))
        plt.show()

        # self.kernel_forward = np.take(self.kernel_forward, take_channels, axis=-1)
        # self.kernel_backward = np.take(self.kernel_backward, take_channels, axis=-1)
        # self.bias_forward = np.take(self.bias_forward, take_channels, axis=-1)
        # self.channel_std = np.take(std, take_channels, axis=-1)

        # rest = x
        # groups = []
        # Ks = []
        # ms = []
        # stds = []
        # # show_common_distributions(rest[:,:4])
        # while len(rest) > min_len:
        # # for group in groups:
        #
        #     # show_common_distributions(rest[:,:6])
        #
        #     # inside = group
        #     # inside, outside = self.get_peak(rest, n_features=n_features, min_len=2*n_features)
        #     inside = rest
        #     rest = rest[:0]
        #     for i in range(100):
        #         # mean = self.get_peak(inside)
        #         mean = np.average(inside)
        #         xi = inside - mean
        #
        #         cov = np.average(np.matmul(np.expand_dims(xi, axis=-1), np.expand_dims(xi, axis=-2)), axis=0)
        #         U, S, V = np.linalg.svd(cov)
        #         K = V
        #
        #         # plt.imshow(np.abs(V))
        #         # plt.show()
        #
        #         xr = inside - mean
        #         xr = np.matmul(xr, K.T)
        #
        #         xr_sec = xr[:,n_features:]
        #         peak_r = self.get_peak(xr_sec)
        #         xr_sec = xr_sec - peak_r
        #
        #         # show_common_distributions(xr_sec[:,:6])
        #
        #         dist = np.linalg.norm(xr_sec, axis=-1)
        #         cut_dist = np.sort(dist)[int(0.5 * len(dist))]
        #         # plt.plot(np.sort(dist))
        #         # plt.show()
        #
        #         if cut_dist <= self.eps * sigma:
        #             cut_dist = self.eps * sigma
        #
        #         rest = np.concatenate([rest, inside[dist >= cut_dist]], axis=0)
        #         inside = inside[dist < cut_dist]
        #         # inside = rest[dist < cut_dist]
        #         # outside = rest[dist >= cut_dist]
        #
        #
        #
        #         if cut_dist <= self.eps * sigma:
        #             break
        #
        #
        #
        #     # rest = outside
        #
        #     # U, S, V = np.linalg.svd(K, full_matrices=False)
        #     # K = np.matmul(U, V)
        #
        #     # rest = outside
        #
        #     print(len(inside), len(rest), len(x), min_len)
        #
        #     # if len(inside) < min_len:
        #     #     continue
        #
        #     mean = self.get_peak(inside)
        #     xi = inside - mean
        #     x_main = xi[:, :n_features]
        #     # stdf = get_std_by_peak(x_main)
        #     stdf = np.std(x_main, axis=0)
        #     x_sec = xi[:, n_features:]
        #     K = np.linalg.lstsq(x_main, x_sec, rcond=None)[0].T
        #
        #     Ks.append(K)
        #     ms.append(mean)
        #     stds.append(stdf)
        #     # stds.append(np.ones_like(inside[0, :n_features]) * 0.1)
        #     groups.append(inside)
        #
        #     # show_common_distributions(rest[:,:4])
        #
        # self.K2 = np.stack(Ks, axis=0)
        # self.m2 = np.stack(ms, axis=0)
        # self.stdf = np.stack(stds, axis=0)
        #
        # # K_norm = np.float32([np.linalg.norm(K) for K in Ks])
        # # # plt.plot(np.sort(K_norm)[::-1])
        # # take_groups = np.argwhere((K_norm > 0.5) & (K_norm < 5))[:,0]
        # # take_groups = take_groups[np.argsort(K_norm[take_groups])][::-1][:500]
        # # plt.plot(K_norm[take_groups])
        # # plt.show()
        #
        # # sigma_near = scipy.special.erfinv(0.4) * np.sqrt(2)
        # # centers = self.get_centers()
        # # close_groups = []
        # # for i in range(centers.shape[0]):
        # #     found = False
        # #     for gr in close_groups:
        # #         if i in gr:
        # #             found = True
        # #             break
        # #     if found:
        # #         continue
        # #     dist = np.linalg.norm((centers - centers[i]) / (self.stdf + self.stdf[i]), axis=-1)[i:]
        # #     near = (i + np.argwhere(dist <= sigma_near)[:, 0]).tolist()
        # #     if len(near) > 0:
        # #         close_groups.append(near)
        # # print(close_groups)
        # #
        # # take_groups = []
        # # for i, gr in enumerate(close_groups):
        # #     max_gr_i = gr[np.argmax([len(groups[g]) for g in gr])]
        # #     take_groups.append(max_gr_i)
        # #
        # # self.K2 = np.take(self.K2, take_groups, axis=0)
        # # self.m2 = np.take(self.m2, take_groups, axis=0)
        # # self.stdf = np.take(self.stdf, take_groups, axis=0)
        #
        # print(self.K2.shape)



    def _forward_2_order(self, x, do_matmul=True, train=False, batch_size=8):
        # n_features = self.n_features
        # centers = self.m2[:, :n_features]

        # xf_a = x[:,:,:,:n_features]
        xf = tf.squeeze(tf.matmul(self.K2, tf.expand_dims(x, axis=-1)), axis=-1)

        # xf_all = []
        # for i in range(0, len(x), batch_size):
        #     xf = xf_a[i:i + batch_size]
        #     xf = tf.expand_dims(xf, axis=-2)
        #     xf = xf - centers
        #
        #     dist = tf.linalg.norm(xf / self.stdf, axis=-1)
        #     closest = tf.argmin(dist, axis=-1)
        #     xf = self.clip(xf)
        #
        #     xf = xf + centers
        #     xf = tf.gather(xf, closest, axis=-2, batch_dims=3)
        #     xf_all.append(xf)
        # xf = tf.concat(xf_all, axis=0)
        # xf = xf_a
        return xf

    def _backward_2_order(self, x, batch_size=1):
        xb = tf.squeeze(tf.matmul(self.K2.T, tf.expand_dims(x, axis=-1)), axis=-1)
        # nz2 = (tf.cast(tf.abs(x) >= self.eps, tf.float32)) * 2 - 1
        nz2 = tf.concat([tf.square(x), tf.square(xb)], axis=-1)
        nz = tf.squeeze(tf.matmul(self.K0, tf.expand_dims(nz2, axis=-1)), axis=-1)
        nz = tf.cast(nz > 0.5 * np.square(self.eps), tf.float32)
        xb = xb * nz

        # n_features = x.shape[-1]
        # centers = self.m2[:,:n_features]
        #
        # sigma = scipy.special.erfinv(0.9) * np.sqrt(2)
        #
        # xb_all = []
        # for i in range(0, len(x), batch_size):
        #     x_batch = x[i:i+batch_size]
        #     xf = x_batch
        #     # dist = tf.linalg.norm(((tf.expand_dims(xf, axis=-2) - centers) / self.stdf), axis=-1, keepdims=True)
        #     dist = tf.linalg.norm(((tf.expand_dims(xf, axis=-2) - centers) / self.stdf), axis=-1)
        #     # dist = tf.linalg.norm(((tf.expand_dims(xf, axis=-2) - centers)), axis=-1)
        #     # closest = tf.exp(-0.5*tf.square(dist))# + 0.1 / dist.shape[-2]
        #     # closest = tf.exp(-dist)
        #     # closest = closest / tf.reduce_sum(closest, axis=-2, keepdims=True)
        #     hit = tf.cast(tf.reduce_any(dist < sigma, axis=-1), tf.float32)
        #     closest = tf.argmin(dist, axis=-1)
        #
        #     xfnh = x_batch[hit == 0,:]
        #     # show_common_distributions(xfnh)
        #
        #     # plt.imshow(hit[0])
        #     # plt.show()
        #
        #     xf = tf.expand_dims(xf, axis=-2)
        #     xf = xf - centers
        #     xf = self.clip(xf)
        #     xf = tf.expand_dims(xf, axis=-1)
        #     xb = tf.matmul(self.K2, xf)
        #     xf = tf.squeeze(xf, axis=-1)
        #     xb = tf.squeeze(xb, axis=-1)
        #
        #     xb = self.clip(xb, reverse=True, back=True)
        #
        #     xf = xf + centers
        #     xb = xb + self.m2[:,n_features:]
        #
        #     xb = xb * tf.expand_dims(tf.expand_dims(hit, axis=-1), axis=-1)
        #
        #     xb = tf.concat([xf, xb], axis=-1)
        #
        #     xb = tf.gather(xb, closest, axis=-2, batch_dims=3)
        #     # xb = tf.reduce_sum(xb * closest, axis=-2)
        #
        #
        #
        #     xb_all.append(xb)
        # xb = tf.concat(xb_all, axis=0)
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

    def clip(self, x, reverse=False, train=False, back=False):
        sigma = scipy.special.erfinv(1 - self.clip_loss) * np.sqrt(2)
        if back:
            # std = self.channel_std
            # std = np.sqrt(np.sum(self.channel_std))
            std = self.eps * sigma
        else:
            std = self.stdf * sigma
        # std = self.eps * 5

        x = x / std
        # x = x.numpy()

        if reverse:
            # x = tf.tanh(x)
            # x = tf.atanh(x / sigma) * sigma
            # mul = 10
            # if back:
            #     # x = tf.tanh(x)
            #     x = tf.clip_by_value(x, -1, 1)
            # else:
            #     x = tf.clip_by_value(x, -1, 1)
            pass
        else:
            # x = tf.tanh(x)
            if train:
                # x = tf.clip_by_value(x, -1, 1)
                pass
            else:
                # x = tf.tanh(x)
                pass
                # x = tf.clip_by_value(x, -1, 1)

        # x[np.abs(x) > sigma] = 0
        # x = tf.clip_by_value(x, -sigma, sigma)
        # x[:,:,:,-1] = tf.clip_by_value(x[:,:,:,-1], -sigma, sigma)
        # x[:,:,:,-1] = np.clip(x[:,:,:,-1], -sigma, sigma)
        x = x * std
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
        np.save("saved/K0{}".format(n), self.K0)
        # np.save("saved/K2_T{}".format(n), self.K2_T)
        # np.save("saved/m2{}".format(n), self.m2)
        # np.save("saved/std{}".format(n), self.channel_std)
        # np.save("saved/stdf{}".format(n), self.stdf)
        np.save("saved/in{}".format(n), self.input_shape)
        np.save("saved/nc{}".format(n), self.n_features)
        # np.save("saved/xs{}".format(n), self.xs)
        # np.save("saved/ys{}".format(n), self.ys)

    def load(self, n):
        self.kernel_forward = np.load("saved/K{}.npy".format(n))
        self.bias_forward = np.load("saved/b{}.npy".format(n))
        self.kernel_backward = np.load("saved/K_T{}.npy".format(n))
        self.bias_backward = np.load("saved/b_T{}.npy".format(n))
        # self.Km = np.load("saved/Km{}.npy".format(n))
        self.K2 = np.load("saved/K2{}.npy".format(n))
        self.K0 = np.load("saved/K0{}.npy".format(n))
        # self.K2_T = np.load("saved/K2_T{}.npy".format(n))
        # self.m2 = np.load("saved/m2{}.npy".format(n))
        # self.channel_std = np.load("saved/std{}.npy".format(n))
        # self.stdf = np.load("saved/stdf{}.npy".format(n))
        self.input_shape = np.load("saved/in{}.npy".format(n))
        self.n_features = np.load("saved/nc{}.npy".format(n))
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