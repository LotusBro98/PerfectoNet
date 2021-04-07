import math

import tensorflow as tf
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from distribution import *

class Layer():
    def __init__(self, ksize=3, stride=2, orient="both", eps=1e-2, distribution_approx_n=100, channels=None, equalize=False, clip_loss=0.05):
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
        # x = self.gather(x)

        # print("Counting std ...")
        # peak_std = get_std_by_peak(x)
        # print(x.shape)
        # x_in_peak = x[(np.abs(x) < peak_std).all(axis=-1)]
        # print(x_in_peak.shape)
        # peak_std = np.std(x, axis=(0,1,2))
        # plt.plot(np.log(peak_std))
        # plt.plot(np.ones_like(peak_std) * np.log(self.eps))
        # plt.show()
        # take_channels = np.argsort(peak_std)[::-1]
        # collect_std = np.sqrt(np.cumsum(np.square(np.sort(peak_std))))
        # n_features = np.count_nonzero(collect_std >= self.eps) if self.channels is None else self.channels

        # n_features = np.count_nonzero(peak_std >= self.eps)
        # n_features = len(peak_std)
        # take_channels = take_channels[:n_features]

        # self.bias_forward = np.take(self.bias_forward, take_channels, axis=-1)
        # self.kernel_forward = np.take(self.kernel_forward, take_channels, axis=-1)
        # self.kernel_backward = np.take(self.kernel_backward, take_channels, axis=-1)

        # self.n_features = n_features
        # self.channel_std = np.take(peak_std, take_channels, axis=-1)
        # self.channel_std = self.channel_std[:n_features]

        # x = tf.gather(x, take_channels, axis=-1)
        # x = tf.take(x, take_channels, axis=-1)

        self.get_2_order_kernel(x)
        # x = x[:, :, :, :n_features]
        print("Slice...")
        x = self._forward_2_order(x, show_plow=True)

        # show_common_distributions(x)

        self.kernel_backward /= np.expand_dims(self.get_mask(), axis=(-1, -2))
        self.bias_backward /= np.expand_dims(self.get_mask(), axis=(-1, -2))

        return x


    def get_cov_svd(self, xp, eps=None, use_cor=False, batch_size_cells=100000000):
        xp = flatten(xp)

        mean = tf.reduce_mean(xp, axis=0)
        std = tf.math.reduce_std(xp, axis=0)

        N = 0
        cov = 0
        batch_size = int(math.ceil(batch_size_cells / (xp.shape[-1] * xp.shape[-1])))
        for i in range(0, xp.shape[0], batch_size):
            batch = xp[i: i + batch_size]
            batch = batch - mean

            if use_cor:
                batch = tf.pow(tf.abs(batch), 2)# * tf.sign(batch)

            batch_cov = tf.matmul(tf.expand_dims(batch, axis=-1), tf.expand_dims(batch, axis=-2))
            batch_cov = tf.reduce_sum(batch_cov, axis=0)

            N += batch.shape[0]
            cov += batch_cov
        cov = cov / N
        cor = cov / np.outer(std, std)

        # if use_cor:
        #   U, S, V = np.linalg.svd(cor)
        # else:
        U, S, V = np.linalg.svd(cov)

        if eps is None:
            return cor

        # print("Counting peak std...")
        xp1 = np.matmul(V, np.expand_dims(xp - mean, axis=-1)).squeeze(-1)
        peak_std = get_std_by_peak(xp1)
        # print(peak_std)
        # peak_std = np.stack([np.std(xp1[np.abs(xp1[:,i]) < peak_std[i]]) for i in range(xp1.shape[-1])], axis=-1)
        # print(peak_std)
        # peak_std = np.sqrt(S)
        take_channels = np.argsort(peak_std)[::-1]

        n_features = np.count_nonzero(peak_std >= eps)
        n_features = max(1, n_features)

        # if eps is not None and eps > 0:
        #     plt.plot(np.log(np.sort(peak_std)[::-1]))
        #     # plt.plot(np.log(collect_std[::-1]))
        #     plt.plot(np.log(np.ones_like(S) * eps))
        #     plt.show()

        K = np.take(V, take_channels, axis=0)[:n_features]
        return K, mean

    def get_outliers(self, x, prob_check=0.1, prob_take=0.999, frac=0.01):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        x = flatten(x)
        peak_std = get_std_by_peak(x, prob_check)
        scale = scipy.special.erfinv(prob_take) * np.sqrt(2)
        # peak_std = np.std(x, axis=0)
        # if frac is not None:
        #   norm = np.linalg.norm(x / peak_std, axis=-1)# / x.shape[-1]
        #   idx = np.argsort(norm)
        #   div = int(frac * len(idx))
        #   inside = x[idx[:div], :]
        #   outside = x[idx[div:], :]
        # else:
        outside_all = []
        rest = x
        for i in range(x.shape[-1]-1, 0, -1):
        # for i in range(x.shape[-1]):
          inside = rest[np.abs(rest[:,i]) < peak_std[i] * scale,:]
          outside = rest[np.abs(rest[:,i]) >= peak_std[i] * scale,:]
          rest = inside
          outside_all.append(outside)
          if len(inside) < frac * len(x):
            break
        outside = np.concatenate(outside_all, axis=0)
        # peak_std = np.std(x, axis=0)
        # rest = x
        # clusters = []
        # for i in range(x.shape[-1]):
        #     xi = rest[np.abs(rest[:,i]) > peak_std[i],:]
        #     xr = rest[np.abs(rest[:,i]) <= peak_std[i],:]
        #     clusters.append(xi)
        #     rest = xr
        #     print(xi.shape, xr.shape)
        # clusters.append(rest)
        # clusters = np.concatenate(clusters, axis=0)
        return inside, outside

    def get_2_order_kernel(self, x, odd=0.01, batch_size=16):
        # x = self.normalize(x)
        x = flatten(x)
        self.channel_std = np.std(x, axis=0)
        min_len = int(odd * len(x))
        # min_len = max(min_len, x.shape[-1])
        # x = self.clip(x)
        rest = x
        groups = []
        # show_common_distributions(rest[:,:8])
        while len(rest) > min_len:
            rest1 = rest
            for i in range(5):
              K, mean = self.get_cov_svd(rest1, eps=0, use_cor=True)
              xi = matmul(K, rest - mean)
              inside, outside = self.get_outliers(xi, frac=(min_len / len(xi)))
              inside = matmul(K.T, inside) + mean
              outside = matmul(K.T, outside) + mean
              rest1 = inside
            # xi = matmul(K, rest - mean)
            # inside, outside = self.get_outliers(xi)
            # inside = matmul(K.T, inside) + mean
            # outside = matmul(K.T, outside) + mean

            groups.append(inside)
            rest = outside

            print(len(inside), len(rest), len(x))

            # show_common_distributions(rest[:,:8])

            # if len(inside) < odd * len(x):
            #     break
        groups.append(rest)

        Ks = []
        ms = []
        for group in groups:
            # if len(group) < odd * len(x):
              # continue
            K, mean = self.get_cov_svd(group, eps=self.eps, use_cor=False)
            # xi = matmul(K, group - mean)
            # inside, outside = self.get_outliers(xi, prob=0.68, frac=0.3)
            # if len(inside) == 0:
            #   continue
            # inside = matmul(K.T, inside) + mean
            # outside = matmul(K.T, outside) + mean
            # K, mean = self.get_cov_svd(inside, eps=self.eps)
            Ks.append(K)
            ms.append(mean)
            # plt.imshow(np.abs(K))
            # plt.show()

        print([K.shape[0] for K in Ks])

        lens = [K.shape[0] for K in Ks]
        maxlen = int(np.average(lens))

        Ks = [np.pad(K[:maxlen], pad_width=((0, maxlen - K[:maxlen].shape[0]), (0,0))) for K in Ks]
        self.K2 = np.stack(Ks, axis=0)
        self.m2 = np.stack(ms, axis=0)
        print(self.K2.shape)

        std = 0
        N = 0
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            x_batch = tf.expand_dims(x_batch, axis=-2) - self.m2
            xf = tf.squeeze(tf.matmul(self.K2, tf.expand_dims(x_batch, axis=-1)), axis=-1)
            std += tf.reduce_sum(tf.square(xf), axis=0)
            N += len(xf)
        std = np.sqrt(std / N)

        idx = np.argsort(std, axis=1)[:,::-1]
        self.K2 = tf.gather(self.K2, idx, axis=1, batch_dims=1)
        std = tf.gather(std, idx, axis=1, batch_dims=1).numpy()

        std[std == 0] = np.min(std[std != 0])
        avg_std = tf.exp(tf.reduce_mean(tf.math.log(std), axis=0, keepdims=True))
        coeff = tf.expand_dims(avg_std / std, axis=-1)
        self.K2_T = tf.transpose(self.K2 / coeff, (0, 2, 1))
        self.K2 = self.K2 * coeff




    def _forward_2_order(self, x, do_matmul=True, show_plow=False, batch_size=16):
        n_groups = self.K2.shape[0]
        # x = self.clip(x)

        xf_all = []
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            # x_batch = self.normalize(x_batch)
            x_batch = tf.expand_dims(x_batch, axis=-2) - self.m2
            xf = tf.matmul(self.K2, tf.expand_dims(x_batch, axis=-1))
            xb = tf.matmul(self.K2_T, xf)
            xf = tf.squeeze(xf, axis=-1)
            xb = tf.squeeze(xb, axis=-1)
            closest = tf.argmin(tf.linalg.norm(x_batch - xb, axis=-1), axis=-1)
            xf = tf.gather(xf, closest, axis=-2, batch_dims=3)
            # closest = tf.one_hot(closest, n_groups)
            closest = tf.cast(tf.expand_dims(closest, axis=-1), dtype=tf.float32)
            mul = np.sqrt(np.sum(np.square(self.channel_std)))
            closest = closest * mul
            xf = tf.concat([xf, closest], axis=-1)
            xf_all.append(xf)
        xf = tf.concat(xf_all, axis=0)
        return xf

    def _backward_2_order(self, x, batch_size=16):
        xb_all = []
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            n_groups = self.K2.shape[0]
            # closest = x_batch[:,:,:,-n_groups:]
            mul = np.sqrt(np.sum(np.square(self.channel_std)))
            closest = x_batch[:,:,:,-1] / mul
            # closest = tf.argmax(closest, axis=-1)
            closest = tf.cast(tf.round(closest), dtype=tf.int32)
            # xf = x_batch[:,:,:,:-n_groups]
            xf = x_batch[:,:,:,:-1]
            xf = tf.expand_dims(xf, axis=-2)
            xf = tf.expand_dims(xf, axis=-1)
            xb = tf.matmul(self.K2_T, xf)
            xb = tf.squeeze(xb, axis=-1)
            xb = xb + self.m2
            xb = tf.gather(xb, closest, axis=-2, batch_dims=3)
            # xb = self.clip(xb)
            # xb = self.denormalize(xb)

            xb_all.append(xb)
        xb = tf.concat(xb_all, axis=0)
        return xb

    def calc_minor_matrix(self, x, std, n_features):
        minor = x[:,:,:,n_features:]
        minor = np.reshape(minor, (minor.shape[0] * minor.shape[1] * minor.shape[2], minor.shape[3]))
        minor = minor / std[n_features:]
        major = x[:,:,:,:n_features]
        major = np.reshape(major, (major.shape[0] * major.shape[1] * major.shape[2], major.shape[3]))
        major = major / std[:n_features]
        Km = np.linalg.lstsq(major, minor, rcond=None)[0].T
        print(np.linalg.norm(Km))
        Km = np.matmul(Km, np.diag(1 / std[:n_features]))
        Km = np.matmul(np.diag(std[n_features:]), Km)
        Km = np.matmul(self.kernel_backward[:, :, :, n_features:], Km)
        return Km

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

    def normalize(self, x):
        norm = tf.linalg.norm(x, axis=-1, keepdims=True)
        x = x / norm
        x = tf.concat([norm, x], axis=-1)
        return x

    def denormalize(self, x):
        norm = x[:,:,:,:1]
        x = x[:,:,:,1:] * norm
        return x

    def clip(self, x):
        sigma = scipy.special.erfinv(1 - self.clip_loss) * np.sqrt(2)
        # std = self.channel_std
        std = np.sqrt(np.sum(self.channel_std))

        x = x / std
        # x = x.numpy()
        x = tf.clip_by_value(x, -sigma, sigma)
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
        np.save("saved/m2{}".format(n), self.m2)
        np.save("saved/std{}".format(n), self.channel_std)
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
        self.m2 = np.load("saved/m2{}.npy".format(n))
        self.channel_std = np.load("saved/std{}.npy".format(n))
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
            K = K / (np.count_nonzero(mask) / np.sum(1 / mask))
        bias_forward = -tf.squeeze(tf.matmul(K, tf.expand_dims(mean_p, axis=-1)))
        K = tf.reshape(K, (V.shape[0], ksize[0], ksize[1], Cin))
        K = tf.transpose(K, (1, 2, 3, 0))

        K_T = U
        if self.equalize:
            K_T = np.matmul(K_T, np.diag(np.sqrt(S / np.average(S)) * std))
        else:
            K_T = K_T * (np.count_nonzero(mask) / np.sum(1 / mask))
        bias_backward = mean_p
        bias_backward = tf.reshape(bias_backward, (ksize[0], ksize[1], Cin, 1))
        # bias_backward /= np.expand_dims(mask, axis=(-1, -2))
        K_T = tf.reshape(K_T, (ksize[0], ksize[1], Cin, V.shape[0]))
        K_T = tf.transpose(K_T, (0, 1, 2, 3))
        # K_T /= np.expand_dims(mask, axis=(-1, -2))

        return K, K_T, bias_forward, bias_backward