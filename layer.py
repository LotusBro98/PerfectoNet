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
        x = dataset

        self.input_shape = x.shape

        K, K_T, bias_forward, bias_backward = self._get_optimal_conv_kernel(x)
        self.kernel_forward = K
        self.kernel_backward = K_T
        self.bias_forward = bias_forward
        self.bias_backward = bias_backward

        print("Processing dataset ...")
        x = self.forward(x, batch_size, do_remap=False)
        # x = self.gather(x)

        # print("Counting std ...")
        # peak_std = get_std_by_peak(x)
        # take_channels = np.argsort(peak_std)[::-1]
        # peak_std = np.std(x, axis=(0,1,2))
        # collect_std = np.sqrt(np.cumsum(np.square(np.sort(peak_std))))
        # n_features = np.count_nonzero(collect_std >= self.eps) if self.channels is None else self.channels

        # self.bias_forward = np.take(self.bias_forward, take_channels, axis=-1)
        # self.kernel_forward = np.take(self.kernel_forward, take_channels, axis=-1)
        # self.kernel_backward = np.take(self.kernel_backward, take_channels, axis=-1)

        # self.n_features = n_features
        # self.channel_std = np.take(peak_std, take_channels, axis=-1)
        # self.channel_std = self.channel_std[:n_features]

        # x = tf.gather(x, take_channels, axis=-1)
        # x = tf.take(x, take_channels, axis=-1)


        self.get_2_order_kernel(x, self.eps)
        # x = x[:, :, :, :n_features]
        print("Polar coordinates...")
        x = self._forward_2_order(x, show_plow=True)
        # x = self.clip(x)


        # show_common_distributions(x)

        # show_distribution(x)

        # Km = self.calc_minor_matrix(x, peak_std, n_features)

        # if n_features < x.shape[-1]:
        #     default_after = get_most_probable_values(x[:,:,:,n_features:])
        #     default_after = np.concatenate([np.zeros((n_features,)), default_after], axis=0)
        # else:
        #     default_after = np.zeros((x.shape[-1],))
        #
        # default_before = np.matmul(self.kernel_backward, default_after)
        # self.bias_backward += np.expand_dims(default_before, axis=-1)

        # self.bias_forward = self.bias_forward[:n_features]
        # self.kernel_forward = self.kernel_forward[:, :, :, :n_features]

        # Km = np.matmul(self.kernel_backward[:, :, :, n_features:], self.Km)
        # self.kernel_backward = self.kernel_backward[:, :, :, :n_features]
        # self.kernel_backward = np.concatenate([self.kernel_backward, np.zeros_like(self.kernel_backward)], axis=-1)
        # self.kernel_backward += Km

        self.kernel_backward /= np.expand_dims(self.get_mask(), axis=(-1, -2))
        self.bias_backward /= np.expand_dims(self.get_mask(), axis=(-1, -2))

        return x


    def get_2_order_kernel(self, x, eps, batch_size_cells=100000000):
        self.channel_std = np.std(x, axis=(0,1,2))

        xp = self._forward_2_order(x, matmul=False)

        xp = np.reshape(xp, (xp.shape[0] * xp.shape[1] * xp.shape[2], xp.shape[3]))

        mean = np.average(xp, axis=0)

        N = 0
        cov = 0
        batch_size = int(math.ceil(batch_size_cells / (xp.shape[-1] * xp.shape[-1])))
        for i in range(0, xp.shape[0], batch_size):
            batch = xp[i: i + batch_size]
            batch = batch - mean

            batch_cov = tf.matmul(tf.expand_dims(batch, axis=-1), tf.expand_dims(batch, axis=-2))
            batch_cov = tf.reduce_sum(batch_cov, axis=0)

            N += batch.shape[0]
            cov += batch_cov
        cov = cov / N

        U, S, V = np.linalg.svd(cov)
        plt.imshow(np.abs(V))
        plt.show()

        # mixed = np.argwhere(np.max(np.abs(V[:,:-1]), axis=0) < 0.999)[:,0]
        # V1 = np.take(V, mixed+1, axis=0)
        # V1 = np.take(V1, mixed, axis=1)


        print("Counting peak std...")
        # xp1 = np.matmul(V, np.expand_dims(xp - mean, axis=-1)).squeeze(-1)
        # peak_std = get_std_by_peak(xp1)
        # print(peak_std)
        # peak_std = np.stack([np.std(xp1[np.abs(xp1[:,i]) < peak_std[i]]) for i in range(xp1.shape[-1])], axis=-1)
        # print(peak_std)
        peak_std = np.sqrt(S)
        take_channels = np.argsort(peak_std)[::-1]
        # print(take_channels)

        # collect_std = np.sqrt(np.cumsum(np.sort(np.square(peak_std))))
        # collect_std /= collect_std[-1]
        # n_features = np.count_nonzero(collect_std >= eps)
        n_features = np.count_nonzero(peak_std >= eps)

        plt.plot(np.log(np.sort(peak_std)[::-1]))
        # plt.plot(np.log(collect_std[::-1]))
        plt.plot(np.log(np.ones_like(S) * eps))
        plt.show()

        self.K2 = np.take(V, take_channels, axis=0)[:n_features]
        # self.K2 = V[:n_features]
        self.m2 = mean

        # s = np.sign(x)
        # x2 = np.matmul(self.K2, np.expand_dims(x2 - self.m2, axis=-1)).squeeze(-1)
        # Ks = np.linalg.lstsq(x2, s, rcond=None)[0].T
        # self.Ks = Ks

        # print(self.K2.shape)

    def _forward_2_order(self, x, matmul=True, show_plow=False):
        # x = self.clip(x)
        x = x / self.channel_std

        xp = [tf.math.asin(x[:,:,:,i] / tf.linalg.norm(x[:,:,:,i:], axis=-1)) for i in range(0, x.shape[-1]-1)]
        xp = xp + [tf.linalg.norm(x, axis=-1) * tf.sign(x[:,:,:,-1])]
        xp = tf.stack(xp, axis=-1)

        xp = tf.concat([
            xp[:, :, :, :-1] * self.channel_std[:-1],
            xp[:, :, :, -1:] * np.sqrt(np.sum(np.square(self.channel_std))),
        ], axis=-1)

        if matmul:
            xp = xp - self.m2

        # xp = self.clip(xp)

        if matmul:
            if show_plow:
                # show_common_distributions(xp.numpy())
                # xp2 = tf.squeeze(tf.matmul(self.K2, tf.expand_dims(xp, axis=-1)), axis=-1)
                # show_common_distributions(xp2.numpy())
                # xp2 = tf.squeeze(tf.matmul(self.K2.T, tf.expand_dims(xp2, axis=-1)), axis=-1) + self.m2
                # show_common_distributions(xp2.numpy())
                # show_distribution(xp)
                pass

            xp = tf.squeeze(tf.matmul(self.K2, tf.expand_dims(xp, axis=-1)), axis=-1)

        return xp

    def _backward_2_order(self, x):
        x = tf.squeeze(tf.matmul(self.K2.T, tf.expand_dims(x, axis=-1)), axis=-1)

        x = x + self.m2

        x = tf.concat([
            x[:, :, :, :-1] / self.channel_std[:-1],
            x[:, :, :, -1:] / np.sqrt(np.sum(np.square(self.channel_std))),
        ], axis=-1)

        # r = self.channel_std#1#x.shape[-1]#x[:,:,:,-1]
        r = x[:,:,:,-1]
        f = x[:,:,:,:-1]
        xd = [tf.abs(r) * tf.sin(f[:,:,:,i]) * tf.reduce_prod(tf.abs(tf.cos(f[:,:,:,:i])), axis=-1) for i in range(0, f.shape[-1])]
        xd = xd + [r * tf.reduce_prod(tf.abs(tf.cos(f[:,:,:,:])), axis=-1)]
        xd = tf.stack(xd, axis=-1)

        xd = xd * self.channel_std
        # xd = self.clip(xd)

        # x = self.clip(x)
        # x2 = tf.concat([x, np.square(x)], axis=-1)
        # x2 = x2 - self.m2
        # xm = np.matmul(self.Km, np.expand_dims(x2, axis=-1)).squeeze(-1)
        # x = np.concatenate([x, xm], axis=-1)
        # x = np.square(x) * np.sign(x)
        # x = np.matmul(self.K2.T, np.expand_dims(x, axis=-1)).squeeze(-1) + self.m2
        # x = np.sqrt(np.abs(x)) * np.sign(x)
        # x = self.unclip(x)
        # x = x2
        return xd

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

    def forward(self, x, batch_size=1, do_remap=True):
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

        if hasattr(self, "K2"):
            x = self._forward_2_order(x)

        # if hasattr(self, "channel_std") and self.channel_std is not None:
        #     x = self.clip(x)

        # if do_remap:
        #     x = remap_distribution(x, self.xs, self.ys)

        return x

    # def unclip(self, x):
    #     sigma = scipy.special.erfinv(1 - self.clip_loss) * np.sqrt(2)
    #     std = self.channel_std
    #
    #     x = np.clip(x, -sigma, sigma)
    #     x = x * std
    #     return x

    def clip(self, x):
        sigma = scipy.special.erfinv(1 - self.clip_loss) * np.sqrt(2)
        std = self.channel_std
        # std = np.sqrt(np.sum(self.channel_std))

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

        mask = self.get_mask()

        K = V
        if self.equalize:
            K = np.matmul(np.diag(1/np.sqrt(S / np.average(S)) / std), K)
        # else:
        #     K = K / (np.count_nonzero(mask) / np.sum(1 / mask))
        bias_forward = -tf.squeeze(tf.matmul(K, tf.expand_dims(mean_p, axis=-1)))
        K = tf.reshape(K, (V.shape[0], ksize[0], ksize[1], Cin))
        K = tf.transpose(K, (1, 2, 3, 0))

        K_T = tf.transpose(V)
        if self.equalize:
            K_T = np.matmul(K_T, np.diag(np.sqrt(S / np.average(S)) * std))
        # else:
        #     K_T = K_T * (np.count_nonzero(mask) / np.sum(1 / mask))
        bias_backward = mean_p
        bias_backward = tf.reshape(bias_backward, (ksize[0], ksize[1], Cin, 1))
        # bias_backward /= np.expand_dims(mask, axis=(-1, -2))
        K_T = tf.reshape(K_T, (ksize[0], ksize[1], Cin, V.shape[0]))
        K_T = tf.transpose(K_T, (0, 1, 2, 3))
        # K_T /= np.expand_dims(mask, axis=(-1, -2))

        return K, K_T, bias_forward, bias_backward