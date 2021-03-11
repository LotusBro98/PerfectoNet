import math

import tensorflow as tf
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

def show_distribution(dataset):
    dataset = np.reshape(dataset, [dataset.shape[0] * dataset.shape[1] * dataset.shape[2], dataset.shape[3]])
    dataset = np.sort(dataset, axis=0)
    dataset /= np.std(dataset, axis=0)
    shift = int(len(dataset) * 1e-1)
    density = 1 / (dataset[shift:] - dataset[:-shift])

    for slice in density.T:
        plt.plot(slice)
    plt.show()

class Layer():
    def __init__(self, ksize=3, stride=2, eps=0.1, distribution_approx_n=100, channels=None, allow_reduce=True):
        self.ksize = ksize
        self.stride = stride
        self.eps = eps
        self.channels = channels
        self.distribution_approx_n = distribution_approx_n
        self.allow_reduce=allow_reduce
        pass

    def get_mask(self):
        if self.stride == 1:
            mask = np.ones((self.ksize, self.ksize)) * (self.ksize * self.ksize)
        elif self.stride == 2:
            if self.ksize == 1:
                mask = np.float32([[1]])
            elif self.ksize == 3:
                mask = np.float32([
                    [4, 2, 4],
                    [2, 1, 2],
                    [4, 2, 4]
                ])
        return mask

    def fit(self, dataset, batch_size=1):
        K, K_T, bias_forward, bias_backward, n_features = self._get_optimal_conv_kernel(dataset)
        self.kernel_forward = K
        self.kernel_backward = K_T
        self.bias_forward = bias_forward
        self.bias_backward = bias_backward

        x = self.forward(dataset, batch_size, do_remap=False)

        default_after = self.get_most_probable_values(x[:,:,:,n_features:])
        default_after = np.concatenate([np.zeros((n_features,)), default_after], axis=0)

        default_before = np.matmul(self.kernel_backward, default_after)

        self.bias_backward = np.expand_dims(self.bias_backward, axis=-1) / np.expand_dims(self.get_mask(), axis=(-1,-2))
        self.bias_backward += np.expand_dims(default_before, axis=-1) / np.expand_dims(self.get_mask(), axis=(-1,-2))
        self.bias_forward = self.bias_forward[:n_features]
        self.kernel_forward = self.kernel_forward[:,:,:,:n_features]
        self.kernel_backward = self.kernel_backward[:,:,:,:n_features]

        x = self.forward(dataset, batch_size, do_remap=False)

        xs, ys = self.build_distribution(x, self.distribution_approx_n)
        self.xs = xs
        self.ys = ys

        self.input_shape = dataset.shape

    def forward(self, x, batch_size=1, do_remap=True):
        x_conv = []
        for i in range(0, x.shape[0], batch_size):
            batch = x[i: i + batch_size]
            batch = tf.nn.conv2d(batch, self.kernel_forward, (1, self.stride, self.stride, 1), padding="VALID") + self.bias_forward
            batch = batch.numpy()
            x_conv.append(batch)
        x = np.concatenate(x_conv, axis=0)
        if do_remap:
            x = self.remap_distribution(x, self.xs, self.ys)
        return x

    def backward(self, x, batch_size=1):
        x = self.remap_distribution(x, self.ys, self.xs)

        b, h, w, c = self.input_shape
        x_conv = []
        for i in range(0, x.shape[0], batch_size):
            batch0 = x[i: i + batch_size]
            print(self.kernel_backward.shape)
            print(self.bias_backward.shape)
            batch = tf.nn.conv2d_transpose(batch0, self.kernel_backward, (batch0.shape[0], h, w, c), (1, self.stride, self.stride, 1), padding="VALID")
            bias = tf.nn.conv2d_transpose(tf.ones_like(batch0[:,:,:,:1]), self.bias_backward, (batch0.shape[0], h, w, c), (1, self.stride, self.stride, 1), padding="VALID")
            batch += bias
            batch = batch.numpy()
            x_conv.append(batch)
        x = np.concatenate(x_conv, axis=0)
        if self.ksize == 3 and self.stride == 2:
            x[:, 0, :, :] *= 2
            x[:, -1, :, :] *= 2
            x[:, :, 0, :] *= 2
            x[:, :, -1, :] *= 2
        # x += self.bias_backward
        return x

    def _get_optimal_conv_kernel(self, dataset, batch_size_cells=100000000):
        assert self.ksize in [1, 3]
        assert self.stride in [1, 2]

        Cin = dataset.shape[-1]
        cells = dataset.shape[1] * dataset.shape[2] * np.square(dataset.shape[3])
        batch_size_images = math.ceil(batch_size_cells / (cells))

        N = 0
        mean_p = 0
        for i in range(0, dataset.shape[0], batch_size_images):
            batch = dataset[i: i + batch_size_images]
            patches = tf.image.extract_patches(batch, [1, self.ksize, self.ksize, 1], [1, self.stride, self.stride, 1], [1, 1, 1, 1],
                                               padding='VALID')
            patches = tf.reshape(patches, [patches.shape[0] * patches.shape[1] * patches.shape[2], patches.shape[3]])

            mean_p += tf.reduce_sum(patches, axis=0)
            N += patches.shape[0]
        mean_p /= N

        N = 0
        M = 0
        for i in range(0, dataset.shape[0], batch_size_images):
            batch = dataset[i: i + batch_size_images]
            patches = tf.image.extract_patches(batch, [1, self.ksize, self.ksize, 1], [1, self.stride, self.stride, 1], [1, 1, 1, 1],
                                               padding='VALID')
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
        plt.plot(Ss)
        plt.plot(np.ones_like(Ss) * thresh)
        plt.show()
        n_features = (tf.math.count_nonzero(Ss > thresh)) if self.channels is None else self.channels
        if (not self.allow_reduce) and n_features < Cin:
            n_features = Cin
        if n_features >= len(S):
            n_features = len(S)

        # V = V[:n_features]

        K = V
        bias_forward = -tf.squeeze(tf.matmul(K, tf.expand_dims(mean_p, axis=-1)))
        # K = tf.reshape(K, (n_features, self.ksize, self.ksize, Cin))
        K = tf.reshape(K, (V.shape[0], self.ksize, self.ksize, Cin))
        K = tf.transpose(K, (1, 2, 3, 0))

        K_T = tf.transpose(V)
        bias_backward = mean_p
        bias_backward = tf.reshape(bias_backward, (self.ksize * self.ksize, Cin))
        bias_backward = tf.reduce_mean(bias_backward, axis=0)
        # K_T = tf.reshape(K_T, (self.ksize, self.ksize, Cin, n_features))
        K_T = tf.reshape(K_T, (self.ksize, self.ksize, Cin, V.shape[0]))
        K_T = tf.transpose(K_T, (0, 1, 2, 3))
        mask = self.get_mask()
        K_T /= np.expand_dims(mask, axis=(-1, -2))

        return K, K_T, bias_forward, bias_backward, n_features

    @staticmethod
    def build_distribution(dataset, approx_n=100, equal=False, weights=None):
        dataset = np.reshape(dataset, [dataset.shape[0] * dataset.shape[1] * dataset.shape[2], dataset.shape[3]])
        dataset = np.sort(dataset, axis=0)

        ys = np.linspace(1/approx_n, 1-1/approx_n, approx_n)
        inds = np.int32(ys * len(dataset))
        inds[0] = 0
        inds[-1] = len(dataset) - 1
        xs = dataset[inds, :].copy()
        ys = np.repeat(np.expand_dims(ys, axis=-1), xs.shape[-1], axis=-1)

        ys = ys * 2 - 1

        ys = scipy.special.erfinv(ys) * np.sqrt(2)
        if not equal:
            if weights is None:
                std = np.std(dataset, axis=0)
                # ys *= (std / np.sqrt(np.sum(np.square(std))))
                ys *= std
            else:
                ys *= weights

        xs = xs[1:-1]
        ys = ys[1:-1]

        # for c in range(xs.shape[-1]):
        #     plt.plot(xs[:,c], ys[:,c])
        # plt.show()

        return xs, ys

    def get_most_probable_values(self, dataset, step=5e-2):
        dataset = np.reshape(dataset, [dataset.shape[0] * dataset.shape[1] * dataset.shape[2], dataset.shape[3]])
        argsort = np.argsort(dataset, axis=0)
        dataset_sorted = np.sort(dataset, axis=0)
        dataset_sorted /= np.std(dataset_sorted, axis=0)
        shift = int(len(dataset_sorted) * step)
        density = 1 / (np.abs(np.roll(dataset_sorted, -shift, axis=0) - np.roll(dataset_sorted, shift, axis=0) + 1e-3))

        for i in range(dataset.shape[-1])[:1]:
            new_density = np.zeros_like(density[:,i])
            new_density[argsort[:,i]] = density[:,i]
            density[:,i] = new_density
            # plt.scatter(dataset[::100,i], density[::100,i])

        # plt.show()
        density = density / np.max(density, axis=0)

        idx = np.argmax(np.sum(np.log(density), axis=-1), axis=0)
        values = dataset[idx]
        return values

    @staticmethod
    def remap_distribution(dataset, xs, ys):
        b, h, w, c = dataset.shape
        dataset = np.reshape(dataset, (b * h * w, c)).copy()

        for n in range(c):
            dataset[:, n] = np.interp(dataset[:, n], xs[:, n], ys[:, n])

        dataset = np.reshape(dataset, (b, h, w, c))

        return dataset