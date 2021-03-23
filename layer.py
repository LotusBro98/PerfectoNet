import math

import tensorflow as tf
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from distribution import *

class Layer():
    def __init__(self, ksize=3, stride=2, orient="both", eps=1e-2, distribution_approx_n=100, channels=None, equalize=False, dropout=0.2):
        self.ksize = ksize
        self.stride = stride
        self.eps = eps
        self.channels = channels
        self.distribution_approx_n = distribution_approx_n
        self.orient = orient
        self.equalize = equalize
        self.dropout = dropout
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
        x, idx_prev = dataset

        self.input_shape = x.shape

        K, K_T, bias_forward, bias_backward = self._get_optimal_conv_kernel(x)
        self.kernel_forward = K
        self.kernel_backward = K_T
        self.bias_forward = bias_forward
        self.bias_backward = bias_backward

        print("Processing dataset ...")
        x = self.forward(x, batch_size, do_remap=False)
        # x = self.gather(x)

        print("Counting std ...")
        peak_std = get_std_by_peak(x)
        take_channels = np.argsort(peak_std)[::-1]
        collect_std = np.sqrt(np.cumsum(np.square(np.sort(peak_std))))
        n_features = np.count_nonzero(collect_std >= self.eps) if self.channels is None else self.channels

        self.bias_forward = np.take(self.bias_forward, take_channels, axis=-1)
        self.kernel_forward = np.take(self.kernel_forward, take_channels, axis=-1)
        self.kernel_backward = np.take(self.kernel_backward, take_channels, axis=-1)
        x = np.take(x, take_channels, axis=-1)
        self.channel_std = np.take(peak_std, take_channels, axis=-1)
        self.n_features = n_features

        self.calc_embed(x, n_features)
        x, idx = self.embed(x)
        idx = self.merge_embeds(idx, idx_prev)
        self.build_embed_lookup(idx)
        idx = self.embed_lookup(idx)

        # x = self.gather(x)

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

        self.bias_forward = self.bias_forward[:n_features]
        self.kernel_forward = self.kernel_forward[:, :, :, :n_features]
        # self.kernel_backward = self.kernel_backward[:, :, :, :n_features]
        # self.kernel_backward += Km
        # x = x[:,:,:,:n_features]
        self.channel_std = self.channel_std[:n_features]

        self.kernel_backward /= np.expand_dims(self.get_mask(), axis=(-1, -2))
        self.bias_backward /= np.expand_dims(self.get_mask(), axis=(-1, -2))

        # sigma = scipy.special.erfinv(1 - self.dropout) * np.sqrt(2)
        # where = np.abs(x) > (sigma * self.channel_std)
        # x[where] = (np.sign(x) * sigma * self.channel_std)[where]

        # self.xs, self.ys = build_distribution(x, self.distribution_approx_n, equal=False, weights=self.channel_std)

        return x, idx

    def calc_embed(self, x, n_features):
        n_embeds = x.shape[-1] - n_features
        sigma = scipy.special.erfinv(1 - self.dropout) * np.sqrt(2)
        std = np.sqrt(np.sum(np.square(self.channel_std)))
        dist = sigma * std
        x_disc = x[:, :, :, n_features:]
        x_disc = np.reshape(x_disc, (x_disc.shape[0] * x_disc.shape[1] * x_disc.shape[2], x_disc.shape[3]))
        idx = 0 * np.ones_like(x_disc, dtype=np.int32)
        idx[x_disc < -(sigma * self.channel_std[n_features:])] = 1
        idx[x_disc > (sigma * self.channel_std[n_features:])] = 2

        embeds = []
        for i in range(n_embeds):
            embed = np.zeros((3, n_embeds), dtype=np.float32)
            ts = [0, -1, 1]
            for j in range(3):
                t = ts[j] * sigma * self.channel_std[n_features + i]
                embed[j] = np.eye(n_embeds)[i] * t
            embeds.append(embed)

        self.embeds = embeds

    def build_embed_lookup(self, idx, eps=0.22):
        n_embeds = idx.shape[-1]
        idx = np.reshape(idx, (idx.shape[0] * idx.shape[1] * idx.shape[2], idx.shape[3]))
        n_vals = np.max(idx, axis=0) + 1

        groups = self.get_dependence_groups(idx)

        idx_lookup = []

        for group in groups:
            n_vals_group = n_vals[group]
            idx_group = idx[:,group]
            n_combos = np.product(n_vals_group)
            combos = np.zeros((n_combos, len(group)))
            combos[:,0] = np.arange(n_combos)
            for i in range(len(group)-1):
                combos[:,i+1] = combos[:,i] // n_vals_group[i]
                combos[:,i] %= n_vals_group[i]

            probs = np.float32([np.count_nonzero((idx_group == combos[i]).all(axis=-1)) for i in range(n_combos)]) / len(idx_group)
            order = np.argsort(probs)[::-1]
            # take_n = np.count_nonzero(probs > eps)
            take_n = np.count_nonzero(np.cumsum([0] + list(np.sort(probs)[::-1])) < 1 - eps)
            # take_n = len(probs)
            if take_n <= 1:
                continue

            lookup = [
                group,
                combos[order][:take_n]
            ]

            idx_lookup.append(lookup)

        print(idx_lookup)
        self.idx_lookup = idx_lookup
        self.old_idx_len = idx.shape[-1]


    def embed_lookup(self, idx):
        new_idx = np.zeros(idx.shape[:-1] + (len(self.idx_lookup),), dtype=np.int32)
        for i in range(len(self.idx_lookup)):
            group, lookup = self.idx_lookup[i]
            group_idx = idx[:,:,:,group]
            new_idx[:,:,:,i] = np.argmax((np.expand_dims(group_idx, axis=-2) == lookup).all(axis=-1), axis=-1)

        return new_idx


    def embed_unlookup(self, idx):
        old_idx = np.zeros(idx.shape[:-1] + (self.old_idx_len,), dtype=np.int32)
        for i in range(len(self.idx_lookup)):
            group, lookup = self.idx_lookup[i]
            group_idx = np.take(lookup, idx[:,:,:,i], axis=0)
            old_idx[:,:,:,group] = group_idx

        return old_idx


    def get_dependence_groups(self, idx, eps=0.35):
        n_embeds = idx.shape[-1]

        cnt = [np.bincount(idx[:, i]) / len(idx) for i in range(n_embeds)]

        idxs_tree = []
        for i in range(n_embeds):
            vals_i = np.max(idx[:, i]) + 1
            idxs_i = []
            for idx_i in range(vals_i):
                idxs_i.append(idx[idx[:, i] == idx_i])
            idxs_tree.append(idxs_i)

        dependence = np.zeros((n_embeds, n_embeds))
        for i in range(n_embeds):
            vals_i = np.max(idx[:, i]) + 1
            for j in range(n_embeds):
                # print(i, j)
                vals_j = np.max(idx[:, j]) + 1
                common_prob = np.zeros((vals_i, vals_j))
                for idx_i in range(vals_i):
                    for idx_j in range(vals_j):
                        # common_prob[idx_i, idx_j] = np.count_nonzero((idx[:,(i,j)] == (idx_i, idx_j)).all(axis=-1)) / len(idx)
                        common_prob[idx_i, idx_j] = np.count_nonzero(idxs_tree[i][idx_i][:, j] == idx_j) / len(idx)

                sep_prob = np.outer(cnt[i], cnt[j])

                dependence[i, j] = np.max(np.abs(sep_prob - common_prob))

        dependence = np.int32(dependence > eps * np.max(dependence))

        # plt.imshow(dependence)
        # plt.show()

        domains = self.group_domains(dependence)

        # print(domains)

        return domains

        # plt.imshow(D)
        # plt.show()

    def group_domains(self, D):
        order = list(range(len(D)))

        def swap(i, i1):
            t = D[i].copy()
            D[i] = D[i1]
            D[i1] = t

            t = D[:,i].copy()
            D[:,i] = D[:,i1]
            D[:,i1] = t

            t = order[i]
            order[i] = order[i1]
            order[i1] = t

        k = 0
        for i in range(0, len(D) - 1):
            k = max(k, i + 1)
            for j in range(k + 1, len(D)):
                if D[i, j] != 0:
                    swap(k, j)
                    k += 1

                    # plt.imshow(D)
                    # plt.show()

        domains = []
        start = 0
        for i in range(1, len(D)):
            if np.count_nonzero(D[i,0:i]) == 0:
                domains.append(order[start:i])
                # domains.append(list(range(start, i)))
                start = i
        domains.append(list(range(start, len(D))))

        return domains

    def embed(self, x):
        x_float = x[:,:,:,:self.n_features]
        x_embed = x[:,:,:,self.n_features:]

        sigma = scipy.special.erfinv(1 - self.dropout) * np.sqrt(2)

        idx = 0 * np.ones_like(x_embed, dtype=np.int32)
        idx[x_embed < -(sigma * self.channel_std[self.n_features:])] = 1
        idx[x_embed > (sigma * self.channel_std[self.n_features:])] = 2

        return (x_float, idx)


    def merge_embeds(self, idx, idx_prev):
        if self.orient == "both":
            ksize = (self.ksize, self.ksize)
            stride = (self.stride, self.stride)
        elif self.orient == "hor":
            ksize = (1, self.ksize)
            stride = (1, self.stride)
        elif self.orient == "ver":
            ksize = (self.ksize, 1)
            stride = (self.stride, 1)

        patches = tf.image.extract_patches(
            idx_prev,
            [1, ksize[0], ksize[1], 1],
            [1, stride[0], stride[1], 1],
            [1, 1, 1, 1], padding='VALID')

        return np.concatenate([idx, patches], axis=-1)

    def unmerge_embeds(self, idx):
        if self.orient == "both":
            ksize = (self.ksize, self.ksize)
            stride = (self.stride, self.stride)
        elif self.orient == "hor":
            ksize = (1, self.ksize)
            stride = (1, self.stride)
        elif self.orient == "ver":
            ksize = (self.ksize, 1)
            stride = (self.stride, 1)

        assert self.ksize == 3
        assert self.stride in (1, 2)

        n_embeds = self.kernel_backward.shape[-1] - self.n_features
        idx_prev = idx[:,:,:,n_embeds:]
        idx = idx[:,:,:,:n_embeds]

        n_embeds_prev = idx_prev.shape[-1] // (ksize[0] * ksize[1])
        b, h, w, c = idx_prev.shape
        new_h = (2 * h + 1) if stride[0] == 2 else h
        new_w = (2 * w + 1) if stride[1] == 2 else w
        new_idx_prev = np.zeros((b, new_h, new_w, n_embeds_prev), dtype=np.int32)
        idx_prev_get = np.reshape(idx_prev, idx_prev.shape[:-1] + ksize + (n_embeds_prev,))

        for i in range(new_h):
            for j in range(new_w):
                new_idx_prev[:, i, j] = idx_prev_get[:, i//ksize[0], j//ksize[1], i%ksize[0], j%ksize[1]]

        for i in range(new_h):
            # new_idx_prev[:, i, 0] = idx_prev_get[:, i//ksize[0], 0, i%ksize[0], 0]
            new_idx_prev[:, i, -1] = idx_prev_get[:, i//ksize[0], -1, i%ksize[0], -1]

        for j in range(new_w):
            # new_idx_prev[:, 0, j] = idx_prev_get[:, 0, j // ksize[1], 0, j % ksize[1]]
            new_idx_prev[:, -1, j] = idx_prev_get[:, -1, j // ksize[1], -1, j % ksize[1]]

        return idx, new_idx_prev


    def unembed(self, x):
        x_float = x[0]
        n_embeds = self.kernel_backward.shape[-1] - x_float.shape[-1]
        idx = x[1]
        idx = self.embed_unlookup(idx)
        idx, idx_prev = self.unmerge_embeds(idx)
        # idx_prev = idx[:,:,:,:0]

        x_embed = np.zeros(idx.shape[:-1] + (n_embeds,), dtype=np.float32)

        for i in range(n_embeds):
            x_embed += np.take(self.embeds[i], idx[:,:,:,i], axis=0)

        # x_embed *= 0

        return np.concatenate([x_float, x_embed], axis=-1), idx_prev



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

        if hasattr(self, "embeds"):
            x, idx = self.embed(x[0])
            idx = self.merge_embeds(idx, x[1])
            idx = self.embed_lookup(idx)

        # if hasattr(self, "channel_std") and self.channel_std is not None:
        #     # x = self.channel_std * np.tanh(x / self.channel_std)
        #     x = self.gather(x)
        #     # sigma = scipy.special.erfinv(1 - self.dropout) * np.sqrt(2)
        #     # where = np.abs(x) > (sigma * self.channel_std)
        #     # x[where] = (np.sign(x) * sigma * self.channel_std)[where]

        # if do_remap:
        #     x = remap_distribution(x, self.xs, self.ys)

        if hasattr(self, "embeds"):
            return (x, idx)
        else:
            return x

    def gather(self, x):
        sigma = scipy.special.erfinv(1 - self.dropout) * np.sqrt(2)
        std = np.sqrt(np.sum(np.square(self.channel_std)))
        # sigma = 5

        # x = np.clip(x / self.channel_std, -sigma, sigma) * self.channel_std

        x = x / std
        # x = np.log(1 + np.abs(x)) * np.sign(x)
        x = np.clip(x, -sigma, sigma)
        x = x * std
        return x

    def scatter(self, x):
        sigma = scipy.special.erfinv(1 - self.dropout) * np.sqrt(2)
        std = np.sqrt(np.sum(np.square(self.channel_std)))
        # sigma = 5

        # x = np.clip(x / self.channel_std, -sigma, sigma) * self.channel_std

        x = x / std
        x = np.clip(x, -sigma, sigma)
        # x = (np.exp(np.abs(x)) - 1) * np.sign(x)
        x = x * std
        return x

    def backward(self, x, batch_size=1):
        x, idx_prev = self.unembed(x)
        # x = remap_distribution(x, self.ys, self.xs)
        # x = self.channel_std * np.arctanh(x / self.channel_std)
        # x = self.scatter(x)

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
            batch = batch.numpy()
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
        return x, idx_prev

    def save(self, n):
        np.save("saved/K{}".format(n), self.kernel_forward)
        np.save("saved/b{}".format(n), self.bias_forward)
        np.save("saved/K_T{}".format(n), self.kernel_backward)
        np.save("saved/b_T{}".format(n), self.bias_backward)
        np.save("saved/std{}".format(n), self.channel_std)
        np.save("saved/in{}".format(n), self.input_shape)
        # np.save("saved/xs{}".format(n), self.xs)
        # np.save("saved/ys{}".format(n), self.ys)

    def load(self, n):
        self.kernel_forward = np.load("saved/K{}.npy".format(n))
        self.bias_forward = np.load("saved/b{}.npy".format(n))
        self.kernel_backward = np.load("saved/K_T{}.npy".format(n))
        self.bias_backward = np.load("saved/b_T{}.npy".format(n))
        self.channel_std = np.load("saved/std{}.npy".format(n))
        self.input_shape = np.load("saved/in{}.npy".format(n))
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
        # if self.equalize:
        #     K = np.matmul(np.diag(1/np.sqrt(S / np.average(S)) / std), K)
        # else:
        K = K / (np.count_nonzero(mask) / np.sum(1 / mask))
        bias_forward = -tf.squeeze(tf.matmul(K, tf.expand_dims(mean_p, axis=-1)))
        K = tf.reshape(K, (V.shape[0], ksize[0], ksize[1], Cin))
        K = tf.transpose(K, (1, 2, 3, 0))

        K_T = tf.transpose(V)
        # if self.equalize:
        #     K_T = np.matmul(K_T, np.diag(np.sqrt(S / np.average(S)) * std))
        # else:
        K_T = K_T * (np.count_nonzero(mask) / np.sum(1 / mask))
        bias_backward = mean_p
        bias_backward = tf.reshape(bias_backward, (ksize[0], ksize[1], Cin, 1))
        # bias_backward /= np.expand_dims(mask, axis=(-1, -2))
        K_T = tf.reshape(K_T, (ksize[0], ksize[1], Cin, V.shape[0]))
        K_T = tf.transpose(K_T, (0, 1, 2, 3))
        # K_T /= np.expand_dims(mask, axis=(-1, -2))

        return K, K_T, bias_forward, bias_backward