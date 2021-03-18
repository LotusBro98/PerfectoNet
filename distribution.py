import math

import tensorflow as tf
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import scipy.stats as st


# def show_distribution(dataset):
#     dataset = np.reshape(dataset, [dataset.shape[0] * dataset.shape[1] * dataset.shape[2], dataset.shape[3]])
#     dataset = np.sort(dataset, axis=0)
#     dataset /= np.std(dataset, axis=0)
#     shift = int(len(dataset) * 1e-1)
#     density = 1 / (dataset[shift:] - dataset[:-shift])
#
#     for slice in density.T:
#         plt.plot(slice)
#     plt.show()

def show_distribution(dataset, approx_n=64):
    dataset = np.reshape(dataset, [dataset.shape[0] * dataset.shape[1] * dataset.shape[2], dataset.shape[3]])
    density = np.zeros((dataset.shape[-1], approx_n,))
    for n in range(dataset.shape[-1]):
        mean = np.average(dataset[:, n])
        std = np.std(dataset[:, n])
        checks = np.linspace(mean - std, mean + std, approx_n + 1)
        for i in range(0, approx_n):
            count = np.count_nonzero((dataset[:, n] >= checks[i]) * (dataset[:, n] < checks[i + 1]))
            density[n, i] = count

    density = density[:,1:-1]
    density /= np.sum(density, axis=-1, keepdims=True)

    plt.plot(density.T)
    plt.show()

def get_density(dataset, approx_n=33, sigma=2):
    dataset = np.reshape(dataset, [dataset.shape[0] * dataset.shape[1] * dataset.shape[2], dataset.shape[3]])
    density = np.zeros((dataset.shape[-1], approx_n,))
    mean = np.average(dataset, axis=0)
    std = np.std(dataset, axis=0)
    for n in range(dataset.shape[-1]):
        checks = np.linspace(mean[n] - sigma * std[n], mean[n] + sigma * std[n], approx_n + 1)
        for i in range(0, approx_n):
            count = np.count_nonzero((dataset[:, n] >= checks[i]) * (dataset[:, n] < checks[i + 1]))
            density[n, i] = count

    density = density[:, 1:-1]
    density /= np.sum(density, axis=-1, keepdims=True)

    return density, mean, std

def get_gaussian_channels(density, sigma=2, eps=-1):
    gx = np.linspace(-sigma, sigma, density.shape[-1]+1)
    kern1d = np.diff(st.norm.cdf(gx))
    kern1d /= np.sum(kern1d)

    gaussian_diff = density - np.expand_dims(kern1d, axis=0)
    gaussian_diff = (np.average(np.abs(gaussian_diff), axis=-1))
    gaussian_diff = np.log(gaussian_diff)

    indices = np.argsort(gaussian_diff)

    gaussian_diff = np.sort(gaussian_diff)
    m, M = gaussian_diff[0], gaussian_diff[-1]
    gaussian_diff = (gaussian_diff - m) / (M - m)
    # gaussian_diff = np.cumsum(gaussian_diff)
    # gaussian_diff /= gaussian_diff[-1]
    plt.plot(gaussian_diff)
    # plt.plot(np.ones_like(gaussian_diff) * eps)
    plt.plot([eps * (density.shape[0]-1)] * 2, [0, 1])
    plt.show()

    # n_channels = np.count_nonzero(gaussian_diff <= eps)
    n_channels = int(math.ceil(density.shape[0] * eps))

    return indices, n_channels


def get_non_peak_channels(dataset, sigma=0.05, eps=0.1):
    dataset = np.reshape(dataset, [dataset.shape[0] * dataset.shape[1] * dataset.shape[2], dataset.shape[3]])
    mean = np.average(dataset, axis=0)
    std = np.std(dataset, axis=0)
    count = np.zeros((dataset.shape[-1]))
    for n in range(dataset.shape[-1]):
        m, M = mean[n] - sigma * std[n], mean[n] + sigma * std[n]
        count[n] = np.count_nonzero((dataset[:, n] >= m) * (dataset[:, n] < M))
        count[n] = count[n] / len(dataset)

    indices = np.argsort(count)
    n_channels = np.count_nonzero(count < eps)
    return indices, n_channels

def build_distribution(dataset, approx_n=100, equal=True, weights=None):
    dataset = np.reshape(dataset, [dataset.shape[0] * dataset.shape[1] * dataset.shape[2], dataset.shape[3]])
    dataset = np.sort(dataset, axis=0)

    ys = np.linspace(1/approx_n, 1-1/approx_n, approx_n)
    inds = np.int32(ys * len(dataset))
    inds[0] = 0
    inds[-1] = len(dataset) - 1
    xs = dataset[inds, :].copy()
    ys = np.repeat(np.expand_dims(ys, axis=-1), xs.shape[-1], axis=-1)

    ys = ys * 2 - 1

    # ys *= 3
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

# def get_most_probable_values(density, mean, std, sigma=2):
#     idx = np.argmax(density, axis=-1)
#     min = mean - sigma * std
#     max = mean + sigma * std
#     values = min + (idx / density.shape[-1]) * (max - min)
#     return values

def get_most_probable_values(dataset, step=2e-2, eps=1e-4):
    dataset = np.reshape(dataset, [dataset.shape[0] * dataset.shape[1] * dataset.shape[2], dataset.shape[3]])
    argsort = np.argsort(dataset, axis=0)
    dataset_sorted = np.sort(dataset, axis=0)
    dataset_sorted /= np.std(dataset_sorted, axis=0)
    shift = int(len(dataset_sorted) * step)
    density = (np.abs(np.roll(dataset_sorted, -shift, axis=0) - np.roll(dataset_sorted, shift, axis=0))) / (2 * shift)
    while np.min(density) < eps * np.max(density):
        shift = int(math.ceil(shift * 1.4))
        density = ((np.abs(np.roll(dataset_sorted, -shift, axis=0) - np.roll(dataset_sorted, shift, axis=0))) / (2 * shift))
    density = 1 / density

    for i in range(dataset.shape[-1]):
        new_density = np.zeros_like(density[:,i])
        new_density[argsort[:,i]] = density[:,i]
        density[:,i] = new_density
        # plt.scatter(dataset[::100,i], density[::100,i])

    # plt.show()
    density = density / np.max(density, axis=0)

    idx = np.argmax(np.sum(np.log(density), axis=-1), axis=0)
    values = dataset[idx]
    return values

# def get_most_probable_values(self, dataset, step=2e-2, eps=1e-4):
#     dataset = np.reshape(dataset, [dataset.shape[0] * dataset.shape[1] * dataset.shape[2], dataset.shape[3]])
#     dataset_sorted = np.sort(dataset, axis=0)
#     dataset_sorted /= np.std(dataset_sorted, axis=0)
#     shift = int(len(dataset_sorted) * step)
#     density = (np.abs(np.roll(dataset_sorted, -shift, axis=0) - np.roll(dataset_sorted, shift, axis=0))) / (2 * shift)
#     while np.min(density) < eps * np.max(density):
#         shift = int(math.ceil(shift * 1.4))
#         density = ((np.abs(np.roll(dataset_sorted, -shift, axis=0) - np.roll(dataset_sorted, shift, axis=0))) / (2 * shift))
#     density = 1 / density
#
#     idx = np.argmax(density, axis=0)
#     values = [dataset[idx[i], i] for i in range(dataset.shape[-1])]
#     values = np.float32(values)
#
#     return values

def remap_distribution(dataset, xs, ys):
    b, h, w, c = dataset.shape
    dataset = np.reshape(dataset, (b * h * w, c)).copy()

    for n in range(c):
        dataset[:, n] = np.interp(dataset[:, n], xs[:, n], ys[:, n])

    dataset = np.reshape(dataset, (b, h, w, c))

    return dataset


def common_distribution(dataset, approx_n=None, c1=0, c2=1):
    dataset = dataset[:, :, :,[c1, c2]]
    dataset = np.reshape(dataset, [dataset.shape[0] * dataset.shape[1] * dataset.shape[2], 2])
    inds = np.argsort(dataset[:, 0])
    dataset = dataset.take(inds, axis=0).copy()

    if approx_n is None:
        approx_n = int(math.ceil(np.sqrt(len(dataset)) / 10))

    density = np.zeros((approx_n, approx_n))

    mean = np.average(dataset, axis=0)
    std = np.std(dataset, axis=0)

    checks = np.linspace(mean[0] - std[0], mean[0] + std[0], approx_n + 1)
    checks1 = np.linspace(mean[1] - std[1], mean[1] + std[1], approx_n + 1)
    for i in range(0, approx_n):
        ind1 = np.argmax(dataset[:, 0] >= checks[i]) if (dataset[:, 0] >= checks[i]).any() else len(dataset)
        ind2 = np.argmax(dataset[:, 0] >= checks[i + 1]) if (dataset[:, 0] >= checks[i + 1]).any() else len(dataset)
        subset = dataset[ind1:ind2, 1].copy()
        if len(subset) == 0:
            continue
        subset.sort()
        for j in range(0, approx_n):
            ind11 = np.argmax(subset >= checks1[j]) if (subset >= checks1[j]).any() else len(subset)
            ind21 = np.argmax(subset >= checks1[j + 1]) if (subset >= checks1[j + 1]).any() else len(subset)
            density[i, j] = ind21 - ind11

    density /= np.sum(density)

    return density

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def show_common_distributions(dataset, eps=0.25):
    approx_n_opt = int(math.ceil(np.sqrt(len(dataset)) * eps))
    approx_n_check = 32
    approx_n_show = 50

    # if approx_n_check > approx_n_opt:
    #     approx_n_check = approx_n_opt
    # if approx_n_show > approx_n_opt:
    #     approx_n_show = approx_n_opt

    dependence_matrix = np.zeros((dataset.shape[-1], dataset.shape[-1]))

    N = dataset.shape[-1]  # int(math.ceil(np.sqrt(len(pairs))))
    fig = plt.figure(figsize=(2 * N, 2 * N))

    pairs = []
    for i in range(dataset.shape[-1]):
        for j in range(dataset.shape[-1]):
            if i == j:
                dependence_matrix[i, j] = 1
                # continue
            density = common_distribution(dataset, c1=i, c2=j, approx_n=approx_n_check)
            density0 = np.sum(density, axis=1)
            density1 = np.sum(density, axis=0)
            density0 /= np.sum(density0)
            density1 /= np.sum(density1)
            density_equal = np.outer(density0, density1)
            density_diff = np.sum(np.abs(density_equal - density))
            print(density_diff)
            if density_diff > eps:
                dependence_matrix[i, j] = 1
            else:
                dependence_matrix[i, j] = 0
                continue

            density = common_distribution(dataset, approx_n=approx_n_show, c1=i, c2=j)
            # density = density_equal
            # print(density)
            # density = density / gkern(len(density), 2)
            # density = density[1:-1, 1:-1]
            mean = np.average(dataset, axis=(0, 1, 2))
            std = np.std(dataset, axis=(0, 1, 2))
            extent = [
                mean[j] - std[j], mean[j] + std[j],
                mean[i] - std[i], mean[i] + std[i],
            ]
            ax = fig.add_subplot(N, N, i * N + j + 1)
            ax.imshow(density[::-1, :],
                       vmax=3 * np.sqrt(np.average(np.square(density))),
                       extent=extent,
                       aspect=(extent[1] - extent[0]) / (extent[3] - extent[2])
                       )
            # ax.axis('off')
    plt.show()


def get_independent_channels(dataset, approx_n=10, eps=0.2):
    dependent = set()
    for i in range(dataset.shape[-1]):
        for j in range(i + 1, dataset.shape[-1]):
            density = common_distribution(dataset, c1=i, c2=j, approx_n=10)
            density = density / gkern(len(density), 2)
            density = density[1:-1, 1:-1]
            if np.std(density) > eps * np.average(density):
                dependent.add(i)
                dependent.add(j)

    independent = []
    for i in range(dataset.shape[-1]):
        if i not in dependent:
            independent.append(i)

    return independent