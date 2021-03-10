import glob
import math
import os
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

SIZE = 511
DATASET_SIZE = 1500
BATCH_SIZE = 4
TEST_SIZE = 2
N_LAYERS = 8
EPS = [-4, -3, -2, -1, 0, 1, 2, 3]
EPS_L_A = 3
EPS_L_B = 3
# CHANNELS_L = 256
NOISE = 0.0


def load_image(filename, image_size=512):
    image = tf.io.decode_image(tf.io.read_file(filename), channels=3)
    image = image / 255
    image.set_shape((image_size, image_size, 3))
    image = tf.image.resize(image, (SIZE, SIZE))

    return image


images = []
pathsA = list(glob.glob("trainA/*.png"))
pathsB = list(glob.glob("trainB/*.png"))
random.shuffle(pathsA)
random.shuffle(pathsB)
paths = pathsA[:DATASET_SIZE] + pathsB[:DATASET_SIZE] + pathsA[DATASET_SIZE:DATASET_SIZE+TEST_SIZE] + pathsB[DATASET_SIZE:DATASET_SIZE+TEST_SIZE]
# random.shuffle(paths)

for path in paths:
    image = load_image(path)
    image = image.numpy()
    images.append(image)
images = np.stack(images)

inp = images


def get_optimal_conv_kernel(dataset, ksize=3, stride=2, eps=0.1, channels=None, batch_size_cells=100000000):
    assert ksize in [1, 3]
    assert stride in [1, 2]

    Cin = dataset.shape[-1]
    cells = dataset.shape[1] * dataset.shape[2] * np.square(dataset.shape[3])
    batch_size_images = math.ceil(batch_size_cells / (cells))

    mean = np.average(dataset, axis=(0, 1, 2))
    mean_p = np.reshape(tf.repeat(tf.expand_dims(mean, axis=0), ksize * ksize, axis=0), (ksize * ksize * Cin))

    N = 0
    M = 0
    for i in range(0, dataset.shape[0], batch_size_images):
        batch = dataset[i: i + batch_size_images]
        patches = tf.image.extract_patches(batch, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='VALID')
        patches = tf.reshape(patches, [patches.shape[0] * patches.shape[1] * patches.shape[2], patches.shape[3]])
        patches = patches - mean_p

        batch_size_mat = math.ceil(batch_size_cells / (patches.shape[1] * patches.shape[1]))
        for j in range(0, len(patches), batch_size_mat):
            batch_mat = patches[j:j+batch_size_mat]
            cov = tf.matmul(tf.expand_dims(batch_mat, axis=-1), tf.expand_dims(batch_mat, axis=-2))
            cov = tf.reduce_sum(cov, axis=0)

            N += batch_mat.shape[0]
            M += cov

        print("\rProcessing sample {} / {}".format(i, dataset.shape[0]), end='')
    print()

    M = M / N
    print("SV Decomposition running...")
    U, S, V = np.linalg.svd(M)
    # S, U, V = tf.linalg.svd(M)
    # V = tf.linalg.adjoint(V)

    Sl = np.log(S)
    thresh = eps
    plt.plot(Sl)
    plt.plot(np.ones_like(Sl) * thresh)
    plt.show()
    # n_features = tf.math.count_nonzero(tf.sqrt(S) > eps) if channels is None else channels
    n_features = tf.math.count_nonzero(Sl > thresh) if channels is None else channels

    V = V[:n_features]
    S = S[:n_features]

    K = V
    bias_forward = -tf.squeeze(tf.matmul(K, tf.expand_dims(mean_p, axis=-1)))
    K = tf.reshape(K, (n_features, ksize, ksize, Cin))
    K = tf.transpose(K, (1, 2, 3, 0))

    K_T = tf.transpose(V)
    bias_backward = mean_p
    bias_backward = tf.reshape(bias_backward, (ksize * ksize, Cin))
    bias_backward = tf.reduce_mean(bias_backward, axis=0)
    K_T = tf.reshape(K_T, (ksize, ksize, Cin, n_features))
    K_T = tf.transpose(K_T, (0, 1, 2, 3))
    if stride == 1:
        mask = np.ones((ksize, ksize)) * (ksize * ksize)
    elif stride == 2:
        if ksize == 1:
            mask = np.float32([[1]])
        elif ksize == 3:
            mask = np.float32([
                [4, 2, 4],
                [2, 1, 2],
                [4, 2, 4]
            ])
    K_T /= np.expand_dims(mask, axis=(-1, -2))

    return K, K_T, bias_forward, bias_backward


Ks = []
K_Ts = []
biases_forward = []
biases_backward = []

x_forward = []
x = inp
for i in range(N_LAYERS):
    # ch = CHANNELS[i] if i < len(CHANNELS) else None
    # eps = EPS[i] if i < len(EPS) else None
    eps = EPS[i] if i < len(EPS) else EPS[-1]
    # K, K_T, bias_forward, bias_backward = get_optimal_conv_kernel(x[:2*DATASET_SIZE], channels=ch, eps=eps)
    K, K_T, bias_forward, bias_backward = get_optimal_conv_kernel(x[:2*DATASET_SIZE], eps=eps)
    Ks.append(K)
    K_Ts.append(K_T)
    biases_forward.append(bias_forward)
    biases_backward.append(bias_backward)
    x_forward.append(x)

    x_conv = []
    for i in range(0, x.shape[0], BATCH_SIZE):
        batch = x[i: i + BATCH_SIZE]
        batch = tf.nn.conv2d(batch, K, (1, 2, 2, 1), padding="VALID") + bias_forward
        batch = batch.numpy()
        x_conv.append(batch)
        print("\rProcessing (Conv2D) sample {} / {}".format(i, x.shape[0]), end='')
    print()
    x = np.concatenate(x_conv, axis=0)
    print(x.shape)

scale_std = np.std(x, axis=(0, 1, 2))
# x /= scale_std

x_A0 = x[:DATASET_SIZE]
x_B0 = x[DATASET_SIZE:2*DATASET_SIZE]

avg_A = np.average(x_A0, axis=(0,1,2))
avg_B = np.average(x_B0, axis=(0,1,2))
std_A = np.std(x_A0, axis=(0,1,2))
std_B = np.std(x_B0, axis=(0,1,2))

KA, KA_T, biasA_forward, biasA_backward = get_optimal_conv_kernel(x_A0, eps=EPS_L_A, ksize=1, stride=1)
KB, KB_T, biasB_forward, biasB_backward = get_optimal_conv_kernel(x_B0, eps=EPS_L_B, ksize=1, stride=1)

x = x[2*DATASET_SIZE:]
inp = inp[2*DATASET_SIZE:]

x_A = x[:TEST_SIZE]
x_B = x[TEST_SIZE:]

x_A -= avg_A
x_A *= std_B / std_A
x_A += avg_B

x_B -= avg_B
x_B *= std_A / std_B
x_B += avg_A

x_A_B = tf.nn.conv2d(x_A, KB, (1, 1, 1, 1), padding="VALID") + biasB_forward
print("x_A->B: {}".format(x_A_B.shape))
x_A = tf.nn.conv2d_transpose(x_A_B, KB_T, x_A.shape, (1, 1, 1, 1), padding="VALID") + biasB_backward

x_B_A = tf.nn.conv2d(x_B, KA, (1, 1, 1, 1), padding="VALID") + biasA_forward
print("x_B->A: {}".format(x_B_A.shape))
x_B = tf.nn.conv2d_transpose(x_B_A, KA_T, x_B.shape, (1, 1, 1, 1), padding="VALID") + biasA_backward

x = np.concatenate([x_A, x_B], axis=0)

# x *= scale_std
for i in range(N_LAYERS-1, -1, -1):
    b, h, w, c = x_forward[i].shape
    x = tf.nn.conv2d_transpose(x, K_Ts[i], (2*TEST_SIZE, h, w, c), (1, 2, 2, 1), padding="VALID")
    x = x.numpy()
    x[:, 0, :, :] *= 2
    x[:, -1, :, :] *= 2
    x[:, :, 0, :] *= 2
    x[:, :, -1, :] *= 2
    x += biases_backward[i]
    print(x.shape)

outp = x

f, ax = plt.subplots(2, 2*TEST_SIZE, figsize=(20, 10))

for i in range(2*TEST_SIZE):
    ax[0, i].imshow(inp[i])
    ax[1, i].imshow(outp[i])
plt.show()