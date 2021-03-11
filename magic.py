import glob
import math
import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from layer import Layer
from model import Model

SIZE = 63
DATASET_SIZE = 1000
BATCH_SIZE = 4
TEST_SIZE = 2
EPS_START = -5
EPS_END = -1
# EPS = np.linspace(EPS_START, EPS_END, 5)
EPS = [EPS_START, -4, -4, -3, EPS_END][:6]
# EPS = [0.002] * 5
EPS_L_A = EPS_END
EPS_L_B = EPS_END
DISTRIBUTION_APPROX_N = 100


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


def show_distribution(dataset):
    dataset = np.reshape(dataset, [dataset.shape[0] * dataset.shape[1] * dataset.shape[2], dataset.shape[3]])
    dataset = np.sort(dataset, axis=0)
    dataset /= np.std(dataset, axis=0)
    shift = int(len(dataset) * 1e-1)
    density = 1 / (dataset[shift:] - dataset[:-shift])

    for slice in density.T:
        plt.plot(slice)
    plt.show()


model = Model(len(EPS), EPS, approx_n=DISTRIBUTION_APPROX_N)
model.fit(inp, BATCH_SIZE)

x = model.forward(inp, BATCH_SIZE)

x_A0 = x[:DATASET_SIZE]
x_B0 = x[DATASET_SIZE:2*DATASET_SIZE]
all_std = np.std(x, axis=(0,1,2))

project_A = Layer(ksize=1, stride=1, eps=EPS_L_A)
xsA, ysA = Layer.build_distribution(x_A0, weights=all_std)
# project_A.fit(Layer.remap_distribution(x_A0, xsA, ysA))
project_A.fit(x_A0)

project_B = Layer(ksize=1, stride=1, eps=EPS_L_B)
xsB, ysB = Layer.build_distribution(x_B0, weights=all_std)
# project_B.fit(Layer.remap_distribution(x_B0, xsB, ysB))
project_B.fit(x_B0)

x = x[2*DATASET_SIZE:]
inp = inp[2*DATASET_SIZE:]

x_A = x[:TEST_SIZE]
x_B = x[TEST_SIZE:]

x_A = Layer.remap_distribution(x_A, xsA, ysA)
x_A = Layer.remap_distribution(x_A, ysB, xsB)
x_A_B = project_B.forward(x_A)
print("x_A->B: {}".format(x_A_B.shape))
x_A = project_B.backward(x_A_B)

x_B = Layer.remap_distribution(x_B, xsB, ysB)
x_B = Layer.remap_distribution(x_B, ysA, xsA)
x_B_A = project_A.forward(x_B)
print("x_B->A: {}".format(x_B_A.shape))
x_B = project_A.backward(x_B_A)

x = np.concatenate([x_A, x_B], axis=0)

x = model.backward(x, BATCH_SIZE)

outp = x

f, ax = plt.subplots(2, 2*TEST_SIZE, figsize=(20, 10))

for i in range(2*TEST_SIZE):
    ax[0, i].imshow(inp[i])
    ax[1, i].imshow(outp[i])
plt.show()