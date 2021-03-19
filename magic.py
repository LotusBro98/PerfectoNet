import glob
import math
import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from distribution import *
from layer import Layer
from model import Model

SIZE = 127
DATASET_SIZE = 500
BATCH_SIZE = 4
TEST_SIZE = 4
EPS_START = -8
EPS_END = 1
EPS_L = 1
ALPHA = 1

N_LAYERS = 6

EPS1 = np.linspace(-3, 0, 6)
EPS2 = np.linspace(0.2, 0.1, 6)
# EPS1 = -4
# EPS2 = 0.12

EPS_L_A = 4
EPS_D_A = 0.06
EPS_L_B = 4
EPS_D_B = 0.06
DISTRIBUTION_APPROX_N = 64


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

inp_A0 = inp[:DATASET_SIZE]
inp_B0 = inp[DATASET_SIZE:2*DATASET_SIZE]
xsA0, ysA0 = build_distribution(inp_A0, equal=True)
xsB0, ysB0 = build_distribution(inp_B0, equal=True)
inp_A0 = remap_distribution(inp_A0, xsA0, ysA0)
inp_B0 = remap_distribution(inp_B0, xsB0, ysB0)

inp[:2*DATASET_SIZE] = np.concatenate([inp_A0, inp_B0])
inp[2*DATASET_SIZE:2*DATASET_SIZE+TEST_SIZE] = remap_distribution(inp[2*DATASET_SIZE:2*DATASET_SIZE+TEST_SIZE], xsA0, ysA0)
inp[2*DATASET_SIZE+TEST_SIZE:] = remap_distribution(inp[2*DATASET_SIZE+TEST_SIZE:], xsB0, ysB0)


def get_nearest(dataset, subset):
    diff = np.expand_dims(dataset, axis=0) - np.expand_dims(subset, axis=1)
    dist = np.sum(np.square(diff), axis=tuple(range(len(diff.shape)))[2:])
    near = np.argmin(dist, axis=-1)
    near = np.take(dataset, near, axis=0)
    return near


model = Model(N_LAYERS, EPS1, EPS2, approx_n=DISTRIBUTION_APPROX_N)
model.fit(inp, BATCH_SIZE)

x = model.forward(inp, BATCH_SIZE)

x_A0 = x[:DATASET_SIZE]
x_B0 = x[DATASET_SIZE:2*DATASET_SIZE]
# all_std = np.std(x, axis=(0,1,2))

project_A = Layer(ksize=1, stride=1, epsL=EPS_L_A, epsD=EPS_D_A)
xsA, ysA = build_distribution(x_A0, equal=True)#, weights=all_std)
# project_A.fit(remap_distribution(x_A0, xsA, ysA))
project_A.fit(x_A0)

project_B = Layer(ksize=1, stride=1, epsL=EPS_L_B, epsD=EPS_D_B)
xsB, ysB = build_distribution(x_B0, equal=True)#, weights=all_std)
# project_B.fit(remap_distribution(x_B0, xsB, ysB))
project_B.fit(x_B0)

avgA = np.average(x_A0, axis=(0,1,2))
stdA = np.std(x_A0, axis=(0,1,2))
# print(avgA)
# print(stdA)
normA = np.sqrt(np.average(np.square(np.linalg.norm(x_A0 - avgA, axis=-1))))
# print(normA)

avgB = np.average(x_B0, axis=(0,1,2))
stdB = np.std(x_B0, axis=(0,1,2))
# print(avgB)
# print(stdB)
normB = np.sqrt(np.average(np.square(np.linalg.norm(x_B0 - avgB, axis=-1))))
# print(normB)

x_A = remap_distribution(x_A0, xsA, ysA)
x_A = remap_distribution(x_A, ysB, xsB)
x_A_B = project_B.forward(x_A)
print("x_A->B: {}".format(x_A_B.shape))
# x_A = project_B.backward(x_A_B)
# xsA_B, ys_A_B = build_distribution(x_A, equal=True)

x_B = remap_distribution(x_B0, xsB, ysB)
x_B = remap_distribution(x_B, ysA, xsA)
x_B_A = project_A.forward(x_B)
print("x_B->A: {}".format(x_B_A.shape))
# x_B = project_A.backward(x_B_A)
# xsB_A, ysB_A = build_distribution(x_B, equal=True)


x = x[2*DATASET_SIZE:]
inp = np.concatenate([inp[:1], inp[2*DATASET_SIZE:2*DATASET_SIZE+TEST_SIZE], inp[DATASET_SIZE:DATASET_SIZE+1], inp[2*DATASET_SIZE+TEST_SIZE:]], axis=0)

x_A = x[:TEST_SIZE]
x_B = x[TEST_SIZE:]

x_A = remap_distribution(x_A, xsA, ysA)
x_A = remap_distribution(x_A, ysB, xsB)
# x_A = (x_A - avgA) / stdA * stdB + avgB
for i in range(16):
    x_A = project_B.forward(x_A)
    x_A = project_B.backward(x_A)
    x_A -= avgB
    norm = np.linalg.norm(x_A, axis=-1, keepdims=True)
    x_A = x_A / norm * normB
    x_A += avgB
# x_A = remap_distribution(x_A, xsA_B, ys_A_B)
# x_A = remap_distribution(x_A, ysB, xsB)
# x_A = project_A.forward(x_A)
# x_A = project_B.backward(x_A)

x_B = remap_distribution(x_B, xsB, ysB)
x_B = remap_distribution(x_B, ysA, xsA)
# x_B = (x_B - avgB) / stdB * stdA + avgA
for i in range(16):
    x_B = project_A.forward(x_B)
    x_B = project_A.backward(x_B)
    x_B -= avgA
    norm = np.linalg.norm(x_B, axis=-1, keepdims=True)
    x_B = x_B / norm * normA
    x_B += avgA
# x_B = remap_distribution(x_B, xsB_A, ysB_A)
# x_B = remap_distribution(x_B, ysA, xsA)
# x_B = project_B.forward(x_B)
# x_B = project_A.backward(x_B)

x = np.concatenate([x_A0[:1], x_A, x_B0[:1], x_B], axis=0)

x = model.backward(x, BATCH_SIZE)

outp = x

inp[:TEST_SIZE+1] = remap_distribution(inp[:TEST_SIZE+1], ysA0, xsA0)
inp[TEST_SIZE+1:] = remap_distribution(inp[TEST_SIZE+1:], ysB0, xsB0)

outp[:TEST_SIZE+1] = remap_distribution(outp[:TEST_SIZE+1], ysB0, xsB0)
outp[TEST_SIZE+1:] = remap_distribution(outp[TEST_SIZE+1:], ysA0, xsA0)

f, ax = plt.subplots(2, 2*(TEST_SIZE+1), figsize=(10*TEST_SIZE, 10))

for i in range(2*(TEST_SIZE+1)):
    ax[0, i].imshow(inp[i])
    ax[1, i].imshow(outp[i])
plt.show()