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

BATCH_SIZE = 4
TEST_SIZE = 4

N_LAYERS = 6

EPS_START = 0.1
EPS_END = 0.2

EPS_L_A = 0.3
EPS_L_B = 0.3


SIZE = 2**(N_LAYERS + 1) - 1

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
paths = pathsA[:TEST_SIZE] + pathsB[:TEST_SIZE]
# random.shuffle(paths)

for path in paths:
    image = load_image(path)
    image = image.numpy()
    images.append(image)
images = np.stack(images)

inp = images


model = Model(N_LAYERS, EPS_START, EPS_END, approx_n=DISTRIBUTION_APPROX_N)
model.load()

xsA0 = np.load("saved/xsA0.npy")
ysA0 = np.load("saved/ysA0.npy")
xsB0 = np.load("saved/xsB0.npy")
ysB0 = np.load("saved/ysB0.npy")

x_A0 = np.load("saved/x_A0.npy")
x_B0 = np.load("saved/x_B0.npy")

project_A = Layer(ksize=1, stride=1, eps=EPS_L_A)
xsA, ysA = build_distribution(x_A0, equal=True)#, weights=all_std)
# project_A.fit(remap_distribution(x_A0, xsA, ysA))
project_A.fit(x_A0)

project_B = Layer(ksize=1, stride=1, eps=EPS_L_B)
xsB, ysB = build_distribution(x_B0, equal=True)#, weights=all_std)
# project_B.fit(remap_distribution(x_B0, xsB, ysB))
project_B.fit(x_B0)

x_A = project_A.forward(x_A0)
# avgA = np.average(project_A.forward(x_A0), axis=(0,1,2))
# avgA_B = np.average(project_B.forward(x_A0), axis=(0,1,2))
# stdA = np.std(project_A.forward(x_A0), axis=(0,1,2))
# stdA_B = np.std(project_B.forward(x_A0), axis=(0,1,2))
# print(avgA)
# print(stdA)
normA = np.sqrt(np.average(np.square(np.linalg.norm(x_A, axis=-1))))
# print(normA)

x_B = project_B.forward(x_B0)
# avgB = np.average(project_B.forward(x_B0), axis=(0,1,2))
# avgB_A = np.average(project_A.forward(x_B0), axis=(0,1,2))
# stdB = np.std(project_B.forward(x_B0), axis=(0,1,2))
# stdB_A = np.std(project_A.forward(x_B0), axis=(0,1,2))
# print(avgB)
# print(stdB)
normB = np.sqrt(np.average(np.square(np.linalg.norm(x_B, axis=-1))))
# print(normB)

# x_A = remap_distribution(x_A0, xsA, ysA)
# x_A = remap_distribution(x_A, ysB, xsB)
x_A_B = project_B.forward(x_A0)
print("x_A->B: {}".format(x_A_B.shape))
xsA_B, ysA_B = build_distribution(x_A_B, equal=True)
# x_A = project_B.backward(x_A_B)

# x_B = remap_distribution(x_B0, xsB, ysB)
# x_B = remap_distribution(x_B, ysA, xsA)
x_B_A = project_A.forward(x_B0)
print("x_B->A: {}".format(x_B_A.shape))
xsB_A, ysB_A = build_distribution(x_B_A, equal=True)
# x_B = project_A.backward(x_B_A)

xsA_A, ysA_A = build_distribution(project_A.forward(x_A0), equal=True)
xsB_B, ysB_B = build_distribution(project_B.forward(x_B0), equal=True)

inp[:TEST_SIZE] = remap_distribution(inp[:TEST_SIZE], xsA0, ysA0)
inp[TEST_SIZE:] = remap_distribution(inp[TEST_SIZE:], xsB0, ysB0)

x = model.forward(inp)

print(x.shape)

x_A = x[:TEST_SIZE]
x_B = x[TEST_SIZE:]

x_A = remap_distribution(x_A, xsA, ysA)
x_A = remap_distribution(x_A, ysB, xsB)
# print("x_A->B: {}".format(project_B.forward(x_A).shape))
for i in range(1):
    x_A = project_B.forward(x_A)
    # x_A = (x_A - avgA_B) / stdA_B * stdB + avgB
    # x_A /= stdB
    # norm = np.linalg.norm(x_A, axis=-1, keepdims=True)
    # x_A = x_A / norm * normB
    # x_A = remap_distribution(x_A, xsA_B, ysA_B)
    # x_A = remap_distribution(x_A, ysB_B, xsB_B)
    # print(norm)
    # x_A *= stdB
    x_A = project_B.backward(x_A)
    # x_A -= avgB
    # x_A += avgB
# x_A = project_A.forward(x_A)
# x_A = project_B.backward(x_A)

x_B = remap_distribution(x_B, xsB, ysB)
x_B = remap_distribution(x_B, ysA, xsA)
# print("x_B->A: {}".format(project_A.forward(x_B).shape))
for i in range(1):
    x_B = project_A.forward(x_B)
    # x_B = (x_B - avgB_A) / stdB_A * stdA + avgA
    # x_B /= stdA
    # norm = np.linalg.norm(x_B, axis=-1, keepdims=True)
    # x_B = x_B / norm * normA
    # x_B = remap_distribution(x_B, xsB_A, ysB_A)
    # x_B = remap_distribution(x_B, ysA_A, xsA_A)
    # x_B *= stdA
    x_B = project_A.backward(x_B)
    # x_B -= avgA
    # x_B += avgA
# x_B = project_B.forward(x_B)
# x_B = project_A.backward(x_B)

x = np.concatenate([x_A, x_B], axis=0)

x = model.backward(x, BATCH_SIZE)

outp = x

inp[:TEST_SIZE] = remap_distribution(inp[:TEST_SIZE], ysA0, xsA0)
inp[TEST_SIZE:] = remap_distribution(inp[TEST_SIZE:], ysB0, xsB0)

outp[:TEST_SIZE] = remap_distribution(outp[:TEST_SIZE], ysB0, xsB0)
outp[TEST_SIZE:] = remap_distribution(outp[TEST_SIZE:], ysA0, xsA0)

f, ax = plt.subplots(2, 2*(TEST_SIZE), figsize=(10*TEST_SIZE, 10))

for i in range(2*(TEST_SIZE)):
    ax[0, i].imshow(inp[i])
    ax[1, i].imshow(outp[i])
plt.show()