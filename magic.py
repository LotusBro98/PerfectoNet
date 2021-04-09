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

EPS = [0.15] * 6

EPS_L_A = 0.3
EPS_L_B = 0.3


SIZE = 2**(N_LAYERS + 1) - 1

DISTRIBUTION_APPROX_N = 64

def _forward_2_order(x):
    xd = x
    xp = [tf.math.asin(xd[:, :, :, i] / tf.linalg.norm(xd[:, :, :, i:], axis=-1)) for i in
          range(0, xd.shape[-1] - 1)]
    xp = xp + [tf.linalg.norm(xd, axis=-1) * tf.sign(xd[:, :, :, -1])]
    xp = tf.stack(xp, axis=-1)

    return xp

def _backward_2_order(x):
    r = x[:, :, :, -1]
    f = x[:, :, :, :-1]
    xd = [tf.abs(r) * tf.sin(f[:, :, :, i]) * tf.reduce_prod(tf.abs(tf.cos(f[:, :, :, :i])), axis=-1) for i in
          range(0, f.shape[-1])]
    xd = xd + [r * tf.reduce_prod(tf.abs(tf.cos(f[:, :, :, :])), axis=-1)]
    xd = tf.stack(xd, axis=-1)

    return xd

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


model = Model(N_LAYERS, EPS, approx_n=DISTRIBUTION_APPROX_N)
model.load()

xsA0 = np.load("saved/xsA0.npy")
ysA0 = np.load("saved/ysA0.npy")
xsB0 = np.load("saved/xsB0.npy")
ysB0 = np.load("saved/ysB0.npy")

x_A0 = np.load("saved/x_A0.npy")
x_B0 = np.load("saved/x_B0.npy")

# project_A = Layer(ksize=1, stride=1, eps=EPS_L_A)
# project_A.fit((x_A0))
#
# project_B = Layer(ksize=1, stride=1, eps=EPS_L_B)
# project_B.fit((x_B0))
#
# x_A = project_A.forward(x_A0)
# normA = np.average(np.linalg.norm(x_A, axis=-1))
# avgA = np.average(project_A.forward(x_A0), axis=(0,1,2))
# avgA_B = np.average(project_B.forward(x_A0), axis=(0,1,2))
# stdA = np.std(project_A.forward(x_A0), axis=(0,1,2))
# stdA_B = np.std(project_B.forward(x_A0), axis=(0,1,2))
#
# x_B = project_B.forward(x_B0)
# normB = np.average(np.linalg.norm(x_B, axis=-1))
# avgB = np.average(project_B.forward(x_B0), axis=(0,1,2))
# avgB_A = np.average(project_A.forward(x_B0), axis=(0,1,2))
# stdB = np.std(project_B.forward(x_B0), axis=(0,1,2))
# stdB_A = np.std(project_A.forward(x_B0), axis=(0,1,2))
#
# x_A_B = project_B.forward(x_A0)
# print("x_A->B: {}".format(x_A_B.shape))
#
# x_B_A = project_A.forward(x_B0)
# print("x_B->A: {}".format(x_B_A.shape))
#
# xsA_A, ysA_A = build_distribution(project_A.forward(x_A0), equal=True)
# xsB_B, ysB_B = build_distribution(project_B.forward(x_B0), equal=True)

inp[:TEST_SIZE] = remap_distribution(inp[:TEST_SIZE], xsA0, ysA0)
inp[TEST_SIZE:] = remap_distribution(inp[TEST_SIZE:], xsB0, ysB0)

x = model.forward(inp)

# print(x.shape)
#
# x_A = x[:TEST_SIZE]
# x_B = x[TEST_SIZE:]
#
# # x_A = _forward_2_order(x_A)
# x_A = project_B.forward(x_A)
# x_A = (x_A - avgA_B) / stdA_B * stdB + avgB
# # x_A = x_A / np.linalg.norm(x_A, axis=-1, keepdims=True) * normB
# x_A = project_B.backward(x_A)
# # x_A = _backward_2_order(x_A)
#
#
# # x_B = _forward_2_order(x_B)
# x_B = project_A.forward(x_B)
# x_B = (x_B - avgB_A) / stdB_A * stdA + avgA
# # x_B = x_B / np.linalg.norm(x_B, axis=-1, keepdims=True) * normA
# x_B = project_A.backward(x_B)
# # x_B = _backward_2_order(x_B)
#
# x = np.concatenate([x_A, x_B], axis=0)

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