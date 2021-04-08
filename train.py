import glob
import random

from distribution import *
from model import Model


DATASET_SIZE = 20
BATCH_SIZE = 4
TEST_SIZE = 4

N_LAYERS = 6
#
# EPS_START = 2e-2
# EPS_END = 2e-2
EPS = [0.1, 0.3, 0.3, 0.3, 0.3, 0.3]
# EPS = np.exp(np.linspace(np.log(0.05), np.log(1), N_LAYERS))
# EPS = [0.1] * 6


SIZE = 2**(N_LAYERS + 1) - 1

DISTRIBUTION_APPROX_N = 32


def load_image(filename, image_size=512):
    image = tf.io.decode_image(tf.io.read_file(filename), channels=3)
    image = image / 255
    image.set_shape((image_size, image_size, 3))
    image = tf.image.resize(image, (SIZE, SIZE))

    return image


images = []
pathsA = list(glob.glob("trainA/*.png"))
pathsB = list(glob.glob("trainA/*.png"))
random.shuffle(pathsA)
random.shuffle(pathsB)
paths = pathsA[:DATASET_SIZE] + pathsB[:DATASET_SIZE]
# paths = pathsA[:2*DATASET_SIZE]
random.shuffle(paths)

for path in paths:
    image = load_image(path)
    image = image.numpy()
    images.append(image)
images = np.stack(images)

inp = images

inp_A0 = inp[:DATASET_SIZE]
inp_B0 = inp[DATASET_SIZE:]
xsA0, ysA0 = build_distribution(inp_A0, equal=True)
xsB0, ysB0 = build_distribution(inp_B0, equal=True)
inp_A0 = remap_distribution(inp_A0, xsA0, ysA0)
inp_B0 = remap_distribution(inp_B0, xsB0, ysB0)

inp = np.concatenate([inp_A0, inp_B0])

model = Model(N_LAYERS, EPS, approx_n=DISTRIBUTION_APPROX_N)
x = model.fit(inp, BATCH_SIZE)
model.save()

# print(x)

x_A0 = model.forward(inp_A0)
x_B0 = model.forward(inp_B0)

np.save("saved/x_A0", x_A0)
np.save("saved/x_B0", x_B0)

np.save("saved/xsA0", xsA0)
np.save("saved/ysA0", ysA0)
np.save("saved/xsB0", xsB0)
np.save("saved/ysB0", ysB0)

# x = np.concatenate([x_A0, x_B0], axis=0)
x = model.backward(x)

outp = x

inp[:DATASET_SIZE] = remap_distribution(inp[:DATASET_SIZE], ysA0, xsA0)
inp[DATASET_SIZE:] = remap_distribution(inp[DATASET_SIZE:], ysB0, xsB0)

outp[:DATASET_SIZE] = remap_distribution(outp[:DATASET_SIZE], ysA0, xsA0)
outp[DATASET_SIZE:] = remap_distribution(outp[DATASET_SIZE:], ysB0, xsB0)

f, ax = plt.subplots(2, 2*(TEST_SIZE), figsize=(10*TEST_SIZE, 10))

for i in range(TEST_SIZE):
    ax[0, i].imshow(inp[i])
    ax[1, i].imshow(outp[i])

for i in range(0, TEST_SIZE):
    ax[0, TEST_SIZE + i].imshow(inp[DATASET_SIZE + i])
    ax[1, TEST_SIZE + i].imshow(outp[DATASET_SIZE + i])

plt.show()
