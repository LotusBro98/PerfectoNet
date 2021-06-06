import glob
import random

from distribution import *
from model import Model


DATASET_SIZE = 20
BATCH_SIZE = 4
TEST_SIZE = 2

N_LAYERS = 2
SKIP_LAYERS = 0

EPS = [0.05, 0.05, 0.03, 0.03, 0.03, 0.05]
# EPS = np.exp(np.linspace(np.log(0.05), np.log(1), N_LAYERS))
# EPS = [0.15] * 6


# SIZE = 2**(N_LAYERS + 1) - 1
SIZE = 127

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

inp = np.concatenate([inp_A0, inp_B0])

model = Model(N_LAYERS, EPS, approx_n=DISTRIBUTION_APPROX_N)
x = model.fit(inp, BATCH_SIZE, skip_layers=SKIP_LAYERS)
model.save()

x_A0 = x[:DATASET_SIZE]
x_B0 = x[DATASET_SIZE:]

np.save("saved/x_A0", x_A0)
np.save("saved/x_B0", x_B0)

# x = np.concatenate([x_A0, x_B0], axis=0)
x = model.backward(x[:2*TEST_SIZE])

outp = x

f, ax = plt.subplots(2, 2*(TEST_SIZE), figsize=(10*TEST_SIZE, 10))

for i in range(2*TEST_SIZE):
    ax[0, i].imshow(inp[i])
    ax[1, i].imshow(outp[i])

plt.show()
