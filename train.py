import glob
import random

from distribution import *
from model import Model


DATASET_SIZE = 1800
BATCH_SIZE = 4
TEST_SIZE = 4

N_LAYERS = 6

EPS_START = 0.1
EPS_END = 0.2


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
paths = pathsA[:DATASET_SIZE] + pathsB[:DATASET_SIZE]
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

inp = np.concatenate([inp_A0, inp_B0])

model = Model(N_LAYERS, EPS_START, EPS_END, approx_n=DISTRIBUTION_APPROX_N)
model.fit(inp, BATCH_SIZE)
model.save()

x_A0 = model.forward(inp_A0)
x_B0 = model.forward(inp_B0)

np.save("saved/x_A0", x_A0)
np.save("saved/x_B0", x_B0)

np.save("saved/xsA0", xsA0)
np.save("saved/ysA0", ysA0)
np.save("saved/xsB0", xsB0)
np.save("saved/ysB0", ysB0)
