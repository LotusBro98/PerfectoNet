import glob
import random
import sys
import time

import tensorflow as tf

from distribution import *
from layer import Layer
from model import Model

SIZE = 63
DATASET_SIZE = 1000
BATCH_SIZE = 1

N_LAYERS = 1

EPS = [-1.5] * N_LAYERS


def load_image(filename, image_size=512):
    image = tf.io.decode_image(tf.io.read_file(filename), channels=3)
    image = image / 255
    image.set_shape((image_size, image_size, 3))
    image = tf.image.resize(image, (SIZE, SIZE))

    return image


pathsA = list(glob.glob("trainA/*.png"))[:DATASET_SIZE]
pathsB = list(glob.glob("trainB/*.png"))[:DATASET_SIZE]
random.shuffle(pathsA)
random.shuffle(pathsB)
paths = pathsA + pathsB
random.shuffle(paths)

images = []
for i, path in enumerate(paths):
    images.append(load_image(path).numpy())
    print(f"\rLoad {i}/{len(paths)}", end='', flush=True)
print()
images = np.stack(images)

imagesA = []
for i, path in enumerate(pathsA):
    imagesA.append(load_image(path).numpy())
    print(f"\rLoad A {i}/{len(pathsA)}", end='', flush=True)
print()
imagesA = np.stack(imagesA)

imagesB = []
for i, path in enumerate(pathsB):
    imagesB.append(load_image(path).numpy())
    print(f"\rLoad B {i}/{len(pathsB)}", end='', flush=True)
print()
imagesB = np.stack(imagesB)

##################### Encoder model #########################

model = Model(N_LAYERS, EPS)
model.fit(images, batch_size=BATCH_SIZE)

################ CNN models ######################

def DownBlock(Cout, use_batchnorm=True, strides=2):
    initializer = tf.random_normal_initializer(0., 0.02)
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2D(Cout, 3, strides=strides, padding='valid', use_bias=False, kernel_initializer=initializer))
    if use_batchnorm:
        block.add(tf.keras.layers.BatchNormalization())
    block.add(tf.keras.layers.LeakyReLU())
    return block

def UpBlock(Cout, use_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2DTranspose(Cout, 3, strides=2, padding='valid', use_bias=False, kernel_initializer=initializer))
    block.add(tf.keras.layers.BatchNormalization())
    if use_dropout:
        block.add(tf.keras.layers.Dropout(0.5))
    block.add(tf.keras.layers.ReLU())
    return block

##################### Generator model ######################

def Generator():
    inp = tf.keras.layers.Input((SIZE, SIZE, 3))
    x = inp

    print(x.shape)

    down_stack = [
        DownBlock(64, use_batchnorm=False),
        DownBlock(128),
        DownBlock(256),
        DownBlock(512),
        DownBlock(512),
        DownBlock(512),
    ]

    up_stack = [
        UpBlock(512, use_dropout=True),
        UpBlock(512, use_dropout=True),
        UpBlock(256),
        UpBlock(128),
        UpBlock(64),
        UpBlock(32),
    ]

    skips = []
    for down in down_stack:
        skips.append(x)
        x = down(x)
        print(x.shape)

    skips = reversed(skips)

    for x1, up in zip(skips, up_stack):
        x = up(x)
        print(x.shape, x1.shape)
        x = tf.concat([x, x1], axis=-1)

    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', kernel_initializer=initializer)(x)
    print(x.shape)

    return tf.keras.Model(inputs=inp, outputs=x)

print("Generator")
generator = Generator()

##################### Discriminator model ##################

def Discriminator():
    inp = tf.keras.layers.Input((SIZE, SIZE, 3))
    x = inp

    print(x.shape)

    down_stack = [
        DownBlock(64, use_batchnorm=False),
        DownBlock(128),
        DownBlock(256),
        DownBlock(512, strides=1),
        # DownBlock(512),
        # DownBlock(512),
    ]

    for down in down_stack:
        x = down(x)
        print(x.shape)

    x = tf.keras.layers.Conv2D(1, (3, 3))(x)
    print(x.shape)

    return tf.keras.Model(inputs=inp, outputs=x)

print("Discriminator")
discriminator = Discriminator()

################# Losses #######################

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse_loss = tf.keras.losses.MeanSquaredError()

def generator_loss(enc_gen_input, enc_gen_output, disc_gen_output):
    L1 = mse_loss(enc_gen_input, enc_gen_output)
    Lg = loss_object(tf.ones_like(disc_gen_output), disc_gen_output)
    L1 = tf.reduce_mean(L1)
    Lg = tf.reduce_mean(Lg)
    return 1*L1 + Lg
    # return Lg

def discriminator_loss(disc_gen_output, disc_real_output):
    Lt = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    Lf = loss_object(tf.zeros_like(disc_gen_output), disc_gen_output)
    return Lt + Lf

############# Train #####################

generator_optimizer = tf.keras.optimizers.Adam(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step(xA, xB):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(xA, training=True)

        enc_gen_output = model.forward(gen_output)
        enc_gen_input = model.forward(xA)

        disc_real_output = discriminator(xB, training=True)
        disc_gen_output = discriminator(gen_output, training=True)

        gen_loss = generator_loss(enc_gen_input, enc_gen_output, disc_gen_output)
        disc_loss = discriminator_loss(disc_gen_output, disc_real_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return tf.reduce_mean(gen_loss), tf.reduce_mean(disc_loss)

def train(datasetA, datasetB, epochs, batch_size=BATCH_SIZE):
    datasetA = np.stack([datasetA[i: i + batch_size] for i in range(0, len(datasetA), batch_size)][1:-1], axis=0)
    datasetB = np.stack([datasetB[i: i + batch_size] for i in range(0, len(datasetB), batch_size)][1:-1], axis=0)
    for epoch in range(epochs):
        start = time.time()

        for xA, xB in zip(datasetA, datasetB):
            gen_loss, disc_loss = train_step(xA, xB)

        print ('\rTime for epoch {} is {} sec. Lg {}, Ld {}'.format(epoch + 1, time.time()-start, gen_loss, disc_loss), end='', flush=True)

        if epoch % 10 == 0:
            x = generator(datasetA[0], training=True)
            f, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(datasetA[0, 0])
            ax[1].imshow(x[0])
            plt.show()

train(imagesB, imagesA, 100000)

################## Test #####################

# x = model.backward(x, BATCH_SIZE)
x = generator(imagesA)
f, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(imagesA[0])
ax[1].imshow(x[0])
plt.show()
