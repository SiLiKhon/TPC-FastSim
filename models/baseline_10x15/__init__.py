import tensorflow as tf
import numpy as np

LATENT_DIM = 32
activation = tf.keras.activations.elu
dropout_rate = 0.2

NUM_DISC_UPDATES = 2
GP_LAMBDA = 10.

generator = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation=activation, input_shape=(LATENT_DIM,)),

    tf.keras.layers.Dense(units=480, activation=activation, input_shape=(LATENT_DIM,)),
    tf.keras.layers.Reshape((3, 4, 40)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation),
    tf.keras.layers.UpSampling2D(), # 6x8

    tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=activation),
    tf.keras.layers.UpSampling2D(), # 12x16

    tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', activation=activation),
    tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 2), padding='valid', activation=activation),

    tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='valid', activation=activation),

    tf.keras.layers.Reshape((10, 15)),
], name='generator')

discriminator = tf.keras.Sequential([
    tf.keras.layers.Reshape((10, 15, 1), input_shape=(10, 15)),

    tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=activation),
    tf.keras.layers.Dropout(dropout_rate),

    tf.keras.layers.MaxPool2D(padding='same'), # 5x8

    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation),
    tf.keras.layers.Dropout(dropout_rate),

    tf.keras.layers.MaxPool2D(padding='same'), # 3x4

    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', activation=activation), # 1x2
    tf.keras.layers.Dropout(dropout_rate),

    tf.keras.layers.Reshape((128,)),

    tf.keras.layers.Dense(units=128, activation=activation),
    tf.keras.layers.Dropout(dropout_rate),

    tf.keras.layers.Dense(units=1, activation=activation),
], name='discriminator')



disc_opt = tf.optimizers.RMSprop()
gen_opt = tf.optimizers.RMSprop()

def make_fake(size):
    return generator(
        np.random.normal(size=(size, LATENT_DIM)).astype('float32')
    )

def disc_loss(d_real, d_fake):
    return tf.reduce_mean(d_fake - d_real)

def gen_loss(d_real, d_fake):
    return tf.reduce_mean(d_real - d_fake)

def gradient_penalty(real, fake):
    alpha = tf.random.uniform(shape=[len(real), 1, 1])
    interpolates = alpha * real + (1 - alpha) * fake
    with tf.GradientTape() as t:
        t.watch(interpolates)
        d_int = discriminator(interpolates)
    grads = tf.reshape(t.gradient(d_int, interpolates), [len(real), -1])
    return tf.reduce_mean(tf.maximum(tf.norm(grads, axis=-1) - 1, 0)**2)

def calculate_losses(batch):
    fake = make_fake(len(batch))
    d_real = discriminator(batch)
    d_fake = discriminator(fake)

    d_loss = disc_loss(d_real, d_fake) + GP_LAMBDA * gradient_penalty(batch, fake)
    g_loss = gen_loss(d_real, d_fake)
    return {'disc_loss' : d_loss, 'gen_loss' : g_loss}

def disc_step(batch):
    batch = tf.convert_to_tensor(batch)

    with tf.GradientTape() as t:
        losses = calculate_losses(batch)
    loss_vals = {k : l.numpy() for k, l in losses.items()}

    grads = t.gradient(losses['disc_loss'], discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(grads, discriminator.trainable_variables))
    return loss_vals

def gen_step(batch):
    batch = tf.convert_to_tensor(batch)

    with tf.GradientTape() as t:
        losses = calculate_losses(batch)
    loss_vals = {k : l.numpy() for k, l in losses.items()}

    grads = t.gradient(losses['gen_loss'], generator.trainable_variables)
    gen_opt.apply_gradients(zip(grads, generator.trainable_variables))
    return loss_vals

step_counter = tf.Variable(0, dtype='int32', trainable=False)

def training_step(batch):
    if step_counter == NUM_DISC_UPDATES:
        result = gen_step(batch)
        step_counter.assign(0)
    else:
        result = disc_step(batch)
        step_counter.assign_add(1)
    return result
