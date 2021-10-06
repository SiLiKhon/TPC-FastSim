import tensorflow as tf


def get_generator(activation, kernel_init, latent_dim):
    generator = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=64, activation=activation, input_shape=(latent_dim,)),
            tf.keras.layers.Reshape((4, 4, 4)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init
            ),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init
            ),
            tf.keras.layers.UpSampling2D(),  # 8x8
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init
            ),
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=3, padding='valid', activation=activation, kernel_initializer=kernel_init
            ),  # 6x6
            tf.keras.layers.UpSampling2D(),  # 12x12
            tf.keras.layers.Conv2D(
                filters=8, kernel_size=3, padding='valid', activation=activation, kernel_initializer=kernel_init
            ),  # 10x10
            tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=1,
                padding='valid',
                activation=tf.keras.activations.relu,
                kernel_initializer=kernel_init,
            ),
            tf.keras.layers.Reshape((10, 10)),
        ],
        name='generator',
    )
    return generator


def get_discriminator(activation, kernel_init, dropout_rate):
    discriminator = tf.keras.Sequential(
        [
            tf.keras.layers.Reshape((10, 10, 1), input_shape=(10, 10)),
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init
            ),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=3, padding='valid', activation=activation, kernel_initializer=kernel_init
            ),  # 8x8
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.MaxPool2D(),  # 4x4
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init
            ),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init
            ),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.MaxPool2D(),  # 2x2
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=2, padding='valid', activation=activation, kernel_initializer=kernel_init
            ),  # 1x1
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Reshape((64,)),
            tf.keras.layers.Dense(units=128, activation=activation),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(units=1, activation=None),
        ],
        name='discriminator',
    )
    return discriminator


def disc_loss(d_real, d_fake):
    return tf.reduce_mean(d_fake - d_real)


def gen_loss(d_real, d_fake):
    return tf.reduce_mean(d_real - d_fake)


class BaselineModel10x10:
    def __init__(
        self,
        activation=tf.keras.activations.relu,
        kernel_init='glorot_uniform',
        dropout_rate=0.2,
        lr=1e-4,
        latent_dim=32,
        gp_lambda=10.0,
        num_disc_updates=3,
    ):
        self.disc_opt = tf.keras.optimizers.RMSprop(lr)
        self.gen_opt = tf.keras.optimizers.RMSprop(lr)
        self.latent_dim = latent_dim
        self.gp_lambda = gp_lambda
        self.num_disc_updates = num_disc_updates

        self.generator = get_generator(activation=activation, kernel_init=kernel_init, latent_dim=latent_dim)
        self.discriminator = get_discriminator(
            activation=activation, kernel_init=kernel_init, dropout_rate=dropout_rate
        )

        self.step_counter = tf.Variable(0, dtype='int32', trainable=False)

    def make_fake(self, size):
        return self.generator(tf.random.normal(shape=(size, self.latent_dim), dtype='float32'))

    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform(shape=[len(real), 1, 1])
        interpolates = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as t:
            t.watch(interpolates)
            d_int = self.discriminator(interpolates)
        grads = tf.reshape(t.gradient(d_int, interpolates), [len(real), -1])
        return tf.reduce_mean(tf.maximum(tf.norm(grads, axis=-1) - 1, 0) ** 2)

    @tf.function
    def calculate_losses(self, batch):
        fake = self.make_fake(len(batch))
        d_real = self.discriminator(batch)
        d_fake = self.discriminator(fake)

        d_loss = disc_loss(d_real, d_fake) + self.gp_lambda * self.gradient_penalty(batch, fake)
        g_loss = gen_loss(d_real, d_fake)
        return {'disc_loss': d_loss, 'gen_loss': g_loss}

    def disc_step(self, batch):
        batch = tf.convert_to_tensor(batch)

        with tf.GradientTape() as t:
            losses = self.calculate_losses(batch)

        grads = t.gradient(losses['disc_loss'], self.discriminator.trainable_variables)
        self.disc_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return losses

    def gen_step(self, batch):
        batch = tf.convert_to_tensor(batch)

        with tf.GradientTape() as t:
            losses = self.calculate_losses(batch)

        grads = t.gradient(losses['gen_loss'], self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(grads, self.generator.trainable_variables))
        return losses

    @tf.function
    def training_step(self, batch):
        if self.step_counter == self.num_disc_updates:
            result = self.gen_step(batch)
            self.step_counter.assign(0)
        else:
            result = self.disc_step(batch)
            self.step_counter.assign_add(1)
        return result
