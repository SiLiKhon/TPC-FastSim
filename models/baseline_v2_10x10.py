import tensorflow as tf


def get_generator(activation, kernel_init, num_features, latent_dim):
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation=activation, input_shape=(num_features + latent_dim,)),

        tf.keras.layers.Reshape((4, 4, 4)),

        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.UpSampling2D(),  # 8x8

        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same' , activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation=activation, kernel_initializer=kernel_init),  # 6x6
        tf.keras.layers.UpSampling2D(),  # 12x12

        tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='valid', activation=activation, kernel_initializer=kernel_init),  # 10x10
        tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='valid', activation=tf.keras.activations.relu, kernel_initializer=kernel_init),

        tf.keras.layers.Reshape((10, 10)),
    ], name='generator')
    return generator


def get_discriminator(activation, kernel_init, dropout_rate, num_features):
    discriminator_tail = tf.keras.Sequential([
        tf.keras.layers.Reshape((10, 10, 1), input_shape=(10, 10)),

        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation=activation, kernel_initializer=kernel_init),  # 8x8
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.MaxPool2D(),  # 4x4

        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.MaxPool2D(),  # 2x2

        tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='valid', activation=activation, kernel_initializer=kernel_init),  # 1x1
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Reshape((64,))
    ], name='discriminator_tail')

    features_input = tf.keras.Input(shape=(num_features,))
    head_input = tf.keras.layers.Concatenate()([features_input, discriminator_tail.output])

    discriminator_head = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, activation=activation, input_shape=(num_features + 64,)),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Dense(units=1, activation=None),
    ], name='discriminator_head')

    inputs = [features_input, discriminator_tail.input]
    outputs = discriminator_head(head_input)

    discriminator = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name='discriminator'
    )

    return discriminator


def disc_loss(d_real, d_fake):
    return tf.reduce_mean(d_fake - d_real)


def gen_loss(d_real, d_fake):
    return tf.reduce_mean(d_real - d_fake)


class BaselineModel10x10:
    def __init__(self, activation=tf.keras.activations.relu, kernel_init='glorot_uniform',
                 dropout_rate=0.2, lr=1e-4, latent_dim=32, gp_lambda=10., num_disc_updates=3,
                 num_features=1, gpdata_lambda=0.):
        self.disc_opt = tf.keras.optimizers.RMSprop(lr)
        self.gen_opt = tf.keras.optimizers.RMSprop(lr)
        self.latent_dim = latent_dim
        self.gp_lambda = gp_lambda
        self.gpdata_lambda = gpdata_lambda
        self.num_disc_updates = num_disc_updates
        self.num_features = num_features

        self.generator = get_generator(
            activation=activation, kernel_init=kernel_init, latent_dim=latent_dim, num_features=num_features
        )
        self.discriminator = get_discriminator(
            activation=activation, kernel_init=kernel_init, dropout_rate=dropout_rate, num_features=num_features
        )

        self.step_counter = tf.Variable(0, dtype='int32', trainable=False)

    def make_fake(self, features):
        size = tf.shape(features)[0]
        latent_input = tf.random.normal(shape=(size, self.latent_dim), dtype='float32')
        return self.generator(
            tf.concat([features, latent_input], axis=-1)
        )

    def gradient_penalty(self, features, real, fake):
        alpha = tf.random.uniform(shape=[len(real), 1, 1])
        interpolates = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as t:
            t.watch(interpolates)
            d_int = self.discriminator([features, interpolates])
        grads = tf.reshape(t.gradient(d_int, interpolates), [len(real), -1])
        return tf.reduce_mean(tf.maximum(tf.norm(grads, axis=-1) - 1, 0)**2)

    def gradient_penalty_on_data(self, features, real):
        with tf.GradientTape() as t:
            t.watch(real)
            d_real = self.discriminator([features, real])
        grads = tf.reshape(t.gradient(d_real, real), [len(real), -1])
        return tf.reduce_mean(tf.reduce_sum(grads**2, axis=-1))

    @tf.function
    def calculate_losses(self, feature_batch, target_batch):
        fake = self.make_fake(feature_batch)
        d_real = self.discriminator([feature_batch, target_batch])
        d_fake = self.discriminator([feature_batch, fake])

        d_loss = disc_loss(d_real, d_fake)

        if self.gp_lambda > 0:
            d_loss = (
                d_loss +
                self.gradient_penalty(
                    feature_batch, target_batch, fake
                ) * self.gp_lambda
            )
        if self.gpdata_lambda > 0:
            d_loss = (
                d_loss +
                self.gradient_penalty_on_data(
                    feature_batch, target_batch
                ) * self.gpdata_lambda
            )

        g_loss = gen_loss(d_real, d_fake)
        return {'disc_loss': d_loss, 'gen_loss': g_loss}

    def disc_step(self, feature_batch, target_batch):
        feature_batch = tf.convert_to_tensor(feature_batch)
        target_batch = tf.convert_to_tensor(target_batch)

        with tf.GradientTape() as t:
            losses = self.calculate_losses(feature_batch, target_batch)

        grads = t.gradient(losses['disc_loss'], self.discriminator.trainable_variables)
        self.disc_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return losses

    def gen_step(self, feature_batch, target_batch):
        feature_batch = tf.convert_to_tensor(feature_batch)
        target_batch = tf.convert_to_tensor(target_batch)

        with tf.GradientTape() as t:
            losses = self.calculate_losses(feature_batch, target_batch)

        grads = t.gradient(losses['gen_loss'], self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(grads, self.generator.trainable_variables))
        return losses

    @tf.function
    def training_step(self, feature_batch, target_batch):
        if self.step_counter == self.num_disc_updates:
            result = self.gen_step(feature_batch, target_batch)
            self.step_counter.assign(0)
        else:
            result = self.disc_step(feature_batch, target_batch)
            self.step_counter.assign_add(1)
        return result
