import tensorflow as tf


@tf.function(experimental_relax_shapes=True)
def preprocess_features(features):
    # features:
    #   crossing_angle [-20, 20]
    #   dip_angle [-60, 60]
    #   drift_length [35, 290]
    #   pad_coordinate [40-something, 40-something]
    bin_fractions = features[:, -2:] % 1
    features = (features[:, :3] - tf.constant([[0.0, 0.0, 162.5]])) / tf.constant([[20.0, 60.0, 127.5]])
    return tf.concat([features, bin_fractions], axis=-1)


_f = preprocess_features


def get_generator(activation, kernel_init, num_features, latent_dim):
    generator = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=64, activation=activation, input_shape=(num_features + latent_dim,)),
            tf.keras.layers.Reshape((2, 4, 8)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=2, padding='same', activation=activation, kernel_initializer=kernel_init
            ),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=2, padding='same', activation=activation, kernel_initializer=kernel_init
            ),  # 2x4
            tf.keras.layers.UpSampling2D(),  # 4x8
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=(2, 3), padding='valid', activation=activation, kernel_initializer=kernel_init
            ),  # 3x6
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=(1, 2), padding='valid', activation=activation, kernel_initializer=kernel_init
            ),  # 3x5
            tf.keras.layers.UpSampling2D(),  # 6x10
            tf.keras.layers.Conv2D(
                filters=8, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init
            ),
            tf.keras.layers.Conv2D(
                filters=8, kernel_size=2, padding='valid', activation=activation, kernel_initializer=kernel_init
            ),  # 5x9
            tf.keras.layers.UpSampling2D(),  # 10x18
            tf.keras.layers.Conv2D(
                filters=8, kernel_size=3, padding='valid', activation=activation, kernel_initializer=kernel_init
            ),  # 8x16
            tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=1,
                padding='valid',
                activation=tf.keras.activations.relu,
                kernel_initializer=kernel_init,
            ),
            tf.keras.layers.Reshape((8, 16)),
        ],
        name='generator',
    )
    return generator


def get_discriminator(
    activation, kernel_init, dropout_rate, num_features, num_additional_layers, cramer=False, features_to_tail=False
):
    input_img = tf.keras.Input(shape=(8, 16))
    features_input = tf.keras.Input(shape=(num_features,))

    img = tf.reshape(input_img, (-1, 8, 16, 1))
    if features_to_tail:
        features_tiled = tf.tile(tf.reshape(features_input, (-1, 1, 1, num_features)), (1, 8, 16, 1))
        img = tf.concat([img, features_tiled], axis=-1)

    discriminator_tail = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init
            ),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init
            ),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.MaxPool2D(pool_size=(1, 2)),  # 8x8
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init
            ),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_init
            ),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.MaxPool2D(),  # 4x4
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, padding='valid', activation=activation, kernel_initializer=kernel_init
            ),  # 2x2
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=2, padding='valid', activation=activation, kernel_initializer=kernel_init
            ),  # 1x1
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Reshape((64,)),
        ],
        name='discriminator_tail',
    )

    head_input = tf.keras.layers.Concatenate()([features_input, discriminator_tail(img)])

    head_layers = [
        tf.keras.layers.Dense(units=128, activation=activation, input_shape=(num_features + 64,)),
        tf.keras.layers.Dropout(dropout_rate),
    ]
    for _ in range(num_additional_layers):
        head_layers += [
            tf.keras.layers.Dense(units=128, activation=activation),
            tf.keras.layers.Dropout(dropout_rate),
        ]

    discriminator_head = tf.keras.Sequential(
        head_layers + [tf.keras.layers.Dense(units=1 if not cramer else 256, activation=None)],
        name='discriminator_head',
    )

    inputs = [features_input, input_img]
    outputs = discriminator_head(head_input)

    discriminator = tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')

    return discriminator


def disc_loss(d_real, d_fake):
    return tf.reduce_mean(d_fake - d_real)


def gen_loss(d_real, d_fake):
    return tf.reduce_mean(d_real - d_fake)


def disc_loss_cramer(d_real, d_fake, d_fake_2):
    return -tf.reduce_mean(
        tf.norm(d_real - d_fake, axis=-1)
        + tf.norm(d_fake_2, axis=-1)
        - tf.norm(d_fake - d_fake_2, axis=-1)
        - tf.norm(d_real, axis=-1)
    )


def gen_loss_cramer(d_real, d_fake, d_fake_2):
    return -disc_loss_cramer(d_real, d_fake, d_fake_2)


class BaselineModel_8x16:
    def __init__(
        self,
        activation=tf.keras.activations.relu,
        kernel_init='glorot_uniform',
        dropout_rate=0.02,
        lr=1e-4,
        latent_dim=32,
        gp_lambda=10.0,
        num_disc_updates=8,
        gpdata_lambda=0.0,
        num_additional_layers=0,
        cramer=False,
        features_to_tail=True,
        stochastic_stepping=True,
    ):
        self.disc_opt = tf.keras.optimizers.RMSprop(lr)
        self.gen_opt = tf.keras.optimizers.RMSprop(lr)
        self.latent_dim = latent_dim
        self.gp_lambda = gp_lambda
        self.gpdata_lambda = gpdata_lambda
        self.num_disc_updates = num_disc_updates
        self.num_features = 5
        self.cramer = cramer
        self.stochastic_stepping = stochastic_stepping

        self.generator = get_generator(
            activation=activation, kernel_init=kernel_init, latent_dim=latent_dim, num_features=self.num_features
        )
        self.discriminator = get_discriminator(
            activation=activation,
            kernel_init=kernel_init,
            dropout_rate=dropout_rate,
            num_features=self.num_features,
            num_additional_layers=num_additional_layers,
            cramer=cramer,
            features_to_tail=features_to_tail,
        )

        self.step_counter = tf.Variable(0, dtype='int32', trainable=False)

    #        # compile the models with an arbitrary loss func for serializablility
    #        self.generator.compile(optimizer=self.gen_opt,
    #                               loss='mean_squared_error')
    #        self.discriminator.compile(optimizer=self.disc_opt,
    #                               loss='mean_squared_error')

    @tf.function
    def make_fake(self, features):
        size = tf.shape(features)[0]
        latent_input = tf.random.normal(shape=(size, self.latent_dim), dtype='float32')
        return self.generator(tf.concat([_f(features), latent_input], axis=-1))

    def gradient_penalty(self, features, real, fake):
        alpha = tf.random.uniform(shape=[len(real), 1, 1])
        interpolates = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as t:
            t.watch(interpolates)
            d_int = self.discriminator([_f(features), interpolates])
        #            if self.cramer:
        #                d_fake = self.discriminator([_f(features), interpolates])
        #                d_int = tf.norm(d_int - d_fake, axis=-1)
        grads = tf.reshape(t.gradient(d_int, interpolates), [len(real), -1])
        return tf.reduce_mean(tf.maximum(tf.norm(grads, axis=-1) - 1, 0) ** 2)

    def gradient_penalty_on_data(self, features, real):
        with tf.GradientTape() as t:
            t.watch(real)
            d_real = self.discriminator([_f(features), real])
        #            if self.cramer:
        #                d_real = tf.norm(d_real, axis=-1)
        grads = tf.reshape(t.gradient(d_real, real), [len(real), -1])
        return tf.reduce_mean(tf.reduce_sum(grads ** 2, axis=-1))

    @tf.function
    def calculate_losses(self, feature_batch, target_batch):
        fake = self.make_fake(feature_batch)
        d_real = self.discriminator([_f(feature_batch), target_batch])
        d_fake = self.discriminator([_f(feature_batch), fake])
        if self.cramer:
            fake_2 = self.make_fake(feature_batch)
            d_fake_2 = self.discriminator([_f(feature_batch), fake_2])

        if not self.cramer:
            d_loss = disc_loss(d_real, d_fake)
        else:
            d_loss = disc_loss_cramer(d_real, d_fake, d_fake_2)

        if self.gp_lambda > 0:
            d_loss = d_loss + self.gradient_penalty(feature_batch, target_batch, fake) * self.gp_lambda
        if self.gpdata_lambda > 0:
            d_loss = d_loss + self.gradient_penalty_on_data(feature_batch, target_batch) * self.gpdata_lambda
        if not self.cramer:
            g_loss = gen_loss(d_real, d_fake)
        else:
            g_loss = gen_loss_cramer(d_real, d_fake, d_fake_2)

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
        if self.stochastic_stepping:
            if tf.random.uniform(shape=[], dtype='int32', maxval=self.num_disc_updates + 1) == self.num_disc_updates:
                result = self.gen_step(feature_batch, target_batch)
            else:
                result = self.disc_step(feature_batch, target_batch)
        else:
            if self.step_counter == self.num_disc_updates:
                result = self.gen_step(feature_batch, target_batch)
                self.step_counter.assign(0)
            else:
                result = self.disc_step(feature_batch, target_batch)
                self.step_counter.assign_add(1)
        return result
