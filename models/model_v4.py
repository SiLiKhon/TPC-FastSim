import h5py
import tensorflow as tf
from tensorflow.python.keras.saving import hdf5_format


from . import scalers, nn


@tf.function(experimental_relax_shapes=True)
def preprocess_features(features):
    # features:
    #   crossing_angle [-20, 20]
    #   dip_angle [-60, 60]
    #   drift_length [35, 290]
    #   pad_coordinate [40-something, 40-something]
    bin_fractions = features[:, 2:4] % 1
    features = (features[:, :3] - tf.constant([[0.0, 0.0, 162.5]])) / tf.constant([[20.0, 60.0, 127.5]])
    return tf.concat([features, bin_fractions], axis=-1)


@tf.function(experimental_relax_shapes=True)
def preprocess_features_v4plus(features):
    # features:
    #   crossing_angle [-20, 20]
    #   dip_angle [-60, 60]
    #   drift_length [35, 290]
    #   pad_coordinate [40-something, 40-something]
    #   padrow {23, 33}
    #   pT [0, 2.5]
    bin_fractions = features[:, 2:4] % 1
    features_1 = (features[:, :3] - tf.constant([[0.0, 0.0, 162.5]])) / tf.constant([[20.0, 60.0, 127.5]])
    features_2 = tf.cast(features[:, 4:5] >= 27, tf.float32)
    features_3 = features[:, 5:6] / 2.5
    return tf.concat([features_1, features_2, features_3, bin_fractions], axis=-1)


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


def logloss(x):
    return tf.nn.softplus(-x)


def disc_loss_js(d_real, d_fake):
    return tf.reduce_sum(logloss(d_real)) + tf.reduce_sum(logloss(-d_fake)) / (len(d_real) + len(d_fake))


def gen_loss_js(d_real, d_fake):
    return tf.reduce_mean(logloss(d_fake))


class Model_v4:
    def __init__(self, config):
        self._f = preprocess_features
        if config['data_version'] == 'data_v4plus':
            self.full_feature_space = config.get('full_feature_space', False)
            self.include_pT_for_evaluation = config.get('include_pT_for_evaluation', False)
            if self.full_feature_space:
                self._f = preprocess_features_v4plus

        self.disc_opt = tf.keras.optimizers.RMSprop(config['lr_disc'])
        self.gen_opt = tf.keras.optimizers.RMSprop(config['lr_gen'])
        self.gp_lambda = config['gp_lambda']
        self.gpdata_lambda = config['gpdata_lambda']
        self.num_disc_updates = config['num_disc_updates']
        self.cramer = config['cramer']
        self.js = config.get('js', False)
        assert not (self.js and self.cramer)

        self.stochastic_stepping = config['stochastic_stepping']
        self.dynamic_stepping = config.get('dynamic_stepping', False)
        if self.dynamic_stepping:
            assert not self.stochastic_stepping
            self.dynamic_stepping_threshold = config['dynamic_stepping_threshold']

        self.latent_dim = config['latent_dim']

        architecture_descr = config['architecture']
        self.generator = nn.build_architecture(
            architecture_descr['generator'], custom_objects_code=config.get('custom_objects', None)
        )
        self.discriminator = nn.build_architecture(
            architecture_descr['discriminator'], custom_objects_code=config.get('custom_objects', None)
        )

        self.step_counter = tf.Variable(0, dtype='int32', trainable=False)

        self.scaler = scalers.get_scaler(config['scaler'])
        self.pad_range = tuple(config['pad_range'])
        self.time_range = tuple(config['time_range'])
        self.data_version = config['data_version']

        self.generator.compile(optimizer=self.gen_opt, loss='mean_squared_error')
        self.discriminator.compile(optimizer=self.disc_opt, loss='mean_squared_error')

    def load_generator(self, checkpoint):
        self._load_weights(checkpoint, 'gen')

    def load_discriminator(self, checkpoint):
        self._load_weights(checkpoint, 'disc')

    def _load_weights(self, checkpoint, gen_or_disc):
        if gen_or_disc == 'gen':
            network = self.generator
            step_fn = self.gen_step
        elif gen_or_disc == 'disc':
            network = self.discriminator
            step_fn = self.disc_step
        else:
            raise ValueError(gen_or_disc)

        model_file = h5py.File(checkpoint, 'r')
        if len(network.optimizer.weights) == 0 and 'optimizer_weights' in model_file:
            # perform single optimization step to init optimizer weights
            features_shape = self.discriminator.inputs[0].shape.as_list()
            targets_shape = self.discriminator.inputs[1].shape.as_list()
            features_shape[0], targets_shape[0] = 1, 1
            step_fn(tf.zeros(features_shape), tf.zeros(targets_shape))

        print(f'Loading {gen_or_disc} weights from {str(checkpoint)}')
        network.load_weights(str(checkpoint))

        if 'optimizer_weights' in model_file:
            print('Also recovering the optimizer state')
            opt_weight_values = hdf5_format.load_optimizer_weights_from_hdf5_group(model_file)
            network.optimizer.set_weights(opt_weight_values)

    @tf.function
    def make_fake(self, features):
        size = tf.shape(features)[0]
        latent_input = tf.random.normal(shape=(size, self.latent_dim), dtype='float32')
        return self.generator(tf.concat([self._f(features), latent_input], axis=-1))

    def gradient_penalty(self, features, real, fake):
        alpha = tf.random.uniform(shape=[len(real)] + [1] * (len(real.shape) - 1))
        interpolates = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as t:
            t.watch(interpolates)
            d_int = self.discriminator([self._f(features), interpolates])

        grads = tf.reshape(t.gradient(d_int, interpolates), [len(real), -1])
        return tf.reduce_mean(tf.maximum(tf.norm(grads, axis=-1) - 1, 0) ** 2)

    def gradient_penalty_on_data(self, features, real):
        with tf.GradientTape() as t:
            t.watch(real)
            d_real = self.discriminator([self._f(features), real])

        grads = tf.reshape(t.gradient(d_real, real), [len(real), -1])
        return tf.reduce_mean(tf.reduce_sum(grads**2, axis=-1))

    @tf.function
    def calculate_losses(self, feature_batch, target_batch):
        fake = self.make_fake(feature_batch)
        d_real = self.discriminator([self._f(feature_batch), target_batch])
        d_fake = self.discriminator([self._f(feature_batch), fake])
        if self.cramer:
            fake_2 = self.make_fake(feature_batch)
            d_fake_2 = self.discriminator([self._f(feature_batch), fake_2])

        if not self.cramer:
            if self.js:
                d_loss = disc_loss_js(d_real, d_fake)
            else:
                d_loss = disc_loss(d_real, d_fake)
        else:
            d_loss = disc_loss_cramer(d_real, d_fake, d_fake_2)

        if self.gp_lambda > 0:
            d_loss = d_loss + self.gradient_penalty(feature_batch, target_batch, fake) * self.gp_lambda
        if self.gpdata_lambda > 0:
            d_loss = d_loss + self.gradient_penalty_on_data(feature_batch, target_batch) * self.gpdata_lambda
        if not self.cramer:
            if self.js:
                g_loss = gen_loss_js(d_real, d_fake)
            else:
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
                if self.dynamic_stepping:
                    if result['disc_loss'] < self.dynamic_stepping_threshold:
                        self.step_counter.assign(self.num_disc_updates)
                else:
                    self.step_counter.assign_add(1)
        return result
