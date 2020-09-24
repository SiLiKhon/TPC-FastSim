import tensorflow as tf
import yaml

from . import scalers, nn

@tf.function(experimental_relax_shapes=True)
def preprocess_features(features):
    # features:
    #   crossing_angle [-20, 20]
    #   dip_angle [-60, 60]
    #   drift_length [35, 290]
    #   pad_coordinate [40-something, 40-something]
    bin_fractions = features[:,-2:] % 1
    features = (
        features[:,:3] - tf.constant([[0., 0., 162.5]])
    ) / tf.constant([[20., 60., 127.5]])
    return tf.concat([features, bin_fractions], axis=-1)

_f = preprocess_features

def disc_loss(d_real, d_fake):
    return tf.reduce_mean(d_fake - d_real)


def gen_loss(d_real, d_fake):
    return tf.reduce_mean(d_real - d_fake)


def disc_loss_cramer(d_real, d_fake, d_fake_2):
    return -tf.reduce_mean(
        tf.norm(d_real - d_fake, axis=-1) +
        tf.norm(d_fake_2, axis=-1) - 
        tf.norm(d_fake - d_fake_2, axis=-1) -
        tf.norm(d_real, axis=-1)
    )

def gen_loss_cramer(d_real, d_fake, d_fake_2):
    return -disc_loss_cramer(d_real, d_fake, d_fake_2)

class Model_v4:
    def __init__(self, description_file='models/architectures/baseline_fc_8x16.yaml',
                 lr=1e-4, latent_dim=32, gp_lambda=10., num_disc_updates=8,
                 gpdata_lambda=0., cramer=False, stochastic_stepping=True):
        self.disc_opt = tf.keras.optimizers.RMSprop(lr)
        self.gen_opt = tf.keras.optimizers.RMSprop(lr)
        self.latent_dim = latent_dim
        self.gp_lambda = gp_lambda
        self.gpdata_lambda = gpdata_lambda
        self.num_disc_updates = num_disc_updates
        self.cramer = cramer
        self.stochastic_stepping = stochastic_stepping

        with open(description_file, 'r') as f:
            architecture_descr = yaml.load(f, Loader=yaml.FullLoader)
        self.generator = nn.build_architecture(architecture_descr['generator'])
        self.discriminator = nn.build_architecture(architecture_descr['discriminator'])

        self.step_counter = tf.Variable(0, dtype='int32', trainable=False)

        self.scaler = scalers.Logarithmic()
        self.pad_range = (-3, 5)
        self.time_range = (-7, 9)
        self.data_version = 'data_v4'


    @tf.function
    def make_fake(self, features):
        size = tf.shape(features)[0]
        latent_input = tf.random.normal(shape=(size, self.latent_dim), dtype='float32')
        return self.generator(
            tf.concat([_f(features), latent_input], axis=-1)
        )

    def gradient_penalty(self, features, real, fake):
        alpha = tf.random.uniform(shape=[len(real), 1, 1])
        interpolates = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as t:
            t.watch(interpolates)
            d_int = self.discriminator([_f(features), interpolates])

        grads = tf.reshape(t.gradient(d_int, interpolates), [len(real), -1])
        return tf.reduce_mean(tf.maximum(tf.norm(grads, axis=-1) - 1, 0)**2)

    def gradient_penalty_on_data(self, features, real):
        with tf.GradientTape() as t:
            t.watch(real)
            d_real = self.discriminator([_f(features), real])

        grads = tf.reshape(t.gradient(d_real, real), [len(real), -1])
        return tf.reduce_mean(tf.reduce_sum(grads**2, axis=-1))

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
            if tf.random.uniform(
                shape=[], dtype='int32',
                maxval=self.num_disc_updates + 1
            ) == self.num_disc_updates:
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
