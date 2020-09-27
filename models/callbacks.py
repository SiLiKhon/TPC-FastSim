import tensorflow as tf

from metrics import make_images_for_model

class SaveModelCallback:
    def __init__(self, model, path, save_period):
        self.model = model
        self.path = path
        self.save_period = save_period

    def __call__(self, step):
        if step % self.save_period == 0:
            print(f'Saving model on step {step} to {self.path}')
            self.model.generator.save(
                str(self.path.joinpath("generator_{:05d}.h5".format(step))))
            self.model.discriminator.save(
                str(self.path.joinpath("discriminator_{:05d}.h5".format(step))))


class WriteHistSummaryCallback:
    def __init__(self, model, sample, save_period, writer):
        self.model = model
        self.sample = sample
        self.save_period = save_period
        self.writer = writer

    def __call__(self, step):
        if step % self.save_period == 0:
            images, images1, img_amplitude, chi2 = make_images_for_model(self.model,
                                                                         sample=self.sample,
                                                                         calc_chi2=True)
            with self.writer.as_default():
                tf.summary.scalar("chi2", chi2, step)

                for k, img in images.items():
                    tf.summary.image(k, img, step)
                for k, img in images1.items():
                    tf.summary.image("{} (amp > 1)".format(k), img, step)
                tf.summary.image("log10(amplitude + 1)", img_amplitude, step)


class ScheduleLRCallback:
    def __init__(self, model, decay_rate, writer):
        self.model = model
        self.decay_rate = decay_rate
        self.writer = writer

    def __call__(self, step):
        self.model.disc_opt.lr.assign(self.model.disc_opt.lr * self.decay_rate)
        self.model.gen_opt.lr.assign(self.model.gen_opt.lr * self.decay_rate)
        with self.writer.as_default():
            tf.summary.scalar("discriminator learning rate", self.model.disc_opt.lr, step)
            tf.summary.scalar("generator learning rate", self.model.gen_opt.lr, step)