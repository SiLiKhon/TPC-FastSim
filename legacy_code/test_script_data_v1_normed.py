import os
from pathlib import Path

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np  #
from sklearn.model_selection import train_test_split
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from data import preprocessing
from models import training, baseline_10x10 as baseline_10x10_normed
from metrics import make_metric_plots

preprocessing._VERSION = 'data_v1'
data = preprocessing.read_csv_2d(pad_range=(39, 49), time_range=(266, 276))
data /= data.sum(axis=(1, 2), keepdims=True) / 100

data_scaled = np.log10(1 + data).astype('float32')
print("_" * 70)
print("MEAN IS:", data_scaled[data_scaled > 0].mean())
print("_" * 70)
X_train, X_test = train_test_split(data_scaled, test_size=0.25, random_state=42)

writer_train = tf.summary.create_file_writer('logs/baseline_10x10_normed/train')
writer_val = tf.summary.create_file_writer('logs/baseline_10x10_normed/validation')


def unscale(x):
    return 10**x - 1


def write_hist_summary(step):
    if step % 50 == 0:
        gen_scaled = baseline_10x10_normed.make_fake(len(X_test)).numpy()
        real = unscale(X_test)
        gen = unscale(gen_scaled)
        gen[gen < 0] = 0
        images = make_metric_plots(real, gen)

        with writer_val.as_default():
            for k, img in images.items():
                tf.summary.image(k, img, step)


model_path = Path("saved_models/baseline_10x10_normed/")
model_path.mkdir(parents=True)


def save_model(step):
    if step % 50 == 0:
        baseline_10x10_normed.generator.save(str(model_path.joinpath("generator_{:05d}.h5".format(step))))
        baseline_10x10_normed.discriminator.save(str(model_path.joinpath("discriminator_{:05d}.h5".format(step))))


def schedule_lr(step):
    baseline_10x10_normed.disc_opt.lr.assign(baseline_10x10_normed.disc_opt.lr * 0.998)
    baseline_10x10_normed.gen_opt.lr.assign(baseline_10x10_normed.gen_opt.lr * 0.998)
    with writer_val.as_default():
        tf.summary.scalar("discriminator learning rate", baseline_10x10_normed.disc_opt.lr, step)
        tf.summary.scalar("generator learning rate", baseline_10x10_normed.gen_opt.lr, step)


baseline_10x10_normed.disc_opt.lr.assign(0.0001)
baseline_10x10_normed.gen_opt.lr.assign(0.0001)

training.train(
    X_train,
    X_test,
    baseline_10x10_normed.training_step,
    baseline_10x10_normed.calculate_losses,
    10000,
    32,
    train_writer=writer_train,
    val_writer=writer_val,
    callbacks=[write_hist_summary, save_model, schedule_lr],
)
