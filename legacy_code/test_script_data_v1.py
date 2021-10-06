import os
from pathlib import Path
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from data import preprocessing
from models.training import train
from models.baseline_10x10 import BaselineModel10x10
from metrics import make_metric_plots, make_histograms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_name', type=str, default='baseline_10x10', required=False)
    parser.add_argument('--batch_size', type=int, default=32, required=False)
    parser.add_argument('--lr', type=float, default=1e-4, required=False)
    parser.add_argument('--num_disc_updates', type=int, default=3, required=False)
    parser.add_argument('--lr_schedule_rate', type=float, default=0.998, required=False)
    parser.add_argument('--save_every', type=int, default=50, required=False)
    parser.add_argument('--num_epochs', type=int, default=10000, required=False)
    parser.add_argument('--latent_dim', type=int, default=32, required=False)
    parser.add_argument('--gpu_num', type=str, default=None, required=False)
    parser.add_argument('--kernel_init', type=str, default='glorot_uniform', required=False)

    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if args.gpu_num is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    assert len(logical_devices) > 0, "Not enough GPU hardware devices available"

    model_path = Path('saved_models') / args.checkpoint_name
    model_path.mkdir(parents=True)
    model = BaselineModel10x10(
        kernel_init=args.kernel_init, lr=args.lr, num_disc_updates=args.num_disc_updates, latent_dim=args.latent_dim
    )

    def save_model(step):
        if step % args.save_every == 0:
            print(f'Saving model on step {step} to {model_path}')
            model.generator.save(str(model_path.joinpath("generator_{:05d}.h5".format(step))))
            model.discriminator.save(str(model_path.joinpath("discriminator_{:05d}.h5".format(step))))

    preprocessing._VERSION = 'data_v1'
    data = preprocessing.read_csv_2d(pad_range=(39, 49), time_range=(266, 276))

    data_scaled = np.log10(1 + data).astype('float32')
    X_train, X_test = train_test_split(data_scaled, test_size=0.25, random_state=42)

    writer_train = tf.summary.create_file_writer(f'logs/{args.checkpoint_name}/train')
    writer_val = tf.summary.create_file_writer(f'logs/{args.checkpoint_name}/validation')

    def unscale(x):
        return 10 ** x - 1

    def write_hist_summary(step):
        if step % args.save_every == 0:
            gen_scaled = model.make_fake(len(X_test)).numpy()
            real = unscale(X_test)
            gen = unscale(gen_scaled)
            gen[gen < 0] = 0
            gen1 = np.where(gen < 1.0, 0, gen)
            images = make_metric_plots(real, gen)
            images1 = make_metric_plots(real, gen1)

            img_amplitude = make_histograms(X_test.flatten(), gen_scaled.flatten(), 'log10(amplitude + 1)', logy=True)

            with writer_val.as_default():
                for k, img in images.items():
                    tf.summary.image(k, img, step)
                for k, img in images1.items():
                    tf.summary.image("{} (amp > 1)".format(k), img, step)
                tf.summary.image("log10(amplitude + 1)", img_amplitude, step)

    def schedule_lr(step):
        model.disc_opt.lr.assign(model.disc_opt.lr * args.lr_schedule_rate)
        model.gen_opt.lr.assign(model.gen_opt.lr * args.lr_schedule_rate)
        with writer_val.as_default():
            tf.summary.scalar("discriminator learning rate", model.disc_opt.lr, step)
            tf.summary.scalar("generator learning rate", model.gen_opt.lr, step)

    train(
        X_train,
        X_test,
        model.training_step,
        model.calculate_losses,
        args.num_epochs,
        args.batch_size,
        train_writer=writer_train,
        val_writer=writer_val,
        callbacks=[write_hist_summary, save_model, schedule_lr],
    )


if __name__ == '__main__':
    main()
