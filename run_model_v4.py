import os, sys
import re
from pathlib import Path
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import PIL

from data import preprocessing
from models.training import train
from models.model_v4 import Model_v4
from metrics import make_metric_plots, make_histograms
import cuda_gpu_config

def make_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32, required=False)
    parser.add_argument('--lr', type=float, default=1e-4, required=False)
    parser.add_argument('--num_disc_updates', type=int, default=8, required=False)
    parser.add_argument('--lr_schedule_rate', type=float, default=0.999, required=False)
    parser.add_argument('--save_every', type=int, default=50, required=False)
    parser.add_argument('--num_epochs', type=int, default=10000, required=False)
    parser.add_argument('--latent_dim', type=int, default=32, required=False)
    parser.add_argument('--gpu_num', type=str, required=False)
    parser.add_argument('--gp_lambda', type=float, default=10., required=False)
    parser.add_argument('--gpdata_lambda', type=float, default=0., required=False)
    parser.add_argument('--cramer_gan', action='store_true', default=False)
    parser.add_argument('--prediction_only', action='store_true', default=False)
    parser.add_argument('--stochastic_stepping', action='store_true', default=True)
    parser.add_argument('--feature_noise_power', type=float, default=None)
    parser.add_argument('--feature_noise_decay', type=float, default=None)
    return parser


def print_args(args):
    print("")
    print("----" * 10)
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"    {k} : {v}")
    print("----" * 10)
    print("")


def parse_args():
    args = make_parser().parse_args()

    assert (
        (args.feature_noise_power is None) ==
        (args.feature_noise_decay is None)
    ), 'Noise power and decay must be both provided'

    print_args(args)

    return args


def write_args(model_path, fname='arguments.txt'):
    with open(model_path / fname, 'w') as f:
        raw_args = [a for a in sys.argv[1:] if a[0] != '@']
        fnames = [a[1:] for a in sys.argv[1:] if a[0] == '@']

        f.write('\n'.join(raw_args))
        f.write('\n')
        for fname in fnames:
            with open(fname, 'r') as f_in:
                f.write(f_in.read())


def epoch_from_name(name):
    epoch, = re.findall('\d+', name)
    return int(epoch)


def load_weights(model, model_path):
    gen_checkpoints = model_path.glob("generator_*.h5")
    disc_checkpoints = model_path.glob("discriminator_*.h5")
    latest_gen_checkpoint = max(
        gen_checkpoints,
        key=lambda path: epoch_from_name(path.stem)
    )
    latest_disc_checkpoint = max(
        disc_checkpoints,
        key=lambda path: epoch_from_name(path.stem)
    )

    assert (
        epoch_from_name(latest_gen_checkpoint.stem) == epoch_from_name(latest_disc_checkpoint.stem)
    ), "Latest disc and gen epochs differ"

    print(f'Loading generator weights from {str(latest_gen_checkpoint)}')
    model.generator.load_weights(str(latest_gen_checkpoint))
    print(f'Loading discriminator weights from {str(latest_disc_checkpoint)}')
    model.discriminator.load_weights(str(latest_disc_checkpoint))
    
    return latest_gen_checkpoint, latest_disc_checkpoint


def get_images(model,
               sample,
               return_raw_data=False,
               calc_chi2=False,
               gen_more=None,
               batch_size=128):
    X, Y = sample
    assert X.ndim == 2
    assert X.shape[1] == 4

    if gen_more is None:
        gen_features = X
    else:
        gen_features = np.tile(
            X,
            [gen_more] + [1] * (X.ndim - 1)
        )
    gen_scaled = np.concatenate([
        model.make_fake(gen_features[i:i+batch_size]).numpy()
        for i in range(0, len(gen_features), batch_size)
    ], axis=0)
    real = model.scaler.unscale(Y)
    gen = model.scaler.unscale(gen_scaled)
    gen[gen < 0] = 0
    gen1 = np.where(gen < 1., 0, gen)

    features = {
        'crossing_angle' : (X[:, 0], gen_features[:,0]),
        'dip_angle'      : (X[:, 1], gen_features[:,1]),
        'drift_length'   : (X[:, 2], gen_features[:,2]),
        'time_bin_fraction' : (X[:, 2] % 1, gen_features[:,2] % 1),
        'pad_coord_fraction' : (X[:, 3] % 1, gen_features[:,3] % 1)
    }

    images = make_metric_plots(real, gen, features=features, calc_chi2=calc_chi2)
    if calc_chi2:
        images, chi2 = images

    images1 = make_metric_plots(real, gen1, features=features)

    img_amplitude = make_histograms(Y.flatten(), gen_scaled.flatten(), 'log10(amplitude + 1)', logy=True)

    result = [images, images1, img_amplitude]

    if return_raw_data:
        result += [(gen_features, gen)]

    if calc_chi2:
        result += [chi2]

    return result


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
            images, images1, img_amplitude, chi2 = get_images(self.model,
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


def evaluate_model(model, path, sample, gen_sample_name=None):
    path.mkdir()
    (
        images, images1, img_amplitude,
        gen_dataset, chi2
    ) = get_images(model, sample=sample,
                   calc_chi2=True, return_raw_data=True, gen_more=10)

    array_to_img = lambda arr: PIL.Image.fromarray(arr.reshape(arr.shape[1:]))

    for k, img in images.items():
        array_to_img(img).save(str(path / f"{k}.png"))
    for k, img in images1.items():
        array_to_img(img).save(str(path / f"{k}_amp_gt_1.png"))
    array_to_img(img_amplitude).save(str(path / "log10_amp_p_1.png"))

    if gen_sample_name is not None:
        with open(str(path / gen_sample_name), 'w') as f:
            for event_X, event_Y in zip(*gen_dataset):
                f.write('params: {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(*event_X))
                for ipad, time_distr in enumerate(event_Y, model.pad_range[0] + event_X[3].astype(int)):
                    for itime, amp in enumerate(time_distr, model.time_range[0] + event_X[2].astype(int)):
                        if amp < 1:
                            continue
                        f.write(" {:2d} {:3d} {:8.3e} ".format(ipad, itime, amp))
                f.write('\n')

    with open(str(path / 'stats'), 'w') as f:
        f.write(f"{chi2:.2f}\n")


def main():
    args = parse_args()

    cuda_gpu_config.setup_gpu(args.gpu_num)

    model_path = Path('saved_models') / args.checkpoint_name

    if args.prediction_only:
        assert model_path.exists(), "Couldn't find model directory"
    else:
        assert not model_path.exists(), "Model directory already exists"
        model_path.mkdir(parents=True)

        write_args(model_path)

    model = Model_v4(lr=args.lr, latent_dim=args.latent_dim, gp_lambda=args.gp_lambda,
                     num_disc_updates=args.num_disc_updates, gpdata_lambda=args.gpdata_lambda,
                     cramer=args.cramer_gan, stochastic_stepping=args.stochastic_stepping)

    if args.prediction_only:
        latest_gen_checkpoint, latest_disc_checkpoint = load_weights(model, model_path)

    preprocessing._VERSION = model.data_version
    data, features = preprocessing.read_csv_2d(pad_range=model.pad_range, time_range=model.time_range)
    features = features.astype('float32')

    data_scaled = model.scaler.scale(data).astype('float32')

    Y_train, Y_test, X_train, X_test = train_test_split(data_scaled, features, test_size=0.25, random_state=42)

    if not args.prediction_only:
        writer_train = tf.summary.create_file_writer(f'logs/{args.checkpoint_name}/train')
        writer_val = tf.summary.create_file_writer(f'logs/{args.checkpoint_name}/validation')


    if args.prediction_only:
        prediction_path = model_path / f"prediction_{epoch_from_name(latest_gen_checkpoint.stem):05d}"
        assert not prediction_path.exists(), "Prediction path already exists"
        prediction_path.mkdir()

        for part in ['train', 'test']:
            evaluate_model(
                model, path=prediction_path / part,
                sample=(
                    (X_train, Y_train) if part == 'train'
                    else (X_test, Y_test)
                ),
                gen_sample_name=(None if part == 'train' else 'generated.dat')
            )

    else:
        features_noise = None
        if args.feature_noise_power is not None:
            def features_noise(epoch):
                current_power = args.feature_noise_power / (10**(epoch / args.feature_noise_decay))
                with writer_train.as_default():
                    tf.summary.scalar("features noise power", current_power, epoch)

                return current_power


        save_model = SaveModelCallback(
            model=model, path=model_path, save_period=args.save_every
        )
        write_hist_summary = WriteHistSummaryCallback(
            model, sample=(X_test, Y_test),
            save_period=args.save_every, writer=writer_val
        )
        schedule_lr = ScheduleLRCallback(
            model, decay_rate=args.lr_schedule_rate, writer=writer_val
        )
        train(Y_train, Y_test, model.training_step, model.calculate_losses, args.num_epochs, args.batch_size,
              train_writer=writer_train, val_writer=writer_val,
              callbacks=[write_hist_summary, save_model, schedule_lr],
              features_train=X_train, features_val=X_test, features_noise=features_noise)


if __name__ == '__main__':
    main()
