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
from models.baseline_v2_10x10 import BaselineModel10x10
from metrics import make_metric_plots, make_histograms


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32, required=False)
    parser.add_argument('--lr', type=float, default=1e-4, required=False)
    parser.add_argument('--num_disc_updates', type=int, default=3, required=False)
    parser.add_argument('--lr_schedule_rate', type=float, default=0.998, required=False)
    parser.add_argument('--save_every', type=int, default=50, required=False)
    parser.add_argument('--num_epochs', type=int, default=10000, required=False)
    parser.add_argument('--latent_dim', type=int, default=32, required=False)
    parser.add_argument('--gpu_num', type=str, required=False)
    parser.add_argument('--kernel_init', type=str, default='glorot_uniform', required=False)
    parser.add_argument('--gp_lambda', type=float, default=10., required=False)
    parser.add_argument('--gpdata_lambda', type=float, default=0., required=False)
    parser.add_argument('--angles_to_radians', action='store_true')
    parser.add_argument('--num_additional_disc_layers', type=int, default=0, required=False)
    parser.add_argument('--cramer_gan', action='store_true', default=False)
    parser.add_argument('--features_to_tail', action='store_true', default=False)
    parser.add_argument('--dropout_rate', type=float, default=0.2, required=False)
    parser.add_argument('--prediction_only', action='store_true', default=False)


    args = parser.parse_args()

    print("")
    print("----" * 10)
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"    {k} : {v}")
    print("----" * 10)
    print("")

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if args.gpu_num is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    assert len(logical_devices) > 0, "Not enough GPU hardware devices available"

    model_path = Path('saved_models') / args.checkpoint_name
    if args.prediction_only:
        assert model_path.exists(), "Couldn't find model directory"
    else:
        assert not model_path.exists(), "Model directory already exists"
        model_path.mkdir(parents=True)

        with open(model_path / 'arguments.txt', 'w') as f:
            raw_args = [a for a in sys.argv[1:] if a[0] != '@']
            fnames = [a[1:] for a in sys.argv[1:] if a[0] == '@']

            f.write('\n'.join(raw_args))
            for fname in fnames:
                with open(fname, 'r') as f_in:
                    if len(raw_args) > 0: f.write('\n')
                    f.write(f_in.read())

    model = BaselineModel10x10(kernel_init=args.kernel_init, lr=args.lr,
                               num_disc_updates=args.num_disc_updates, latent_dim=args.latent_dim,
                               gp_lambda=args.gp_lambda, gpdata_lambda=args.gpdata_lambda,
                               num_additional_layers=args.num_additional_disc_layers,
                               cramer=args.cramer_gan, features_to_tail=args.features_to_tail,
                               dropout_rate=args.dropout_rate)

    if args.prediction_only:
        def epoch_from_name(name):
            epoch, = re.findall('\d+', name)
            return int(epoch)

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



    def save_model(step):
        if step % args.save_every == 0:
            print(f'Saving model on step {step} to {model_path}')
            model.generator.save(str(model_path.joinpath("generator_{:05d}.h5".format(step))))
            model.discriminator.save(str(model_path.joinpath("discriminator_{:05d}.h5".format(step))))

    preprocessing._VERSION = 'data_v2'
    pad_range = (39, 49)
    time_range = (266, 276)
    data, features = preprocessing.read_csv_2d(pad_range=pad_range, time_range=time_range)
    assert np.isclose(features[:,1].std(), 0), features[:,1].std()
    features = features[:,:1].astype('float32')
    if args.angles_to_radians:
        features *= np.pi / 180

    data_scaled = np.log10(1 + data).astype('float32')
    Y_train, Y_test, X_train, X_test = train_test_split(data_scaled, features, test_size=0.25, random_state=42)

    if not args.prediction_only:
        writer_train = tf.summary.create_file_writer(f'logs/{args.checkpoint_name}/train')
        writer_val = tf.summary.create_file_writer(f'logs/{args.checkpoint_name}/validation')

    unscale = lambda x: 10 ** x - 1

    def get_images(return_raw_data=False, calc_chi2=False, gen_more=None):
        if gen_more is None:
            gen_features = X_test
        else:
            gen_features = np.tile(
                X_test,
                [gen_more] + [1] * (X_test.ndim - 1)
            )
        gen_scaled = model.make_fake(gen_features).numpy()
        real = unscale(Y_test)
        gen = unscale(gen_scaled)
        gen[gen < 0] = 0
        gen1 = np.where(gen < 1., 0, gen)
        images = make_metric_plots(real, gen, features={'angle' : (X_test, gen_features)}, calc_chi2=calc_chi2)
        if calc_chi2:
            images, chi2 = images

        images1 = make_metric_plots(real, gen1, features={'angle' : (X_test, gen_features)})

        img_amplitude = make_histograms(Y_test.flatten(), gen_scaled.flatten(), 'log10(amplitude + 1)', logy=True)

        result = [images, images1, img_amplitude]

        if return_raw_data:
            result += [(gen_features, gen)]

        if calc_chi2:
            result += [chi2]

        return result


    def write_hist_summary(step):
        if step % args.save_every == 0:
            images, images1, img_amplitude, chi2 = get_images(calc_chi2=True)

            with writer_val.as_default():
                tf.summary.scalar("chi2", chi2, step)

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

    if args.prediction_only:
        prediction_path = model_path / f"prediction_{epoch_from_name(latest_gen_checkpoint.stem):05d}"
        assert not prediction_path.exists(), "Prediction path already exists"
        prediction_path.mkdir()

        array_to_img = lambda arr: PIL.Image.fromarray(arr.reshape(arr.shape[1:]))

        (
            images, images1, img_amplitude,
            gen_dataset, chi2
        ) = get_images(calc_chi2=True, return_raw_data=True, gen_more=10)
        for k, img in images.items():
            array_to_img(img).save(str(prediction_path / f"{k}.png"))
        for k, img in images.items():
            array_to_img(img).save(str(prediction_path / f"{k}_amp_gt_1.png"))
        array_to_img(img_amplitude).save(str(prediction_path / "log10_amp_p_1.png"))

        with open(str(prediction_path / 'generated.dat'), 'w') as f:
            for event_X, event_Y in zip(*gen_dataset):
                f.write(f'params: {float(event_X):.3f} 16.941\n')
                for ipad, time_distr in enumerate(event_Y, pad_range[0]):
                    for itime, amp in enumerate(time_distr, time_range[0]):
                        if amp < 1:
                            continue
                        f.write(" {:2d} {:3d} {:8.3e} ".format(ipad, itime, amp))
                f.write('\n')
        with open(str(prediction_path / 'stats'), 'w') as f:
            f.write(f"{chi2:.2f}\n")

    else:
        train(Y_train, Y_test, model.training_step, model.calculate_losses, args.num_epochs, args.batch_size,
              train_writer=writer_train, val_writer=writer_val,
              callbacks=[write_hist_summary, save_model, schedule_lr],
              features_train=X_train, features_val=X_test)


if __name__ == '__main__':
    main()
