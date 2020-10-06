import os, sys
from pathlib import Path
import shutil
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import PIL
import yaml

from data import preprocessing
from models.utils import latest_epoch, load_weights
from models.training import train
from models.callbacks import SaveModelCallback, WriteHistSummaryCallback, ScheduleLRCallback
from models.model_v4 import Model_v4
from metrics import evaluate_model
import cuda_gpu_config

def make_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--config', type=str, required=False)
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--gpu_num', type=str, required=False)
    parser.add_argument('--prediction_only', action='store_true', default=False)

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
    print_args(args)
    return args


def load_config(file):
    with open(file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert (
        (config['feature_noise_power'] is None) ==
        (config['feature_noise_decay'] is None)
    ), 'Noise power and decay must be both provided'

    return config


def main():
    args = parse_args()

    cuda_gpu_config.setup_gpu(args.gpu_num)

    model_path = Path('saved_models') / args.checkpoint_name

    if args.prediction_only:
        assert model_path.exists(), "Couldn't find model directory"
        assert not args.config, "Config should be read from model path when doing prediction"
        args.config = str(model_path / 'config.yaml')
    else:
        assert not model_path.exists(), "Model directory already exists"
        assert args.config, "No config provided"

        model_path.mkdir(parents=True)
        config_destination = str(model_path / 'config.yaml')
        shutil.copy(args.config, config_destination)

        args.config = config_destination

    config = load_config(args.config)

    model = Model_v4(config)

    if args.prediction_only:
        load_weights(model, model_path)

    preprocessing._VERSION = model.data_version
    data, features = preprocessing.read_csv_2d(pad_range=model.pad_range, time_range=model.time_range)
    features = features.astype('float32')

    data_scaled = model.scaler.scale(data).astype('float32')

    Y_train, Y_test, X_train, X_test = train_test_split(data_scaled, features, test_size=0.25, random_state=42)

    if not args.prediction_only:
        writer_train = tf.summary.create_file_writer(f'logs/{args.checkpoint_name}/train')
        writer_val = tf.summary.create_file_writer(f'logs/{args.checkpoint_name}/validation')


    if args.prediction_only:
        epoch = latest_epoch(model_path)
        prediction_path = model_path / f"prediction_{epoch:05d}"
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
        if config['feature_noise_power'] is not None:
            def features_noise(epoch):
                current_power = config['feature_noise_power'] / (10**(epoch / config['feature_noise_decay']))
                with writer_train.as_default():
                    tf.summary.scalar("features noise power", current_power, epoch)

                return current_power


        save_model = SaveModelCallback(
            model=model, path=model_path, save_period=config['save_every']
        )
        write_hist_summary = WriteHistSummaryCallback(
            model, sample=(X_test, Y_test),
            save_period=config['save_every'], writer=writer_val
        )
        schedule_lr = ScheduleLRCallback(
            model, decay_rate=config['lr_schedule_rate'], writer=writer_val
        )
        train(Y_train, Y_test, model.training_step, model.calculate_losses, config['num_epochs'],
              config['batch_size'], train_writer=writer_train, val_writer=writer_val,
              callbacks=[write_hist_summary, save_model, schedule_lr],
              features_train=X_train, features_val=X_test, features_noise=features_noise)


if __name__ == '__main__':
    main()
