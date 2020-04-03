import tensorflow as tf
import numpy as np
from tqdm import trange


def train(data_train, data_val, train_step_fn, loss_eval_fn, num_epochs, batch_size,
          train_writer=None, val_writer=None, callbacks=[], features_train=None, features_val=None):
    if not ((features_train is None) or (features_val is None)):
        assert features_train is not None
        assert features_val is not None

    for i_epoch in range(num_epochs):
        print("Working on epoch #{}".format(i_epoch), flush=True)

        tf.keras.backend.set_learning_phase(1)  # training
        
        shuffle_ids = np.random.permutation(len(data_train))
        losses_train = {}

        for i_sample in trange(0, len(data_train), batch_size):
            batch = data_train[shuffle_ids][i_sample:i_sample + batch_size]
            if features_train is not None:
                feature_batch = features_train[shuffle_ids][i_sample:i_sample + batch_size]

            if features_train is None:
                losses_train_batch = train_step_fn(batch)
            else:
                losses_train_batch = train_step_fn(feature_batch, batch)
            for k, l in losses_train_batch.items():
                losses_train[k] = losses_train.get(k, 0) + l.numpy() * len(batch)
        losses_train = {k : l / len(data_train) for k, l in losses_train.items()}
        
        tf.keras.backend.set_learning_phase(0)  # testing
        
        if features_train is None:
            losses_val = {k : l.numpy() for k, l in loss_eval_fn(data_val).items()}
        else:
            losses_val = {k : l.numpy() for k, l in loss_eval_fn(features_val, data_val).items()}
        for f in callbacks:
            f(i_epoch)
        
        if train_writer is not None:
            with train_writer.as_default():
                for k, l in losses_train.items():
                    tf.summary.scalar(k, l, i_epoch)
        
        if val_writer is not None:
            with val_writer.as_default():
                for k, l in losses_val.items():
                    tf.summary.scalar(k, l, i_epoch)

        print("", flush=True)
        print("Train losses:", losses_train)
        print("Val losses:", losses_val)


def average(models):
    parameters = [model.trainable_variables for model in models]
    assert len(np.unique([len(par) for par in parameters])) == 1

    result = tf.keras.models.clone_model(models[0])
    for params in zip(result.trainable_variables, *parameters):
        params[0].assign(tf.reduce_mean(params[1:], axis=0))

    return result
