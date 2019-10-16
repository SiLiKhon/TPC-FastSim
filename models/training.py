import tensorflow as tf
import numpy as np
from tqdm import trange

def train(data_train, data_val, train_step_fn, loss_eval_fn, num_epochs, batch_size):

    losses_history = {}

    for i_epoch in range(num_epochs):
        print("Working on epoch #{}".format(i_epoch), flush=True)

        shuffle_ids = np.random.permutation(len(data_train))
        losses_train = {}

        for i_sample in trange(0, len(data_train), batch_size):
            batch = data_train[shuffle_ids][i_sample:i_sample + batch_size]

            losses_train_batch = train_step_fn(batch)
            for k, l in losses_train_batch.items():
                losses_train[k] = losses_train.get(k, 0) + l * len(batch)
        losses_train = {k : l / len(data_train) for k, l in losses_train.items()}
        losses_val = {k : l.numpy() for k, l in loss_eval_fn(data_val).items()}

        print("Train losses:", losses_train)
        print("Val losses:", losses_val)

        for key in losses_train:
            key_train = 'train_{}'.format(key)
            key_val   = 'val_{}'  .format(key)
            losses_history.setdefault(key_train, []).append(losses_train[key])
            losses_history.setdefault(key_val  , []).append(losses_val  [key])
    
    return losses_history