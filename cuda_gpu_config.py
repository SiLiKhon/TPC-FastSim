import os

import tensorflow as tf

def setup_gpu(gpu_num=None):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if gpu_num is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    assert len(logical_devices) > 0, "Not enough GPU hardware devices available"
