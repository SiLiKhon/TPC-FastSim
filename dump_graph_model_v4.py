import argparse
from pathlib import Path

import tensorflow as tf

from cuda_gpu_config import setup_gpu
from model_export import dump_graph
from models.model_v4 import preprocess_features, Model_v4
from models.utils import load_weights
from run_model_v4 import load_config


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='model_export/model_v4/graph.pbtxt')
    parser.add_argument('--latent_dim', type=int, default=32, required=False)
    parser.add_argument('--dont_hack_upsampling_op', default=True, action='store_true')
    parser.add_argument('--test_input', type=float, nargs=4, default=None)
    parser.add_argument('--constant_seed', type=float, default=None)
    parser.add_argument('--gpu_num', type=str, default=None)

    args, _ = parser.parse_known_args()

    setup_gpu(args.gpu_num)

    print("")
    print("----" * 10)
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"    {k} : {v}")
    print("----" * 10)
    print("")

    model_path = Path('saved_models') / args.checkpoint_name

    full_model = Model_v4(load_config(model_path / 'config.yaml'))
    load_weights(full_model, model_path)
    model = full_model.generator

    if args.constant_seed is None:

        def preprocess(x):
            size = tf.shape(x)[0]
            latent_input = tf.random.normal(shape=(size, args.latent_dim), dtype='float32')
            return tf.concat([preprocess_features(x), latent_input], axis=-1)

    else:

        def preprocess(x):
            size = tf.shape(x)[0]
            latent_input = tf.ones(shape=(size, args.latent_dim), dtype='float32') * args.constant_seed
            return tf.concat([preprocess_features(x), latent_input], axis=-1)

    def postprocess(x):
        x = 10 ** x - 1
        return tf.where(x < 1.0, 0.0, x)

    dump_graph.model_to_graph(
        model,
        preprocess,
        postprocess,
        input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32)],
        output_file=args.output_path,
        test_input=args.test_input,
        hack_upsampling=not args.dont_hack_upsampling_op,
    )


if __name__ == '__main__':
    main()
