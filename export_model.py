import os
import argparse
import tensorflow as tf
import tf2onnx

from pathlib import Path

from model_export import dump_graph
from models.model_v4 import Model_v4, preprocess_features
from models.utils import load_weights
from run_model_v4 import load_config


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--dont_hack_upsampling_op', default=True, action='store_true')
    parser.add_argument('--test_input', type=float, nargs=4, default=None)

    parser.add_argument('--latent_space', choices=['normal', 'uniform', 'constant', 'none'], default='normal')
    parser.add_argument('--latent_dim', type=int, default=32, required=False)
    parser.add_argument('--constant_latent', type=float, default=None)

    parser.add_argument('--export_format', choices=['pbtxt', 'onnx'], default='pbtxt')

    parser.add_argument('--upload_to_mlflow', action='store_true')
    parser.add_argument('--aws_access_key_id', type=str, required=False)
    parser.add_argument('--aws_secret_access_key', type=str, required=False)
    parser.add_argument('--mlflow_url', type=str, required=False)
    parser.add_argument('--s3_url', type=str, required=False)
    parser.add_argument('--mlflow_model_name', type=str, required=False)

    args, _ = parser.parse_known_args()

    if args.upload_to_mlflow:
        assert args.export_format == 'onnx', 'Only onnx export format is supported when uploading to MLFlow'
        assert args.aws_access_key_id, 'You need to specify aws_access_key_id to upload model to MLFlow'
        assert args.aws_secret_access_key, 'You need to specify aws_secret_access_key to upload model to MLFlow'
        assert args.mlflow_url, 'You need to specify mlflow_url to upload model to MLFlow'
        assert args.s3_url, 'You need to specify s3_url to upload model to MLFlow'
        assert args.mlflow_model_name, 'You need to specify mlflow_model_name to upload model to MLFlow'

    if args.output_path is None:
        if args.export_format == 'pbtxt':
            args.output_path = Path('model_export/model_v4')
        else:
            args.output_path = Path('model_export/onnx')

    args.output_path.mkdir(parents=True, exist_ok=True)

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

    input_signature, preprocess = construct_preprocess(args)

    def postprocess(x):
        x = 10**x - 1
        return tf.where(x < 1.0, 0.0, x)

    @tf.function(input_signature=input_signature)
    def to_save(x):
        return postprocess(model(preprocess(x)))

    if args.export_format == 'pbtxt':
        dump_graph.model_to_graph(
            to_save,
            output_file=Path(args.output_path) / "graph.pbtxt",
            test_input=args.test_input,
            hack_upsampling=not args.dont_hack_upsampling_op,
        )
    else:
        onnx_model, _ = tf2onnx.convert.from_function(
            to_save,
            input_signature=input_signature,
            output_path=Path(args.output_path) / f'{args.checkpoint_name}.onnx',
        )

        if args.upload_to_mlflow:
            import mlflow

            os.environ['AWS_ACCESS_KEY_ID'] = args.aws_access_key_id
            os.environ['AWS_SECRET_ACCESS_KEY'] = args.aws_secret_access_key
            os.environ['MLFLOW_S3_ENDPOINT_URL'] = args.s3_url
            mlflow.set_tracking_uri(args.mlflow_url)
            mlflow.set_experiment('model_export')

            mlflow.onnx.log_model(onnx_model, artifact_path='model_onnx', registered_model_name=args.mlflow_model_name)


def construct_preprocess(args):
    latent_input_gen = None
    predefined_batch_size = None if args.export_format == 'pbtxt' else 1

    if args.latent_space == 'normal':

        def latent_input_gen(batch_size):
            return tf.random.normal(shape=(batch_size, args.latent_dim), dtype='float32')

    elif args.latent_space == 'uniform':

        def latent_input_gen(batch_size):
            return tf.random.uniform(shape=(batch_size, args.latent_dim), dtype='float32')

    if latent_input_gen is None:
        input_signature = [tf.TensorSpec(shape=[predefined_batch_size, 36], dtype=tf.float32)]

        def preprocess(x):
            return tf.concat([preprocess_features(x[..., :4]), x[..., 4:]], axis=-1)

    else:
        input_signature = [tf.TensorSpec(shape=[predefined_batch_size, 4], dtype=tf.float32)]

        def preprocess(x):
            size = tf.shape(x)[0]
            latent_input = latent_input_gen(size)
            return tf.concat([preprocess_features(x), latent_input], axis=-1)

    return input_signature, preprocess


if __name__ == '__main__':
    main()
