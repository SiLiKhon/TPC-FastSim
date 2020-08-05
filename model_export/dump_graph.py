from pathlib import Path

import tensorflow as tf
from tensorflow.python.framework import convert_to_constants
from tensorflow.core.framework import attr_value_pb2

from . import tf2xla_pb2

def model_to_graph(model, preprocess, postprocess, input_signature, output_file, test_input=None,
        hack_upsampling=False, batch_sizes=(1, 10, 100, 1000, 10000)):
    tf.keras.backend.set_learning_phase(0)

    @tf.function(input_signature=input_signature)
    def to_save(x):
        return postprocess(
            model(preprocess(x))
        )

    constant_graph = \
        convert_to_constants.convert_variables_to_constants_v2(
            to_save.get_concrete_function()
        )

    if hack_upsampling:
        print("Warning: hacking upsampling operations")
        for op in constant_graph.graph.get_operations():
            if 'ResizeNearestNeighbor' == op.type:
                op._set_attr('align_corners', attr_value_pb2.AttrValue(b=True))
                op._set_attr('half_pixel_centers', attr_value_pb2.AttrValue(b=False))


    output_file = Path(output_file)
    path = str(output_file.parent)
    filename = output_file.name

    tf.io.write_graph(
        constant_graph.graph.as_graph_def(),
        path, filename
    )

    for batch_size in batch_sizes:
        config = tf2xla_pb2.Config()

        for x in constant_graph.inputs:
            shape = tf.TensorShape([batch_size] + list(x.shape)[1:])
            feed = config.feed.add()
            feed.id.node_name = x.op.name
            feed.shape.MergeFrom(shape.as_proto())

        for x in constant_graph.outputs:
            fetch = config.fetch.add()
            fetch.id.node_name = x.op.name

        config_filename = Path(path) / f"{output_file.stem}-{batch_size}.config{output_file.suffix}"
        with open(str(config_filename), 'w') as f:
            f.write(str(config))

    if test_input is not None:
        print(to_save(
            tf.convert_to_tensor([test_input])
        ))
