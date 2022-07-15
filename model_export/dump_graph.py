from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import convert_to_constants
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.tools import optimize_for_inference_lib

from . import tf2xla_pb2


def model_to_graph(
    model,
    output_file,
    test_input=None,
    hack_upsampling=False,
    batch_sizes=(1, 10, 100, 1000, 10000),
    perf_iterations=5,
):
    tf.keras.backend.set_learning_phase(0)

    constant_graph = convert_to_constants.convert_variables_to_constants_v2(model.get_concrete_function())

    if hack_upsampling:
        print("Warning: hacking upsampling operations")
        for op in constant_graph.graph.get_operations():
            if 'ResizeNearestNeighbor' == op.type:
                op._set_attr('align_corners', attr_value_pb2.AttrValue(b=True))
                op._set_attr('half_pixel_centers', attr_value_pb2.AttrValue(b=False))

    output_file = Path(output_file)
    path = str(output_file.parent)
    filename = output_file.name

    optimized_graph = optimize_for_inference_lib.optimize_for_inference(
        constant_graph.graph.as_graph_def(),
        [i.op.name for i in constant_graph.inputs],
        [o.op.name for o in constant_graph.outputs],
        tf.float32.as_datatype_enum,
    )

    tf.io.write_graph(optimized_graph, path, filename)

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
        print(model(tf.convert_to_tensor([test_input])))

        for batch_size in batch_sizes[::-1]:
            timings = []
            iterations = perf_iterations * max(1, 100 // batch_size)
            for i in range(iterations):
                batched_input = tf.random.normal(shape=(batch_size, len(test_input)), dtype='float32')
                t0 = perf_counter()
                model(batched_input).numpy()
                t1 = perf_counter()
                timings.append((t1 - t0) * 1000.0 / batch_size)

            timings = np.array(timings)
            mean = timings.mean()
            err = timings.std() / (len(timings) - 1) ** 0.5
            print(f'With batch size = {batch_size}, duration per 1 generation is: {mean} +\\- {err} ms')
