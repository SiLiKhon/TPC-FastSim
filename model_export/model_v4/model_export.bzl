load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

def define_export_targets(
    batch_sizes=(1, 10, 100, 1000, 10000)
):
    for batch_size in batch_sizes:
        tf_library(
            name = 'graph_' + str(batch_size) + "/graph",
            config = 'graph-' + str(batch_size) + '.config.pbtxt',
            cpp_class = 'Graph_' + str(batch_size),
            graph = 'graph.pbtxt',
        )

        native.cc_binary(
            name = "libmodel_" + str(batch_size) + ".so",
            srcs = ["model.cc", "graph_" + str(batch_size) + "/graph.h", "model.h"],
            deps = [":graph_" + str(batch_size) + "/graph", "//third_party/eigen3"],
            includes = ['.', "graph_" + str(batch_size)],
            linkopts = ["-lpthread"],
            linkshared = 1,
            copts = [
                "-fPIC",
                '-DGRAPH_CLASS=Graph_' + str(batch_size),
                '-DBATCH_SIZE=' + str(batch_size)
            ],
        )
