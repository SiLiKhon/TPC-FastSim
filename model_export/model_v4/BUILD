load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

tf_library(
    name = 'graph',
    config = 'graph.config.pbtxt',
    cpp_class = 'Graph',
    graph = 'graph.pbtxt',
)

cc_binary(
    name = "libmodel.so",
    srcs = ["model.cc", "graph.h", "model.h"],
    deps = [":graph", "//third_party/eigen3"],
    includes = ['.'],
    linkopts = ["-lpthread"],
    linkshared = 1,
    copts = ["-fPIC"],
)
