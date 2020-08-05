load(":model_export.bzl", "define_export_targets")

define_export_targets()
filegroup(
    name = "all_models",
    srcs = [
        "libmodel_" + str(batch_size) + ".so"
        for batch_size in [1, 10, 100, 1000, 10000]
    ]
)
