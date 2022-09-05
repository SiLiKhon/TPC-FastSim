# Build the model lib:

If running in docker, first run the container and cd to the home path:
```bash
./run_docker_centos.sh
cd ~
```

The `setup_centos.sh` script (`model_export/setup_centos.sh`) contains all the environment setup.

*Note: this was only tested in a clean CentOS environment (in the docker container above), there might be conflicts with the main environment used for model training and evaluation.*

The `setup_centos.sh` script also creates a symlink to the `model_export/model_v4` folder (which should contain the `*.pbtxt` files describing the exported model) and puts it into `tensorflow`. This folder contains bazel build instructions and needs to be inside the `tensorflow` directory to be built.
```bash
./setup_centos.sh
```

Now we need to build TensorFlow and our model (note: this make take a few hours for the first time...). The `./configure` command will ask for a bunch of configuration options - all of these, except for the path to python3, need to be set to default (i.e., just hit ENTER):
```bash
cd tensorflow/
./configure
# show path to python3 (/usr/bin/python3)
# rest options to default
BAZEL_LINKLIBS=-l%:libstdc++.a bazel build --local_resources 30000,15,1.0 -c opt model_v4:all_models
```
The `--local_resources` parameter from the last line allows to limit the resources to be used by bazel. 30000 is memory in megabytes, 15 is the number of CPU cores and 1.0 is the I/O capability. Not limiting the resources may result in a compiler crash. To further rebuild the model (e.g., after updating the `*.pbtxt` files) only the last command is needed.

# Test:
```bash
# copy .so and .h files to the testing folder
# cd to the testing folder, run docker inside
yum -y install gcc-c++
g++ main.cpp -L. -lmodel -std=c++11
LD_LIBRARY_PATH=. ./a.out
```
