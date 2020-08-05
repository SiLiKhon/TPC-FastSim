# Build the model lib:

```bash
./run_docker_centos.sh
cd ~
./setup_centos.sh
cd tensorflow/
./configure
# show path to python3 (/usr/bin/python3)
# rest options to default
BAZEL_LINKLIBS=-l%:libstdc++.a bazel build --local_resources 30000,15,1.0 -c opt model_v4:all_models
```

# Test:
```bash
# copy .so and .h files to the testing folder
# cd to the testing folder, run docker inside
yum -y install gcc-c++
g++ main.cpp -L. -lmodel -std=c++11
LD_LIBRARY_PATH=. ./a.out
```
