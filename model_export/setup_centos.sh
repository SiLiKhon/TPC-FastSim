yum -y install git python3-devel.x86_64 python3-pip zip unzip which patch

yum -y install centos-release-scl
yum-config-manager --enable rhel-server-rhscl-7-rpms
yum -y install devtoolset-7

if [ ! -d tensorflow ]; then
  git clone --depth=1 --single-branch --branch v2.1.0 https://github.com/tensorflow/tensorflow
else
  echo FOUND TENSORFLOW DIRECTORY!
fi

pip3 install -U pip six 'numpy<1.19.0' wheel setuptools mock 'future>=0.17.1'
pip3 install -U keras_applications --no-deps
pip3 install -U keras_preprocessing --no-deps
curl -LJO https://github.com/bazelbuild/bazel/releases/download/0.29.1/bazel-0.29.1-installer-linux-x86_64.sh
chmod +x bazel-0.29.1-installer-linux-x86_64.sh
./bazel-0.29.1-installer-linux-x86_64.sh

if [ ! -d tensorflow/model_v4 ]; then
  ln -s ../model_v4 tensorflow/model_v4
else
  echo FOUND MODEL SUBFOLDER
fi

scl enable devtoolset-7 bash
