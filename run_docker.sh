docker run -it \
  --rm \
  -u $(id -u):$(id -g) \
  --env HOME=`pwd` \
  --runtime nvidia \
  -v `pwd`:`pwd` \
  silikhon/tensorflow2:v1 \
  /bin/bash
