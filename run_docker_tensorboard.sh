docker run -it \
  -u $(id -u):$(id -g) \
  --rm \
  --env HOME=`pwd` \
  -p 127.0.0.1:6126:6006/tcp \
  --runtime nvidia \
  -v `pwd`:`pwd` \
  silikhon/tensorflow2:v1 \
  /bin/bash -c 'cd '`pwd`'; tensorboard --logdir=logs --host=0.0.0.0 --samples_per_plugin images=100'
