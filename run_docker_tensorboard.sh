docker run -it \
  -p 127.0.0.1:6006:6006/tcp \
  --runtime nvidia \
  -v `pwd`:`pwd` \
  silikhon/tensorflow2:v0 \
  /bin/bash -c 'cd '`pwd`'; tensorboard --logdir=. --host=0.0.0.0 --samples_per_plugin images=100'
