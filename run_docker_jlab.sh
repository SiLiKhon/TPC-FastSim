docker run -it \
  -u $(id -u):$(id -g) \
  --env HOME=`pwd` \
  -p 127.0.0.1:8890:8888/tcp \
  --runtime nvidia \
  -v `pwd`:`pwd` \
  silikhon/tensorflow2:v1 \
  /bin/bash -c 'cd '`pwd`'; jupyter lab --ip=0.0.0.0 --allow-root'
