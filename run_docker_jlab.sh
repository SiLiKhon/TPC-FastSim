docker run -it \
  -p 127.0.0.1:8890:8888/tcp \
  --runtime nvidia \
  -v `pwd`:`pwd` \
  silikhon/tensorflow2:v0 \
  /bin/bash -c 'cd '`pwd`'; jupyter lab --ip=0.0.0.0 --allow-root'
