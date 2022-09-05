docker run -it \
  --rm \
  -u $(id -u):$(id -g) \
  --env HOME=`pwd` \
  --gpus all \
  -v `pwd`:`pwd` \
  alexdrydew/tpc-trainer \
  /bin/bash
