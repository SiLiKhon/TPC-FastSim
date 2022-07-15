docker run -it --rm \
  -u $(id -u):$(id -g) \
  --env HOME=`pwd` \
  -v `pwd`:`pwd` \
  texlive/texlive \
  /bin/bash

