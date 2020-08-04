docker run -it \
  --env HOME=`pwd` \
  -v `pwd`:`pwd` \
  centos:centos7.8.2003 \
  /bin/bash
