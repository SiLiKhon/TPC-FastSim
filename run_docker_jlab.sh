if [ -z "$1" ]
  then
    PORT=8890
else
	PORT=$1
fi

docker run -it \
  --rm \
  -u $(id -u):$(id -g) \
  --env HOME=`pwd` \
  -p 127.0.0.1:$PORT:8888/tcp \
  --gpus all \
  -v `pwd`:`pwd` \
  alexdrydew/tpc-trainer \
  /bin/bash -c 'cd '`pwd`'; jupyter lab --ip=0.0.0.0 --allow-root'
