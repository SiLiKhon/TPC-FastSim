docker run -u $(id -u):$(id -g) --runtime nvidia -v `pwd`:`pwd` silikhon/tensorflow2:v1 /bin/bash -c 'cd '`pwd`'; python test_script.py'
