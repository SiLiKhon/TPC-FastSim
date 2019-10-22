docker run -u $(id -u):$(id -g) --runtime nvidia -v `pwd`:`pwd` silikhon/tensorflow2:v0 /bin/bash -c 'cd '`pwd`'; python test_script.py'
