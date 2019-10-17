docker run -v `pwd`:`pwd` --gpus 1 silikhon/tensorflow2:v0 /bin/bash -c 'cd '`pwd`'; python test_script.py'
