# from tensorflow 2.1.0 gpu py3 jupyter
FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter

RUN pip install --upgrade \
    pip \
    setuptools

RUN pip3 install jupyterlab==1.2.6 \
                 pandas==1.0.1 \
                 numpy==1.18.1 \
                 scikit-learn==0.22.1 \
                 tqdm==4.42.1 \
                 matplotlib==3.1.3 \
                 seaborn==0.10.0 \
                 scipy==1.4.1 \
                 Pillow==7.0.0 \
                 PyYAML==5.3
