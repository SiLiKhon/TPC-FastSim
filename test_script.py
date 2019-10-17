import os

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

import numpy as np
from sklearn.model_selection import train_test_split

from data import preprocessing

from models import training, baseline_10x15

data = preprocessing.read_csv_2d()

data_scaled = np.log10(1 + data).astype('float32')
X_train, X_test = train_test_split(data_scaled, test_size=0.25, random_state=42)

training.train(X_train, X_test, baseline_10x15.training_step, baseline_10x15.calculate_losses, 5, 32)
