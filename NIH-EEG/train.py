# -*- coding: utf-8 -*-

from utils import *
from scipy.io import loadmat
import os


filename = os.path.join(DATA_PATH, "Train/train_1/1_1_0.mat")
data_struct = loadmat(filename)["dataStruct"][0][0]
data = data_struct[0]
