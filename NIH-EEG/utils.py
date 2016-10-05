# -*- coding: utf-8 -*-

import json, os, pickle, random
import warnings
import numpy as np
from Spectral import *

warnings.filterwarnings('ignore')

from scipy.io import loadmat

with open('SETTINGS.json') as settings_file:
    SETTINGS = json.load(settings_file)
    
DATA_PATH = str(SETTINGS["DATA_PATH"])
MODEL_PATH = str(SETTINGS["MODEL_PATH"])


class TooMuchDropOut(ValueError):
    def __init__(self, message, *args):
        self.message = message
        super(TooMuchDropOut, self).__init__(message, *args)
        
def checkDropOuts(data):
    total = data.shape[0] * data.shape[1]
    nonzeros = np.count_nonzero(total)
    dropouts = total - nonzeros
    if dropouts > 5000:
        raise TooMuchDropOut("Error. Too much drop out in mat file.")

def matToDataset(mat):
    data_struct = mat["dataStruct"][0][0]
    data = dict()
    data["data"] = data_struct[0]
    data["sampling_rate"] = data_struct[1][0][0]
    data["n_samples"] = data_struct[2][0][0]
    data["channel_indices"] = data_struct[3][0]
    data["sequence_number"] = data_struct[4][0][0]
    return data
        
def getDataset(I, hour, K):
    dataset = []
    J = hour * 6
    for j in range(1, 7):
        filename = "Train/train_1/%i_%i_%i.mat" % (I, J + j, K)
        filename = os.path.join(DATA_PATH, filename)
        data_struct = loadmat(filename)["dataStruct"][0][0]
        data = dict()
        data["data"] = data_struct[0]
        data["sampling_rate"] = data_struct[1][0][0]
        data["n_samples"] = data_struct[2][0][0]
        data["channel_indices"] = data_struct[3][0]
        data["sequence_number"] = data_struct[4][0][0]
        dataset.append(data)
    return dataset

def getFileDataset(I, J, K):
    filename = "Train/train_1/%i_%i_%i.mat" % (I, J, K)
    filename = os.path.join(DATA_PATH, filename)
    dataset = matToDataset(loadmat(filename))
    return dataset

def exploreDatabase():
    for i in range(18):
        dataset = getDataset(1, i + 1, 1)
        sequence_number = dataset["sequence_number"]
        print(sequence_number) 
    """
    path = os.path.join(DATA_PATH, "Train/train_1")
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        data_struct = loadmat(filepath)["dataStruct"][0][0]
        sequence_number = data_struct[4][0][0]
        print(filename, sequence_number)
    """
    
def downSample(signal, f1, f2):
    step = np.round(float(f1) / f2)
    return signal[::step] if len(signal.shape) == 1 else signal[::step, :]


def extractFeatures(dataset, begin, end):
    assert(begin < end)
    electrode_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1]
    data = np.copy(dataset["data"][begin:end])
    window = np.hanning(end - begin)
    for i in range(16):
        data[:, i] = window * data[:, i]
    autosp = AutospectralDensities(data, dataset["sampling_rate"])
    features = np.empty(32, dtype = float)
    for i in range(16):
        features[i] = AlphaCoherence(data, autosp, electrode_ids[i], 
                                  electrode_ids[i + 1], dataset["sampling_rate"])
    features[16:32] = STE(data)
    return features

def a(dataset):
    checkDropOuts(dataset["data"])
    obs = np.empty((100, 32), dtype = np.float)
    step = len(dataset["data"]) / 100
    for i in range(100):
        obs[i] = extractFeatures(dataset, i * step, i * step + 1024)
    return obs



