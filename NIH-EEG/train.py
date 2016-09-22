# -*- coding: utf-8 -*-

from utils import *
from scipy.io import loadmat
import os, sys
import numpy as np

sys.path.insert(0, 'C:\Users\Xanto183\git\ArchMM\ArchMM\ArchMM\CyFiles')

from HMM_Core import AdaptiveHMM
from Spectral import *
        
def downSample(signal, f1, f2):
    step = np.round(float(f1) / f2)
    return signal[::step] if len(signal.shape) == 1 else signal[::step, :]

def getDataset(I, J, K):
    dataset = dict()
    filename = "Train/train_1/%i_%i_%i.mat" % (I, J, K)
    filename = os.path.join(DATA_PATH, filename)
    data_struct = loadmat(filename)["dataStruct"][0][0]
    dataset["data"] = data_struct[0]
    dataset["sampling_rate"] = data_struct[1][0][0]
    dataset["n_samples"] = data_struct[2][0][0]
    dataset["channel_indices"] = data_struct[3][0]
    dataset["sequence_number"] = data_struct[4][0][0]
    return dataset

DS_0 = getDataset(1, 1, 0)
DS_1 = getDataset(1, 1, 1)


alpha = AlphaCoherence(DS_0["data"][15000:16000, :], DS_0["sampling_rate"])
print(alpha)
"""
hmm = AdaptiveHMM(10)
hmm.fit(np.asarray(DS_0["data"][3000:6000], dtype = np.float), dynamic_features = False)
print(hmm.score(np.asarray(DS_0["data"][3000:6000], dtype = np.float)))
print(hmm.score(np.asarray(DS_1["data"][3000:6000], dtype = np.float)))
hmm = AdaptiveHMM(10)
hmm.fit(np.asarray(DS_1["data"][3000:6000], dtype = np.float), dynamic_features = False)
print(hmm.score(np.asarray(DS_0["data"][3000:6000], dtype = np.float)))
print(hmm.score(np.asarray(DS_1["data"][3000:6000], dtype = np.float)))
"""
print("Finished")