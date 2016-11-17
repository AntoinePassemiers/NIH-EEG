# -*- coding: utf-8 -*-

import json, os, pickle, random, csv
import warnings
import numpy as np
from scipy.io import loadmat

from Spectral import *
from outliers import *

with open('SETTINGS.json') as settings_file:
    SETTINGS = json.load(settings_file)
    
DATA_PATH = str(SETTINGS["DATA_PATH"])
MODEL_PATH = str(SETTINGS["MODEL_PATH"])


class TooMuchDropOut(ValueError):
    def __init__(self, message, *args):
        self.message = message
        super(TooMuchDropOut, self).__init__(message, *args)
        
def checkDropOuts(data, raise_error = True):
    total = data.shape[0] * data.shape[1]
    nonzeros = np.count_nonzero(data)
    dropouts = total - nonzeros
    if dropouts > 0 and raise_error:
        raise TooMuchDropOut("Error. Too much drop out in mat file.")
    return float(dropouts) / float(total) > 0.008

def matToDataset(mat):
    data_struct = mat["dataStruct"][0][0]
    data = dict()
    data["data"] = data_struct[0]
    data["sampling_rate"] = data_struct[1][0][0]
    data["n_samples"] = data_struct[2][0][0]
    data["channel_indices"] = data_struct[3][0]
    if len(data_struct) > 4:
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


def extractFeatures(dataset, featureset, begin, end):
    assert(begin < end)
    n_features = len(featureset)
    data = dataset["data"][begin:end]
    has_dropouts = checkDropOuts(data, raise_error = False)
    # window = np.hanning(end - begin)
    # for i in range(16):
    #    data[:, i] = window * data[:, i]
    grubbs = GrubbsTest(end - begin, alpha = 0.005)
    for i in range(featureset.n_electrodes):
        while grubbs.test(data[:, i]):
            pass
    if not has_dropouts:
        assert(len(featureset) > 0)
        features = np.empty(len(featureset), dtype = np.float64)
        for f in featureset.shared:
            f.process(data)
        k, l = 0, 0
        for f in featureset.getFeatures():
            l += len(f)
            features[k:l] = f.process(data)
            k += len(f)
        return features, False
    else:
        features = np.empty(n_features)
        features[:] = np.nan
        return features, True

def process(dataset, featureset):
    n_samples = 400
    n_features = len(featureset)
    obs = np.zeros((n_samples, n_features), dtype = np.float)
    step = len(dataset["data"]) / n_samples
    dropout_rate = 0
    for i in range(n_samples):
        features, has_nan = extractFeatures(dataset, featureset, i * step, i * step + 512)
        dropout_rate = dropout_rate if not has_nan else dropout_rate + 1
        if len(features) > 0:
            obs[i] = features
        else:
            if 0 < i:
                obs[i] = obs[i - 1]
    dropout_rate /= float(n_samples)
    return obs, dropout_rate

def MCC(TP, FP, TN, FN):
    """ Matthew's Correlation Coefficient """
    mcc = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if mcc != 0:
        mcc = (TP * TN + FP * FN) / mcc
    positive_rate = 0 if TP + FN == 0 else float(TP) / (TP + FP)
    negative_rate = 0 if FP + TN == 0 else float(TN) / (TN + FN)  
    return mcc, positive_rate, negative_rate

def removeCorruptedFiles():
    safe_labels_file = open(os.path.join(DATA_PATH, "train_and_test_data_labels_safe.csv"), "r")
    spamreader = csv.reader(safe_labels_file)
    spamreader.next()
    for line in spamreader:
        filename = line[0]
        is_safe = int(line[2])
        if not is_safe:
            patient_id = int(filename.split('_')[0])
            filepath = os.path.join(DATA_PATH, "Train/train_%i/%s" % (patient_id, filename))
            if os.path.isfile(filepath):
                os.remove(filepath)
            
    safe_labels_file.close()
    
def preprocessDataset(filepaths, labels, featureset):
    n_files = len(filepaths)
    n_features = len(featureset)
    inputs, outputs = list(), list()
    all_means = np.empty((n_files, n_features))
    all_stds  = np.empty((n_files, n_features))
    all_dropout_rates = list()
    for i in range(n_files):
        filepath = filepaths[i]
        print("Processing file number %i" % i)
        mat = loadmat(filepath)
        data, dropout_rate = process(matToDataset(mat), featureset)
        data = np.asarray(data, dtype = np.float32)
        all_dropout_rates.append(dropout_rate)
        masked = np.ma.masked_array(data, np.isnan(data))
        all_means[i, :] = np.mean(masked, axis = 0)
        all_stds[i, :] = np.std(masked, axis = 0)
        label = labels[i]
        if label == 1:
            output = np.ones(len(data), dtype = np.int32)
        elif label == 0:
            output = np.zeros(len(data), dtype = np.int32)
        else:
            raise ValueError("Label is not 0 or 1")
        print("Label : %i" % label)
        inputs.append(data)
        outputs.append(output)
    """
    global_mean = all_means.mean(axis = 0)
    global_std  = all_stds.mean(axis = 0)
    for i in range(n_files):
        for j in range(len(global_std)):
            inputs[i][:, j] -= global_mean[j]
            inputs[i][:, j] /= global_std[j]
    """
    return inputs, outputs, all_dropout_rates
    
    
    
    
    
    
    
    
