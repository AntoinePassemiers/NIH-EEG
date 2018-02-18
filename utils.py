# -*- coding: utf-8 -*-
# utils.py
# author : Antoine Passemiers

from spectral import *
from outliers import *

import json
import os
import pickle
import random
import csv
import warnings
import numpy as np
from scipy.io import loadmat


with open('SETTINGS.json') as settings_file:
    SETTINGS = json.load(settings_file)

DATA_PATH = str(SETTINGS["DATA_PATH"])
MODEL_PATH = str(SETTINGS["MODEL_PATH"])


class TooMuchDropOut(ValueError):
    def __init__(self, message, *args):
        self.message = message
        super(TooMuchDropOut, self).__init__(message, *args)


def checkDropOutsByChannel(data, raise_error=False, threshold=0.05):
    has_nan = False
    total = data.shape[0]
    nonzeros = np.count_nonzero(data)
    dropouts = total - nonzeros
    if float(dropouts) / float(total) > threshold:
        has_nan = True
        if raise_error:
            raise TooMuchDropOut("Error. Too much drop out in mat file.")
    return has_nan


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


def extractFeatures(dataset, featureset, begin, end):
    assert(begin < end)
    n_features = len(featureset)
    data = dataset["data"][begin:end]
    grubbs = GrubbsTest(end - begin, alpha=0.01)
    for i in range(featureset.n_electrodes):
        n_outliers = 0
        while grubbs.test(data[:, i]) and n_outliers < 60:
            n_outliers += 1
        data[:, i] = grubbs.removeNans(data[:, i])
    assert(len(featureset) > 0)
    features = np.empty(len(featureset), dtype=np.float64)
    """
    for f in featureset.shared:
        f.process(data)
    """
    k, l = 0, 0
    for f in featureset.getFeatures():
        l += len(f)
        features[k:l] = f.process(data)
        k += len(f)
    return features


def closestExponent(step):
    exponent = 1
    while (exponent << 1) < step:
        exponent = exponent << 1
    return exponent


def process(dataset, featureset):
    n_samples = 20
    step = len(dataset["data"]) / n_samples
    # win_size = closestExponent(step)
    win_size = step
    data = dataset["data"]
    assert(len(data) == 240000)
    n_features = len(featureset)
    obs = np.zeros((n_samples, n_features), dtype=np.float)
    for i in range(n_samples):
        features = extractFeatures(
            dataset, featureset, i * step, i * step + win_size)
        if len(features) > 0:
            obs[i] = features
        else:
            if 0 < i:
                obs[i] = obs[i - 1]
    return obs


def accuracy(predictions, ys):
    acc = 0.0
    for i in range(len(ys)):
        acc += 1 - np.abs(predictions[i] - ys[i])
    return acc / len(ys)


def MCC(predictions, ys):
    """ Matthew's Correlation Coefficient """
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(predictions)):
        if predictions[i] == 1:
            if ys[i] == 1:
                TP += 1
            else:
                FP += 1
        elif predictions[i] == 0.5:
            pass
        else:
            if ys[i] == 1:
                FN += 1
            else:
                TN += 1
    mcc = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if mcc != 0:
        mcc = (TP * TN + FP * FN) / mcc
    return mcc


def removeCorruptedFiles():
    safe_labels_file = open(
        os.path.join(
            DATA_PATH,
            "train_and_test_data_labels_safe.csv"),
        "r")
    spamreader = csv.reader(safe_labels_file)
    spamreader.next()
    for line in spamreader:
        filename = line[0]
        is_safe = int(line[2])
        if not is_safe:
            patient_id = int(filename.split('_')[0])
            filepath = os.path.join(
                DATA_PATH, "Train/train_%i/%s" %
                (patient_id, filename))
            if os.path.isfile(filepath):
                os.remove(filepath)

    safe_labels_file.close()


def preprocessDataset(filepaths, labels, featureset):
    n_files = len(filepaths)
    n_features = len(featureset)
    inputs, outputs = list(), list()
    all_dropout_rates = list()
    for i in range(n_files):
        filepath = filepaths[i]
        print("Processing file number %i" % i)
        mat = loadmat(filepath)
        data = process(matToDataset(mat), featureset)
        data = np.asarray(data, dtype=np.float32)

        """
        masked = np.ma.masked_array(data, np.isnan(data))
        loc = np.mean(masked, axis = 1)
        scale = np.std(masked, axis = 1)
        for j in range(data.shape[1]):
            data[:, j] -= loc
            data[:, j] /= scale
        """

        label = labels[i]
        if label == 1:
            output = np.ones(len(data), dtype=np.int32)
        elif label == 0:
            output = np.zeros(len(data), dtype=np.int32)
        else:
            raise ValueError("Label is not 0 or 1")
        print("Label : %i" % label)
        inputs.append(data)
        outputs.append(output)

    return inputs, outputs
