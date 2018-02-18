# -*- coding: utf-8 -*-
# predict.py
# author : Antoine Passemiers

from spectral import *
from features import *
from utils import *

# https://github.com/AntoinePassemiers/ArchMM
from archmm.core import AdaptiveHMM, IOConfig

import sys
import random
import theano

from utils import *
import os
import time
import pickle
import numpy as np


np.seterr(invalid='warn')


def pickleMatFiles():
    folders = ["New_test/test_3_new"]
    for folder in folders:
        directory = os.path.join(DATA_PATH, folder)
        dest_dir = os.path.join(DATA_PATH, folder + "_pkl")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        i = 0
        for filename in os.listdir(directory):
            try:
                dataset = matToDataset(
                    loadmat(
                        os.path.join(
                            directory,
                            filename)))
                dataset["data"] = downSample(dataset["data"], 400, 200)
                pickle.dump(
                    dataset,
                    open(
                        os.path.join(
                            dest_dir,
                            filename.split(".")[0]),
                        "wb"))
                print("File number %i processed : %s" % (i, filename))
            except BaseException:
                pass
            i += 1


def train(n_files):
    featureset = FeatureSet(16, fs=400)
    featureset.add(FeatureSTE())
    featureset.add(FeatureZeroCrossings())
    featureset.add(
        FeatureSpectralCoherence().config(
            architecture="circular",
            band="all"))
    featureset.add(FeatureHurstExponent())
    featureset.add(FeatureLyapunovExponent())
    """
    featureset.add(FeatureSpectralCoherence().config(architecture = "circular", band = "delta"))
    featureset.add(FeatureSpectralCoherence().config(architecture = "circular", band = "theta"))
    featureset.add(FeatureSpectralCoherence().config(architecture = "circular", band = "alpha"))
    featureset.add(FeatureSpectralCoherence().config(architecture = "circular", band = "beta"))
    featureset.add(FeatureSpectralCoherence().config(architecture = "circular", band = "gamma"))
    featureset.add(FeatureSpectralCoherence().config(architecture = "circular", band = "high gamma"))
    """
    """
    inputs, labels, outputs, all_dropout_rates = preprocessDataset(n_files, featureset)
    temp_dir = os.path.join(DATA_PATH, "temp")
    pickle.dump((inputs, labels, outputs, all_dropout_rates), open(os.path.join(temp_dir, "temp_file"), "wb"))
    """
    temp_dir = os.path.join(DATA_PATH, "temp")
    inputs, labels, outputs, all_dropout_rates = pickle.load(
        open(os.path.join(temp_dir, 'temp_file'), "rb"))

    config = IOConfig()
    config.n_iterations = 50
    config.pi_learning_rate = 0.01
    config.pi_nhidden = 30
    config.pi_nepochs = 2
    config.s_learning_rate = 0.01
    config.s_nhidden = 30
    config.s_nepochs = 2
    config.o_learning_rate = 0.01
    config.o_nhidden = 30
    config.o_nepochs = 2
    config.missing_value_sym = np.nan
    iohmm = AdaptiveHMM(5, has_io=True)

    model_path = os.path.join(DATA_PATH, "model")
    classifiers = customBoost(inputs, labels, 1, 5, config, model_path)

    """
    for i in range(4):
        np.save(open("iohmm_training_%i" % i, "wb"), fit[i])
    pickle.dump(filenames[:len(ids)], open("sequence_names", "wb"))
    """
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(inputs)):
        prediction = classifiers[0].predictIO(
            inputs[i], binary_prediction=True)
        label = labels[i]
        if prediction == 1:
            if label == 1:
                TP += 1
            else:
                FP += 1
        else:
            if label == 1:
                FN += 1
            else:
                TN += 1
        print(labels[i], prediction, all_dropout_rates[i])
    print("True positives : %i" % TP)
    print("False positives : %i" % FP)
    print("True negatives : %i" % TN)
    print("False negatives : %i" % FN)
    mcc, positive_rate, negative_rate = MCC(TP, FP, TN, FN)
    print("True positive rate : %f" % positive_rate)
    print("True negative rate : %f" % negative_rate)
    print("MCC : %f" % mcc)


def main():
    removeCorruptedFiles()
    train(25)
    pickleMatFiles()


if __name__ == "__main__":
    predictAll()
    print("Finished")
