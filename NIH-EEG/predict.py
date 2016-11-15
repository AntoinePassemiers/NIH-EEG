# -*- coding: utf-8 -*-

import sys, random
import theano
sys.path.insert(0, 'C://Users/Xanto183/git/ArchMM/ArchMM/')
sys.path.insert(0, 'C://Users/Xanto183/git/ArchMM/ArchMM/Cyfiles/')
from utils import *
import os, time, pickle
import numpy as np

from HMM_Core import AdaptiveHMM, IOConfig
from Spectral import *
from features import *
from AdaBoost import *

# https://github.com/MichaelHills/seizure-prediction
# https://en.wikipedia.org/wiki/Hurst_exponent

np.seterr(invalid = 'warn')

def pickleMatFiles():
    folders = ["New_test/test_3_new"]
    for folder in folders:
        directory = os.path.join(DATA_PATH, folder)
        dest_dir  = os.path.join(DATA_PATH, folder + "_pkl")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        i = 0
        for filename in os.listdir(directory):
            try:
                dataset = matToDataset(loadmat(os.path.join(directory, filename)))
                dataset["data"] = downSample(dataset["data"], 400, 200)
                pickle.dump(dataset, open(os.path.join(dest_dir, filename.split(".")[0]), "wb"))
                print("File number %i processed : %s" % (i, filename))
            except:
                pass
            i += 1
            
def preprocessDataset(n_files, featureset):
    directory = os.path.join(DATA_PATH, "Train/train_1")
    filenames = os.listdir(directory)
    random.shuffle(filenames)
    training_files = ["1_77_1.mat", "1_89_0.mat", "1_16_1.mat", "1_23_0.mat",
                      "1_992_0.mat", "1_117_1.mat", "1_81_1.mat", "1_556_0.mat",
                      "1_302_0.mat", "1_91_1.mat", "1_812_0.mat", "1_140_1.mat"]
    filenames = training_files + filenames
    n_features = len(featureset)
    inputs, outputs = list(), list()
    all_means = np.empty((n_files, n_features))
    all_stds  = np.empty((n_files, n_features))
    all_dropout_rates = list()
    labels = list()
    for i in range(n_files):
        filename = filenames[i]
        print("Processing file number %i : %s" % (i, filename))
        filepath = os.path.join(directory, filename)
        mat = loadmat(filepath)
        data, dropout_rate = process(matToDataset(mat), featureset)
        all_dropout_rates.append(dropout_rate)
        masked = np.ma.masked_array(data, np.isnan(data))
        all_means[i, :] = np.mean(masked, axis = 0)
        all_stds[i, :] = np.std(masked, axis = 0)
        label = int((filename.split(".")[0]).split("_")[2])
        labels.append(label)
        if label == 1:
            output = np.ones(len(data), dtype = np.int32)
        elif label == 0:
            output = np.zeros(len(data), dtype = np.int32)
        else:
            raise ValueError("Label is not 0 or 1")
        print("Label : %i" % label)
        inputs.append(data)
        outputs.append(output)
    global_mean = all_means.mean(axis = 0)
    global_std  = all_stds.mean(axis = 0)
    for i in range(n_files):
        for j in range(len(global_std)):
            inputs[i][:, j] -= global_mean[j]
            inputs[i][:, j] /= global_std[j]
    return inputs, labels, outputs, all_dropout_rates

def train(n_files):
    featureset = FeatureSet(16, fs = 400)
    featureset.add(FeatureSTE())
    featureset.add(FeatureZeroCrossings())
    featureset.add(FeatureSpectralCoherence().config(architecture = "circular", band = "all"))
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
    inputs, labels, outputs, all_dropout_rates = pickle.load(open(os.path.join(temp_dir, 'temp_file'), "rb"))
    
    
    config = IOConfig()
    config.n_iterations = 30
    config.pi_learning_rate = 0.005
    config.pi_nhidden = 30
    config.pi_nepochs = 2
    config.s_learning_rate  = 0.005
    config.s_nhidden  = 30
    config.s_nepochs = 2
    config.o_learning_rate  = 0.005
    config.o_nhidden  = 30
    config.o_nepochs = 2
    config.missing_value_sym = np.nan
    iohmm = AdaptiveHMM(5, has_io = True)
    
    model_path = os.path.join(DATA_PATH, "model")
    classifiers, alpha = AdaBoost(inputs, labels, 8, 5, config, model_path)
    
    """
    for i in range(4):
        np.save(open("iohmm_training_%i" % i, "wb"), fit[i])
    pickle.dump(filenames[:len(ids)], open("sequence_names", "wb"))
    """
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(inputs)):
        prediction = AdaBoostPredict(inputs[i], classifiers, alpha)
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
    train(25)
    # pickleMatFiles()
    

if __name__ == "__main__":
    main()
    print("Finished")