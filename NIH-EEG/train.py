# -*- coding: utf-8 -*-

import sys
import theano
sys.path.insert(0, 'C://Users/Xanto183/git/ArchMM/ArchMM/')
sys.path.insert(0, 'C://Users/Xanto183/git/ArchMM/ArchMM/Cyfiles/')
from utils import *
import os, time
import numpy as np

from HMM_Core import AdaptiveHMM, IOConfig
from features import *

NUM_EXAMPLES_BY_MODEL = 4
MIN_FILE_SIZE = 4500000

def main(): 
    directory = os.path.join(MODEL_PATH, "HMMs")
    if not os.path.exists(directory):
        os.makedirs(directory)
    p_filenames, n_filenames = list(), list()
    for k in range(3):
        p_temp, n_temp = list(), list()
        directory = os.path.join(DATA_PATH, "Train/train_%i" % (k + 1))
        for filename in os.listdir(directory):
            if filename.split('_')[2][0] == '1':
                filepath = os.path.join(directory, filename)
                filesize = os.path.getsize(filepath)
                if filesize > MIN_FILE_SIZE:
                    p_temp.append(filepath)
        n_pairs = 0
        for filename in os.listdir(directory):
            if filename.split('_')[2][0] == '0':
                filepath = os.path.join(directory, filename)
                filesize = os.path.getsize(filepath)
                if filesize > MIN_FILE_SIZE:
                    n_temp.append(filepath)
                    n_pairs += 1
                    if n_pairs == len(p_filenames):
                        break
        p_filenames.append(p_temp)
        n_filenames.append(n_temp)
            
    n_states = 5        
    config = IOConfig()
    config.n_iterations = 50
    config.pi_learning_rate = 0.005
    config.pi_nhidden = 40
    config.pi_nepochs = 2
    config.s_learning_rate  = 0.005
    config.s_nhidden  = 40
    config.s_nepochs = 2
    config.o_learning_rate  = 0.005
    config.o_nhidden  = 40
    config.o_nepochs = 2
    config.missing_value_sym = np.nan
    featureset = FeatureSet(16, fs = 400)
    featureset.add(FeatureSTE())
    featureset.add(FeatureZeroCrossings())
    featureset.add(FeatureSpectralCoherence().config(architecture = "circular", band = "all"))
    featureset.add(FeatureSpectralEntropy())
    model_path = os.path.join(DATA_PATH, "model")
    model_id = 0
    classifiers = list()
    for k in range(3):
        for j in range(2 * len(p_filenames[k]) / NUM_EXAMPLES_BY_MODEL):
            filepaths  = p_filenames[k][j*NUM_EXAMPLES_BY_MODEL/2:(j+1)*NUM_EXAMPLES_BY_MODEL/2]
            filepaths += n_filenames[k][j*NUM_EXAMPLES_BY_MODEL/2:(j+1)*NUM_EXAMPLES_BY_MODEL/2]
            labels = [1] * (NUM_EXAMPLES_BY_MODEL / 2) + [0] * (NUM_EXAMPLES_BY_MODEL / 2)
            rd_ids = list(range(NUM_EXAMPLES_BY_MODEL))
            random.shuffle(rd_ids)
            rd_filepaths, rd_labels = list(), list()
            for id in rd_ids:
                rd_filepaths.append(filepaths[id])
                rd_labels.append(labels[id])
            labels = rd_labels
            filepaths = rd_filepaths
            iohmm = AdaptiveHMM(n_states, has_io = True)
            print(filepaths)
            if not os.path.isfile(os.path.join(model_path, "classifier_%i" % model_id)):
                inputs, outputs, all_dropout_rates = preprocessDataset(filepaths, labels, featureset)
                np.save(open("features", "wb"), inputs)
                target_sequences = list()
                for j in range(len(labels)):
                    if labels[j] == 0:
                        target_sequences.append(np.zeros(len(inputs[j])))
                    else:
                        target_sequences.append(np.ones(len(inputs[j])))
                fit = iohmm.fit(inputs, targets = target_sequences, n_classes = 2,
                            is_classifier = True, parameters = config)
                for i in range(4):
                    np.save(open("iohmm_training_%i" % i, "wb"), fit[i])
                info = [(label, dropout) for label, dropout in zip(labels, all_dropout_rates)]
                pickle.dump(info, open('sequence_info', "wb"))
                iohmm.pySave(os.path.join(model_path, "classifier_%i" % model_id))
                # iohmm = AdaptiveHMM(n_states, has_io = True)
                # iohmm.pyLoad(os.path.join(model_path, "classifier_%i" % model_id))
                for i in range(len(inputs)):
                    print(iohmm.predictIO(inputs[i]), labels[i])
            else:
                iohmm.pyLoad(os.path.join(model_path, "classifier_%i" % model_id))
                print("Classifier %i loaded" % model_id)
            classifiers.append(iohmm)
            model_id += 1
            
def test():
    featureset = FeatureSet(16, fs = 400)
    featureset.add(FeatureSTE())
    featureset.add(FeatureZeroCrossings())
    featureset.add(FeatureSpectralCoherence().config(architecture = "circular", band = "all"))
    featureset.add(FeatureSpectralEntropy())
    
    labels = [0, 1, 1, 0]
    filenames = ["1_110_1.mat", "1_10_1.mat", "1_1029_0.mat", "1_1028_0"]
    filepaths = [os.path.join(DATA_PATH, "Train/train_1/" + filename) for filename in filenames]
    model_path = os.path.join(DATA_PATH, "model")
    
    inputs, outputs, all_dropout_rates = preprocessDataset(filepaths, labels, featureset)
    
    iohmm = AdaptiveHMM(5, has_io = True)
    iohmm.pyLoad(os.path.join(model_path, "classifier_%i" % 0))
    for i in range(len(labels)):
        print(iohmm.predictIO(inputs[0]), labels[i])
        
if __name__ == "__main__":
    test()
    print("Finished")
    
    