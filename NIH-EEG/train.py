# -*- coding: utf-8 -*-

import sys
import theano
sys.path.insert(0, 'C://Users/Xanto183/git/ArchMM/ArchMM/')
sys.path.insert(0, 'C://Users/Xanto183/git/ArchMM/ArchMM/Cyfiles/')
from utils import *
import os, pickle
import numpy as np

from HMM_Core import AdaptiveHMM, IOConfig
from features import *
from AdaBoost import *

NUM_EXAMPLES_BY_MODEL = 64
MIN_FILE_SIZE = 4500000

featureset = FeatureSet(16, fs = 400)
featureset.add(FeatureLogSpectrum())
"""
featureset.add(FeatureSTE())
featureset.add(FeatureZeroCrossings())
featureset.add(FeatureSpectralCoherence().config(architecture = "circular", band = "all"))
featureset.add(FeatureSpectralEntropy())
"""

def showNaNs(input):
    N = len(input)
    nan_counts = np.empty(input.shape[1], dtype = np.float)
    for i in range(input.shape[1]):
        nan_counts[i] = float(np.count_nonzero(np.isnan(input[:, i]))) / float(N)
    print(nan_counts.max())

def pickleFiles(use_test_set = True):
    if not use_test_set:
        source_folder_base = "Train/train_%i"
        dest_folder_base = "Train/pklcnn_train_%i"
    else:
        source_folder_base = "New_test/test_%i_new"
        dest_folder_base = "New_test/pklcnn_test_%i_new"
    for k in range(3):
        directory = os.path.join(DATA_PATH, source_folder_base % (k + 1))
        dest_directory = os.path.join(DATA_PATH, dest_folder_base % (k + 1))
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        for filename in os.listdir(directory):
            dest_filepath = os.path.join(dest_directory, filename).split(".")[0]
            if not os.path.isfile(dest_filepath):
                if use_test_set:
                    label = 1
                else:
                    label = int(filename.split('_')[2][0])
                filepath = os.path.join(directory, filename)
                inputs, outputs = preprocessDataset([filepath], [label], featureset)
                data = [inputs[0], outputs[0]]
                pickle.dump(data, open(dest_filepath, "wb"))
                
def randomValidationSet(patient_id, n_files):
    filepaths = list()
    labels = list()
    directory = os.path.join(DATA_PATH, "Train/pkl_train_%i" % patient_id)
    for filename in os.listdir(directory):
        label = int(filename.split('_')[2][0])
        labels.append(label)
        filepath = os.path.join(directory, filename)
        filepaths.append(filepath)
    rd_ids = list(range(n_files))
    random.shuffle(rd_ids)
    rd_filepaths, rd_labels = list(), list()
    for id in rd_ids:
        rd_filepaths.append(filepaths[id])
        rd_labels.append(labels[id])
    inputs, outputs = list(), list()
    for i in range(n_files):
        input = pickle.load(open(rd_filepaths[i], "rb"))[0]
        inputs.append(input)
        if rd_labels[i] == 1:
            outputs.append(np.ones(len(input)))
        else:
            outputs.append(np.zeros(len(input)))
    return inputs, outputs

def randomTrainingFilenames(patient_id, n_files = None):
    p_filenames, n_filenames = list(), list()
    directory = os.path.join(DATA_PATH, "Train/pkl_train_%i" % patient_id)
    for filename in os.listdir(directory):
        if filename.split('_')[2][0] == '1':
            filepath = os.path.join(directory, filename)
            p_filenames.append(filepath)
    n_pairs = 0
    for filename in os.listdir(directory):
        if filename.split('_')[2][0] == '0':
            filepath = os.path.join(directory, filename)
            n_filenames.append(filepath)
            n_pairs += 1
            if n_pairs == len(p_filenames):
                break
    return p_filenames, n_filenames

def main(): 
    p_filenames, n_filenames = list(), list()
    for k in range(1):
        p_temp, n_temp = randomTrainingFilenames(k + 1, None)
        p_filenames.append(p_temp)
        n_filenames.append(n_temp)
            
    n_states = 5
    config = IOConfig()
    config.n_iterations = 50
    config.pi_learning_rate = 0.005
    config.pi_nhidden = 100
    config.pi_nepochs = 2
    config.pi_activation = "sigmoid"
    config.s_learning_rate  = 0.005
    config.s_nhidden  = 100
    config.s_nepochs = 2
    config.s_activation = "sigmoid"
    config.o_learning_rate  = 0.005
    config.o_nhidden  = 100
    config.o_nepochs = 2
    config.o_activation = "sigmoid"
    config.missing_value_sym = np.nan
    model_path = os.path.join(DATA_PATH, "model")
    model_id = 0
    classifiers = list()
    # for k in range(3):
    for k in range(1):
        print("Processing patient %i" % (k + 1))
        patient_classifiers = list()
        n_classifiers_for_this_patient = 1
        # n_classifiers_for_this_patient = 2 * len(p_filenames[k]) / NUM_EXAMPLES_BY_MODEL
        for j in range(n_classifiers_for_this_patient):
            filepaths  = p_filenames[k][j*NUM_EXAMPLES_BY_MODEL/2:(j+1)*NUM_EXAMPLES_BY_MODEL/2]
            filepaths += n_filenames[k][j*NUM_EXAMPLES_BY_MODEL/2:(j+1)*NUM_EXAMPLES_BY_MODEL/2]
            labels = [1] * (NUM_EXAMPLES_BY_MODEL / 2) + [0] * (NUM_EXAMPLES_BY_MODEL / 2)
            rd_ids = list(range(len(filepaths)))
            random.shuffle(rd_ids)
            rd_filepaths, rd_labels = list(), list()
            for id in rd_ids:
                rd_filepaths.append(filepaths[id])
                rd_labels.append(labels[id])
            labels = rd_labels
            filepaths = rd_filepaths
            iohmm = AdaptiveHMM(n_states, has_io = True)
            if not os.path.isfile(os.path.join(model_path, "classifier_%i" % model_id)):
                inputs, outputs = list(), list()
                for i in range(len(filepaths)):
                    input = pickle.load(open(filepaths[i], "rb"))[0]
                    showNaNs(input)
                    inputs.append(input)
                np.save(open("features", "wb"), inputs)
                target_sequences = list()
                for j in range(len(labels)):
                    if labels[j] == 0:
                        target_sequences.append(np.zeros(len(inputs[j])))
                    else:
                        target_sequences.append(np.ones(len(inputs[j])))
                fit = iohmm.fit(inputs, targets = target_sequences, n_classes = 2,
                            is_classifier = True, parameters = config)
                for i in range(5):
                    np.save(open("iohmm_training_%i" % i, "wb"), fit[i])
                pickle.dump(labels, open('sequence_info', "wb"))
                iohmm.pySave(os.path.join(model_path, "classifier_%i" % model_id))
                
                inputs, outputs = randomValidationSet(k + 1, 500)
                predictions, ys = list(), list()
                for i in range(len(inputs)):
                    prediction = iohmm.predictIO(inputs[i])[0]
                    predictions.append(prediction)
                    ys.append(outputs[i][-1])
                    print(prediction, outputs[i][-1])
                print("MCC : %f" % float(MCC(predictions, ys)))
                return
                
            else:
                iohmm.pyLoad(os.path.join(model_path, "classifier_%i" % model_id))
                print("Classifier %i loaded" % model_id)
            patient_classifiers.append(iohmm)
            model_id += 1
            
        classifiers.append(patient_classifiers)
    return classifiers
            
def test():
    classifiers = main()
    sample_submission = csv.reader(open(os.path.join(DATA_PATH, "sample_submission.csv"), "r"))
    sample_submission.next()
    submission = open(os.path.join(DATA_PATH, "submission.csv"), "w")
    submission.write("File,Class\n")
    for line in sample_submission:
        filename = line[0]
        patient_id = int(filename.split('_')[1])
        directory = os.path.join(DATA_PATH, "New_test/pkl_test_%i_new" % patient_id)
        filepath = os.path.join(directory, filename.split('.')[0])
        data = pickle.load(open(filepath, "rb"))
        input, label = data[0], data[1][-1]
        if patient_id != 2:
            prediction = classifiers[0][0].predictIO(input, binary_prediction = True)[0]
        else:
            prediction = 0
        submission.write("%s,%i\n" % (filename, prediction))
        print(prediction)
        
if __name__ == "__main__":
    # main()
    pickleFiles(use_test_set = False)
    print("Finished")
    
    