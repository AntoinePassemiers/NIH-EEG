# -*- coding: utf-8 -*-
# train.py
# author : Antoine Passemiers

from features import *
from configs import *
from utils import *

# https://github.com/AntoinePassemiers/ArchMM
from archmm.core import AdaptiveHMM

import sys
import theano
import os
import pickle
import numpy as np


MIN_FILE_SIZE = 4500000

featureset = FeatureSet(16, fs=400)
# featureset.add(FeatureLogSpectrum())
featureset.add(FeatureSTE())
featureset.add(FeatureZeroCrossings())
featureset.add(FeatureSpectralCoherence().config(architecture="full", band="all"))
featureset.add(FeatureSpectralEntropy())


def dropoutRate(inputs):
    T, n_features = inputs[0].shape
    n_files = len(inputs)
    dropouts = np.zeros((n_files, n_features), dtype=np.float)
    for i in range(n_files):
        for j in range(n_features):
            dropouts[i, j] = float(np.count_nonzero(
                np.isnan(inputs[i][:, j]))) / float(T)
    return dropouts.mean()


def showNaNs(input):
    N = len(input)
    nan_counts = np.empty(input.shape[1], dtype=np.float)
    for i in range(input.shape[1]):
        nan_counts[i] = float(np.count_nonzero(
            np.isnan(input[:, i]))) / float(N)
    print(nan_counts.max())


def pickleFiles(use_test_set=True):
    if not use_test_set:
        source_folder_base = "Train/train_%i"
        dest_folder_base = "Train/pkl_train_%i"
    else:
        source_folder_base = "New_test/test_%i_new"
        dest_folder_base = "New_test/pklfill_test_%i_new"
    for k in range(1, 2):
        directory = os.path.join(DATA_PATH, source_folder_base % (k + 1))
        dest_directory = os.path.join(DATA_PATH, dest_folder_base % (k + 1))
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        for filename in os.listdir(directory):
            dest_filepath = os.path.join(
                dest_directory, filename).split(".")[0]
            if not os.path.isfile(dest_filepath):
                if use_test_set:
                    label = 1
                else:
                    label = int(filename.split('_')[2][0])
                filepath = os.path.join(directory, filename)
                inputs, outputs = preprocessDataset(
                    [filepath], [label], featureset)
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
    return inputs, outputs, rd_filepaths


def randomTrainingFilenames(patient_id, n_files=None):
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


def main(validate=True):
    p_filenames, n_filenames = list(), list()
    for k in range(0, 3):
        p_temp, n_temp = randomTrainingFilenames(k + 1, None)
        p_filenames.append(p_temp)
        n_filenames.append(n_temp)

    model_path = os.path.join(DATA_PATH, "model")
    classifiers = list()

    for k in range(0, 3):
        model_id = k
        print("Processing patient %i" % (k + 1))
        config = io_configs[k]
        NUM_EXAMPLES_BY_MODEL = config.n_examples
        NUM_PREICTAL_EXAMPLES_BY_MODEL = int(
            NUM_EXAMPLES_BY_MODEL * config.preictal_proportion)
        NUM_INTERICTAL_EXAMPLES_BY_MODEL = NUM_EXAMPLES_BY_MODEL - \
            NUM_PREICTAL_EXAMPLES_BY_MODEL
        patient_classifiers = list()
        n_classifiers_for_this_patient = config.n_classifiers
        for j in range(n_classifiers_for_this_patient):
            filepaths = p_filenames[k][j *
                                       NUM_PREICTAL_EXAMPLES_BY_MODEL:(j +
                                                                       1) *
                                       NUM_PREICTAL_EXAMPLES_BY_MODEL]
            filepaths += n_filenames[k][j *
                                        NUM_INTERICTAL_EXAMPLES_BY_MODEL:(j +
                                                                          1) *
                                        NUM_INTERICTAL_EXAMPLES_BY_MODEL]
            labels = [1] * NUM_PREICTAL_EXAMPLES_BY_MODEL + \
                [0] * NUM_INTERICTAL_EXAMPLES_BY_MODEL
            rd_ids = list(range(len(filepaths)))
            random.shuffle(rd_ids)
            rd_filepaths, rd_labels = list(), list()
            for id in rd_ids:
                rd_filepaths.append(filepaths[id])
                rd_labels.append(labels[id])
            labels = rd_labels
            filepaths = rd_filepaths
            iohmm = AdaptiveHMM(config.n_states, has_io=True)
            if not os.path.isfile(
                os.path.join(
                    model_path, "classifier_%i_%i" %
                    (model_id, j))):
                inputs, outputs = list(), list()
                for i in range(len(filepaths)):
                    input = pickle.load(open(filepaths[i], "rb"))[0]
                    showNaNs(input)
                    inputs.append(input)
                np.save(open("features", "wb"), inputs)
                target_sequences = list()
                for l in range(len(labels)):
                    if labels[l] == 0:
                        target_sequences.append(np.zeros(len(inputs[l])))
                    else:
                        target_sequences.append(np.ones(len(inputs[l])))
                fit = iohmm.fit(inputs, targets=target_sequences, n_classes=2,
                                is_classifier=True, parameters=config)
                for i in range(5):
                    np.save(open("iohmm_training_%i" % i, "wb"), fit[i])
                pickle.dump(labels, open('sequence_info', "wb"))
                iohmm.pySave(
                    os.path.join(
                        model_path, "classifier_%i_%i" %
                        (model_id, j)))

            else:
                iohmm.pyLoad(
                    os.path.join(
                        model_path, "classifier_%i_%i" %
                        (model_id, j)))
                print("Classifier %i - %i loaded" % (model_id, j))

            patient_classifiers.append(iohmm)

        if validate:
            validation_results = open(
                os.path.join(
                    DATA_PATH,
                    "validation_%i.txt" %
                    k),
                "w")
            inputs, outputs, filenames = randomValidationSet(k + 1, 500)
            predictions, ys = list(), list()
            for i in range(len(inputs)):
                label = outputs[i][-1]
                validation_results.write(
                    "Patient : %i / Label : %i\n" %
                    (k, int(label)))
                all_predictions, all_results = list(), list()
                for j in range(n_classifiers_for_this_patient):
                    result = patient_classifiers[j].predictIO(inputs[i])
                    prediction = result[0]
                    validation_results.write("%s\n" % str(result[1]))
                    validation_results.write("%s\n" % str(result[2]))
                    all_predictions.append(prediction)
                    all_results.append(result)

                if n_classifiers_for_this_patient == 1:
                    prediction = all_predictions[0]
                    result = all_results[0]
                    if j == 2:
                        if len(np.nonzero(np.diff(result[2][0]))[0]) <= 1 and \
                                len(np.nonzero(np.diff(result[2][1]))[0]) >= 1:
                            prediction = 0
                else:
                    p_points, n_points = 0, 0
                    global_memory = np.zeros(2)
                    for i in range(n_classifiers_for_this_patient):
                        if len(np.nonzero(np.diff(all_results[i][2][0]))[0]) == 0 and \
                                len(np.nonzero(np.diff(all_results[i][2][1]))[0]) >= 1:
                            n_points += 2
                        if all_predictions[i] == 1:
                            p_points += 1
                        else:
                            n_points += 1
                        global_memory += all_results[i][1][:, 0]
                    if global_memory[0] >= global_memory[1]:
                        n_points += 1
                    else:
                        p_points += 1
                    prediction = 0.0 if n_points >= p_points else 1.0

                predictions.append(prediction)
                ys.append(label)
                print(prediction, label)
                validation_results.write(
                    "%i, %i\n" %
                    (int(prediction), int(label)))
                print("")
                validation_results.write("\n")
            mcc = float(MCC(predictions, ys))
            print("MCC : %f" % mcc)
            validation_results.write(
                "\nMCC for patient %i : %f\n\n" %
                (k + 1, mcc))
            validation_results.close()

        classifiers.append(patient_classifiers)
    return classifiers


def test():
    classifiers = main(validate=False)
    sample_submission = csv.reader(
        open(
            os.path.join(
                DATA_PATH,
                "sample_submission.csv"),
            "r"))
    sample_submission.next()
    submission = open(os.path.join(DATA_PATH, "submission.csv"), "w")
    submission.write("File,Class\n")
    test_results = open(os.path.join(DATA_PATH, "test.txt"), "w")
    for line in sample_submission:
        filename = line[0]
        patient_id = int(filename.split('_')[1])
        directory = os.path.join(
            DATA_PATH,
            "New_test/pkl_test_%i_new" %
            patient_id)
        filepath = os.path.join(directory, filename.split('.')[0])
        data = pickle.load(open(filepath, "rb"))
        input, label = data[0], data[1][-1]

        if patient_id == 9:
            prediction1 = classifiers[0][0].predictIO(
                input, binary_prediction=True)[0]
            prediction2 = classifiers[1][0].predictIO(
                input, binary_prediction=True)[0]
            prediction3 = classifiers[2][0].predictIO(
                input, binary_prediction=True)[0]
            prediction = 1.0 / 3.0 * prediction1 + 1.0 / \
                3.0 * prediction2 + 1.0 / 3.0 * prediction3
        else:
            prediction = classifiers[patient_id -
                                     1][0].predictIO(input, binary_prediction=True)[0]
        submission.write("%s,%i\n" % (filename, prediction))
        print(prediction)
    test_results.close()


if __name__ == "__main__":
    test()
    # pickleFiles(use_test_set = True)
    print("Finished")
