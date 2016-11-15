import numpy as np
import os

from HMM_Core import AdaptiveHMM, IOConfig


def AdaBoost(inputs, targets, n_boost, n_states, io_config, model_path):
    target_sequences = list()
    for j in range(len(targets)):
        if targets[j] == 0:
            target_sequences.append(np.zeros(len(inputs[j])))
        else:
            target_sequences.append(np.ones(len(inputs[j])))
    Y = np.copy(targets)
    Y[Y == 0] = -1
    classifiers = list()
    alpha = np.empty(n_boost, dtype = np.float64)
    weights = np.ones(len(targets), dtype = np.float64) / len(targets)
    predictions = np.empty(len(targets), dtype = np.int)
    for i in range(n_boost):
        iohmm = AdaptiveHMM(n_states, has_io = True)
        if not os.path.isfile(os.path.join(model_path, "classifier_%i" % i)):
            iohmm.fit(inputs, targets = target_sequences, weights = weights, n_classes = 2,
                        is_classifier = True, parameters = io_config)
            iohmm.pySave(os.path.join(model_path, "classifier_%i" % i))
        else:
            iohmm.pyLoad(os.path.join(model_path, "classifier_%i" % i))
        classifiers.append(iohmm)
        for j in range(len(targets)):
            predictions[j] = iohmm.predictIO(inputs[j], binary_prediction = True)
            predictions[j] = -1 if predictions[j] == 0 else 1
        error = np.dot(weights, predictions != Y)
        print(error)
        print(0.5 * np.log(float(1 - error) / float(error)))
        alpha[i] = 0.5 * np.log(float(1 - error) / float(error))
        weights *= np.exp(- alpha[i] * Y * predictions)
        weights /= np.sum(weights)
    return classifiers, alpha

def AdaBoostPredict(input, classifiers, alpha):
    result = 0
    for i in range(len(classifiers)):
        prediction = -1 if classifiers[i].predictIO(input) == 0 else 1
    result += alpha[i] * prediction
    return 1 if result > 0 else 0