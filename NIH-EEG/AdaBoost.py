import numpy as np
import os

from HMM_Core import AdaptiveHMM, IOConfig


def customBoost(inputs, targets, n_boost, n_states, io_config, model_path):
    classifiers = list()
    weights = np.ones(len(targets), dtype = np.float64) / len(targets)
    target_sequences = list()
    for j in range(len(targets)):
        if targets[j] == 0:
            target_sequences.append(np.zeros(len(inputs[j])))
        else:
            target_sequences.append(np.ones(len(inputs[j])))
    shift = 4
    for i in range(n_boost):
        iohmm = AdaptiveHMM(n_states, has_io = True)
        if not os.path.isfile(os.path.join(model_path, "classifier_%i" % i)):
            iohmm.fit(inputs[i*shift:(i+1)*shift], targets = target_sequences[i*shift:(i+1)*shift], 
                      weights = weights, n_classes = 2,
                        is_classifier = True, parameters = io_config)
            iohmm.pySave(os.path.join(model_path, "classifier_%i" % i))
        else:
            iohmm.pyLoad(os.path.join(model_path, "classifier_%i" % i))
        classifiers.append(iohmm)
    return classifiers

def customBoostPredict(input, classifiers):
    result = 0
    for i in range(len(classifiers)):
        prediction = classifiers[i].predictIO(input)
        result += prediction
    print(result)
    return result > (float(len(classifiers)) / 2)