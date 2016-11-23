import numpy as np
import os

def AdaBoost(classifiers, inputs, target_sequences):
    n_boost = len(classifiers)
    Y = np.empty(len(target_sequences))
    for i in range(len(target_sequences)):
        Y[i] = target_sequences[i][-1]
    Y[Y == 0] = -1
    alpha = np.empty(n_boost, dtype = np.float64)
    weights = np.ones(len(target_sequences), dtype = np.float64) / len(target_sequences)
    predictions = np.empty(len(target_sequences), dtype = np.int)
    for i in range(n_boost):
        for j in range(len(target_sequences)):
            predictions[j] = classifiers[i].predictIO(inputs[j], binary_prediction = True)[0]
            predictions[j] = -1 if predictions[j] == 0 else 1
        error = np.dot(weights, predictions != Y)
        print(error)
        print(0.5 * np.log(float(1 - error) / float(error)))
        alpha[i] = 0.5 * np.log(float(1 - error) / float(error))
        weights *= np.exp(- alpha[i] * Y * predictions)
        weights /= np.sum(weights)
    return alpha

def AdaBoostPredict(input, classifiers, alpha):
    result = 0
    for i in range(len(classifiers)):
        prediction = -1 if classifiers[i].predictIO(input) == 0 else 1
        result += alpha[i] * prediction
    return 1 if result > 0 else 0