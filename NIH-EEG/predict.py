# -*- coding: utf-8 -*-

import sys, random
sys.path.insert(0, 'C:\Users\Xanto183\git\ArchMM\ArchMM\ArchMM\CyFiles')
from utils import *
import os, time
import numpy as np
from HMM_Core import AdaptiveHMM
from Spectral import *


from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer


def predict(n_files):
    train_X, train_Y = [], []
    hmms_0, hmms_1 = [], []
    directory = os.path.join(MODEL_PATH, "HMMs")
    for i in range(5):
        for j in range(6):
            hmm = AdaptiveHMM(0, "", standardize = True)
            hmm.pyLoad(os.path.join(directory, "hmm_%i_%i_%i" % (i, j, 0)))
            hmms_0.append(hmm)
            hmm = AdaptiveHMM(0, "", standardize = True)
            hmm.pyLoad(os.path.join(directory, "hmm_%i_%i_%i" % (i, j, 1)))
            hmms_1.append(hmm)
    
    directory = os.path.join(DATA_PATH, "Train/train_1")
    filenames = os.listdir(directory)
    rd = range(len(filenames))
    np.random.shuffle(rd)
    for i in range(n_files):
        print("Processing file number %i\n" % i)
        filename = filenames[rd[i]]
        filepath = os.path.join(directory, filename)
        mat = loadmat(filepath)
        data = a(matToDataset(mat))
        temp = []
        for k in range(len(hmms_0)):
            s1 = hmms_0[k].score(data)
            print("Cost of t0 on interictal model : %f" % s1)
            s2 = hmms_1[k].score(data)
            print("Cost of t0 on preictal model : %f" % s2)
            temp.append(s1 / s2)
        train_X.append(temp)
        label = int((filename.split(".")[0]).split("_")[2])
        train_Y.append(label)
        print("")
    return train_X, train_Y

def randomTestset(n_files):
    directory = os.path.join(DATA_PATH, "Train/train_1")
    filenames = os.listdir(directory)
    rd = range(len(filenames))
    np.random.shuffle(rd)
    train_X, train_Y = [], []
    for i in range(n_files):
        filename = filenames[rd[i]]
        filepath = os.path.join(directory, filename)
        mat = loadmat(filepath)
        data = downSample(matToDataset(mat)["data"], 400, 40)
        train_X.append(data)
        label = int((filename.split(".")[0]).split("_")[2])
        train_Y.append(label)
    return train_X, train_Y

def main():
    train_X, train_Y = randomTestset(4)
    directory = os.path.join(MODEL_PATH, "HMMs")
    hmm = AdaptiveHMM(0, "", standardize = True)
    hmm.pyLoad(os.path.join(directory, "hmm_%i_%i_%i" % (0, 0, 0)))
    hmm2 = AdaptiveHMM(0, "", standardize = True)
    hmm2.pyLoad(os.path.join(directory, "hmm_%i_%i_%i" % (0, 0, 1)))
    hmm3 = AdaptiveHMM(0, "", standardize = True)
    hmm3.pyLoad(os.path.join(directory, "hmm_%i_%i_%i" % (0, 1, 0)))
    hmm4 = AdaptiveHMM(0, "", standardize = True)
    hmm4.pyLoad(os.path.join(directory, "hmm_%i_%i_%i" % (0, 1, 1)))
    for i in range(4):
        print(hmm3.score(train_X[i]))
        print(hmm4.score(train_X[i]))
        print(train_Y[i])
        print("")
    

if __name__ == "__main__":
    main()
    print("Finished")