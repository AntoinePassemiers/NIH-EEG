# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'C:\Users\Xanto183\git\ArchMM\ArchMM\ArchMM\CyFiles')
from utils import *
import os, time
import numpy as np
from HMM_Core import AdaptiveHMM

""" TODO
- Manage electrode drop-outs
- Remove artifacts from the input signals
"""

PREICTAL_LABEL = 1
INTERICTAL_LABEL = 0

class MyAppValueError(ValueError):
    '''Raise when a specific subset of values in context of app is wrong'''
    def __init__(self, message, foo, *args):
        self.message = message # without this you may get DeprecationWarning
        # Special attribute you desire with your Error, 
        # perhaps the value that caused the error?:
        self.foo = foo         
        # allow users initialize misc. arguments as any other builtin Error
        super(MyAppValueError, self).__init__(message, foo, *args) 

def main(): 
    directory = os.path.join(MODEL_PATH, "HMMs")
    if not os.path.exists(directory):
        os.makedirs(directory)
    """
    for i in range(5):
        DS_0 = getDataset(1, i, 0)
        DS_1 = getDataset(1, i, 1)
        DS = [DS_0, DS_1]
        for j in range(6):
            for k in range(2):
                dt = a(DS[k][j])
                d0, d1 = downSample(DS_0[j]["data"], 400, 40), downSample(DS_1[j]["data"], 400, 40)
                hmm = AdaptiveHMM(10, "ergodic", standardize = True)
                hmm.fit(d0, dynamic_features = False)
                hmm.pySave(os.path.join(directory, "hmm_%i_%i_%i" % (i, j, 0)))
                hmm2 = AdaptiveHMM(10, "ergodic", standardize = True)
                hmm2.fit(d1, dynamic_features = False)
                hmm2.pySave(os.path.join(directory, "hmm_%i_%i_%i" % (i, j, 1)))
    """
    D_0_0 = a(getFileDataset(1, 0, 0))
    print("ok")
    D_1_0 = a(getFileDataset(1, 2, 0))
    print("ok")
    D_1_1 = a(getFileDataset(1, 2, 1))
    
    print("")
    
    """
    for i in [3, 5, 8, 10, 15, 20]:
        hmm = AdaptiveHMM(i, "ergodic", standardize = True)
        hmm.fit(D_0_0, dynamic_features = False)
        print("\n%i" % i)
        print(hmm.score(D_0_0))
        print(hmm.score(D_1_0))
        print(hmm.score(D_1_1))
    """ 
    
    
        
if __name__ == "__main__":
    main()
    print("Finished")
    
    