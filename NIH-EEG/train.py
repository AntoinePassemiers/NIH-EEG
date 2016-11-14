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
- Downsample to 200%
- Divide the bandwidth in 6 parts : theta, alpha, low gamma, ...
- Use the wavelet coherence in place of the spectral coherence
"""

PREICTAL_LABEL = 1
INTERICTAL_LABEL = 0

def main(): 
    directory = os.path.join(MODEL_PATH, "HMMs")
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in range(0, 50):
        DS_0 = getDataset(1, i, 0)
        DS_1 = getDataset(1, i, 1)
        DS = [DS_0, DS_1]
        for j in range(6):
            for k in range(2):
                dt = a(DS[k][j])
                for n_clusters in [3, 5, 12]:
                    hmm = AdaptiveHMM(n_clusters, "ergodic", standardize = False)
                    hmm.fit(dt, dynamic_features = False)
                    hmm.pySave(os.path.join(directory, "hmm_%i_%i_%i_%i" % (i, j, k, n_clusters)))
    
        
if __name__ == "__main__":
    main()
    print("Finished")
    
    