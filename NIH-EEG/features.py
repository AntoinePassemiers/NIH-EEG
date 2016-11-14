import numpy as np
from Spectral import *


class Feature:
    def __init__(self, n_electrodes, fs = 400):
        self.n_electrodes = n_electrodes
        self.fs = fs
    def __len__(self):
        return self.n_electrodes

class FeatureSTE(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
    def process(self, signals):
        features = np.empty(self.__len__(), dtype = np.float64)
        for i in range(self.__len__()):
            features[i] = STE(signals[:, i])
        return features
    
class FeatureZeroCrossings(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
    def process(self, signals):
        features = np.empty(self.__len__(), dtype = np.float64)
        for i in range(self.__len__()):
            features[i] = ZCR(signals[:, i])
        return features
    
class FeatureSpectralEntropy(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
    def process(self, signals):
        features = np.empty(self.__len__(), dtype = np.float64)
        for i in range(self.__len__()):
            features[i] = PowerSpectralEntropy(signals[:, i])
        return features
    
class FeatureSpectralCoherence(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
        self.electrode_ids = list(range(self.__len__())) + [0]
    def process(self, signals):
        features = np.empty(self.__len__(), dtype = float)
        for i in range(self.__len__()):
            features[i] = SpectralCoherence(signals, self.electrode_ids[i], 
                self.electrode_ids[i + 1], self.fs)
        return features
    
class FeatureSet:
    def __init__(self):
        self.features = list()
        self.n = 0
    def add(self, feature):
        self.features.append(feature)
        self.n += len(feature)
    def getFeatures(self):
        return self.features
    def __len__(self):
        return self.n
    def __getitem__(self, key):
        return self.features[key]