import numpy as np
from Spectral import *


class SharedData:
    pass        
class SharedCoherence(SharedData):
    pass

class Feature:
    def __init__(self):
        self.n_electrodes = 16
        self.fs = 400
        self.shared = None
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
    coherences = list()
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
        self.architecture = "circular"
        self.electrode_ids = list(range(self.__len__())) + [0]
        self.shared = SharedCoherence
        FeatureSpectralCoherence.coherences.append(self)
    def __len__(self):
        if self.architecture == "circular":
            return self.n_electrodes
        else:
            return self.n_electrodes * (self.n_electrodes - 1)
    def config(self, architecture = "circular"):
        self.architecture = architecture
        return self
    def process(self, signals):
        features = np.empty(self.__len__(), dtype = float)
        if self.architecture == "circular":
            for i in range(self.__len__()):
                features[i] = SpectralCoherence(signals, self.electrode_ids[i], 
                    self.electrode_ids[i + 1], self.fs)
        elif self.architecture == "full":
            k = 0
            for i in range(self.n_electrodes):
                for j in range(self.n_electrodes):
                    if i != j:
                        features[k] = SpectralCoherence(signals, self.electrode_ids[i], 
                            self.electrode_ids[j], self.fs)
                        k += 1
        else:
            NotImplementedError()
        return features
    
class FeatureSet:
    def __init__(self, n_electrodes, fs = 400):
        self.features = list()
        self.n = 0
        self.n_electrodes = n_electrodes
        self.fs = fs
    def add(self, feature):
        feature.fs = self.fs
        feature.n_electrodes = self.n_electrodes
        self.features.append(feature)
        self.n += len(feature)
    def getFeatures(self):
        return self.features
    def __len__(self):
        return self.n
    def __getitem__(self, key):
        return self.features[key]