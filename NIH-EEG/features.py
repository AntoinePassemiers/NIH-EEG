import numpy as np
from Spectral import *
import nolds

class SharedData:
    def __init__(self):
        self.memory = None
        
class SharedCoherence(SharedData):
    memory = list()
    fs = 400
    n_electrodes = 0
    architecture = "circular"
    electrode_ids = list()
    @staticmethod
    def process(signals):
        SharedCoherence.memory = list()
        if SharedCoherence.architecture == "circular":
            for i in range(SharedCoherence.n_electrodes):
                SharedCoherence.memory.append(SpectralCoherence(signals, SharedCoherence.electrode_ids[i], 
                    SharedCoherence.electrode_ids[i + 1], SharedCoherence.fs))
        elif SharedCoherence.architecture == "full":
            for i in range(SharedCoherence.n_electrodes):
                for j in range(SharedCoherence.n_electrodes):
                    if i != j:
                        SharedCoherence.memory.append(SpectralCoherence(signals, 
                            SharedCoherence.electrode_ids[i], 
                            SharedCoherence.electrode_ids[j], SharedCoherence.fs))
        else:
            NotImplementedError()
        return SharedCoherence.memory
    @staticmethod
    def getMemory():
        return SharedCoherence.memory

class Feature:
    def __init__(self):
        self.n_electrodes = 16
        self.fs = 400
        self.shared = None
    def __len__(self):
        return self.n_electrodes
    
class FeatureLyapunovExponent(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
    def process(self, signals):
        features = np.empty(self.__len__(), dtype = np.float64)
        for i in range(self.__len__()):
            features[i] = nolds.lyap_r(signals[:, i])
        return features
    
class FeatureHurstExponent(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
    def process(self, signals):
        features = np.empty(self.__len__(), dtype = np.float64)
        for i in range(self.__len__()):
            features[i] = nolds.hurst_rs(signals[:, i])
        return features

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
        self.band = all_bands["all"]
        self.electrode_ids = list(range(self.__len__())) + [0]
        self.shared = SharedCoherence
        self.shared.architecture = self.architecture
        self.shared.electrode_ids = self.electrode_ids
        FeatureSpectralCoherence.coherences.append(self)
    def __len__(self):
        if self.architecture == "circular":
            return self.n_electrodes
        else:
            return self.n_electrodes * (self.n_electrodes - 1)
    def config(self, architecture = "circular", band = "all"):
        self.architecture = architecture
        self.band = all_bands[band]
        return self
    def process(self, signals):
        features = np.empty(self.__len__(), dtype = float)
        if self.architecture == "circular":
            autosp = AutospectralDensities(signals, self.fs)
            for i in range(self.__len__()):
                features[i] = AlphaCoherence(signals, autosp, self.electrode_ids[i], 
                    self.electrode_ids[i + 1], self.fs)
            """
            for i in range(self.__len__()):
                features[i] = memory[i][self.band].mean()
            """
        elif self.architecture == "full":
            memory = self.shared.getMemory()
            k = 0
            for i in range(self.n_electrodes):
                for j in range(self.n_electrodes):
                    if i != j:
                        features[k] = memory[k][self.band].mean()
                        k += 1
        else:
            NotImplementedError()
        return features
    
class FeatureSet:
    def __init__(self, n_electrodes, fs = 400):
        self.features = list()
        self.shared = list()
        self.n = 0
        self.n_electrodes = n_electrodes
        self.fs = fs
    def add(self, feature):
        feature.fs = self.fs
        feature.n_electrodes = self.n_electrodes
        self.features.append(feature)
        self.n += len(feature)
        if feature.shared and feature.shared not in self.shared:
            feature.shared.n_electrodes = len(feature)
            feature.shared.fs = self.fs
            self.shared.append(feature.shared)
    def getFeatures(self):
        return self.features
    def __len__(self):
        return self.n
    def __getitem__(self, key):
        return self.features[key]