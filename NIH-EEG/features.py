import numpy as np
import nolds

from Spectral import *
from utils import checkDropOutsByChannel

class Feature:
    def __init__(self):
        self.n_electrodes = 16
        self.fs = 400
    def __len__(self):
        return self.n_electrodes
    
class FeatureLyapunovExponent(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
    def process(self, signals):
        features = np.empty(self.__len__(), dtype = np.float64)
        for i in range(self.__len__()):
            has_nan = checkDropOutsByChannel(signals[:, i])
            features[i] = nolds.lyap_r(signals[:, i]) if not has_nan else np.nan
        return features
    
class FeatureHurstExponent(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
    def process(self, signals):
        features = np.empty(self.__len__(), dtype = np.float64)
        for i in range(self.__len__()):
            has_nan = checkDropOutsByChannel(signals[:, i])
            features[i] = nolds.hurst_rs(signals[:, i]) if not has_nan else np.nan
        return features

class FeatureSTE(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
    def process(self, signals):
        features = np.empty(self.__len__(), dtype = np.float64)
        for i in range(self.__len__()):
            has_nan = checkDropOutsByChannel(signals[:, i])
            features[i] = STE(signals[:, i]) if not has_nan else np.nan
        return features
    
class FeatureZeroCrossings(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
    def process(self, signals):
        features = np.empty(self.__len__(), dtype = np.float64)
        for i in range(self.__len__()):
            has_nan = checkDropOutsByChannel(signals[:, i])
            features[i] = ZCR(signals[:, i]) if not has_nan else np.nan
        return features
    
class FeatureSpectralEntropy(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
    def process(self, signals):
        features = np.empty(self.__len__(), dtype = np.float64)
        for i in range(self.__len__()):
            has_nan = checkDropOutsByChannel(signals[:, i])
            features[i] = PowerSpectralEntropy(signals[:, i]) if not has_nan else np.nan
        return features
    
class FeatureSpectralCoherence(Feature):
    coherences = list()
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
        self.architecture = "circular"
        self.band = all_bands["all"]
        self.electrode_ids = list(range(self.__len__())) + [0]
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
                has_nan = checkDropOutsByChannel(signals[:, i])
                features[i] = AlphaCoherence(signals, autosp, self.electrode_ids[i], 
                    self.electrode_ids[i + 1], self.fs) if not has_nan else np.nan
            """
            for i in range(self.__len__()):
                features[i] = memory[i][self.band].mean()
            """
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
    def set_n_electrodes(self, n_electrodes):
        self.n_electrodes = n_electrodes
        self.n = 0
        for feature in self.features:
            feature.fs = self.fs
            feature.n_electrodes = self.n_electrodes
            self.n += len(feature)
    def getFeatures(self):
        return self.features
    def __len__(self):
        return self.n
    def __getitem__(self, key):
        return self.features[key]