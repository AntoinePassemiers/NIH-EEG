# -*- coding: utf-8 -*-
# features.py
# author : Antoine Passemiers

from spectral import *
from utils import checkDropOutsByChannel

import numpy as np
import nolds # non-linear measures for dynamical systems


class Feature:
    def __init__(self):
        self.n_electrodes = 16
        self.fs = 400

    def __len__(self):
        return self.n_electrodes


class FeatureLogSpectrum(Feature):
    coherences = list()

    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
        self.n_coef = 144

    def __len__(self):
        return self.n_electrodes * self.n_coef

    def process(self, signals):
        features = np.empty(self.__len__(), dtype=float)
        for i in range(self.n_electrodes):
            has_nan = checkDropOutsByChannel(signals[:, i])
            if not has_nan:
                features[(i * self.n_coef):((i + 1) * self.n_coef)
                         ] = LogSpectrum(signals[:, i], self.n_coef)
            else:
                features[(i * self.n_coef):((i + 1) * self.n_coef)] = np.nan
        return features


class FeatureLyapunovExponent(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)

    def process(self, signals):
        features = np.empty(self.__len__(), dtype=np.float64)
        for i in range(self.__len__()):
            has_nan = checkDropOutsByChannel(signals[:, i])
            features[i] = nolds.lyap_r(
                signals[:, i]) if not has_nan else np.nan
        return features


class FeatureHurstExponent(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)

    def process(self, signals):
        features = np.empty(self.__len__(), dtype=np.float64)
        for i in range(self.__len__()):
            has_nan = checkDropOutsByChannel(signals[:, i])
            features[i] = nolds.hurst_rs(
                signals[:, i]) if not has_nan else np.nan
        return features


class FeatureSTE(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)

    def process(self, signals):
        features = np.empty(self.__len__(), dtype=np.float64)
        for i in range(self.__len__()):
            has_nan = checkDropOutsByChannel(signals[:, i])
            features[i] = STE(signals[:, i]) if not has_nan else np.nan
        return features


class FeatureZeroCrossings(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)

    def process(self, signals):
        features = np.empty(self.__len__(), dtype=np.float64)
        for i in range(self.__len__()):
            has_nan = checkDropOutsByChannel(signals[:, i])
            features[i] = ZCR(signals[:, i]) if not has_nan else np.nan
        return features


class FeatureSpectralEntropy(Feature):
    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)

    def process(self, signals):
        features = np.empty(self.__len__(), dtype=np.float64)
        for i in range(self.__len__()):
            has_nan = checkDropOutsByChannel(signals[:, i])
            features[i] = PowerSpectralEntropy(
                signals[:, i]) if not has_nan else np.nan
        return features


class FeatureSpectralCoherence(Feature):

    coherences = list()

    def __init__(self, *args, **kwargs):
        Feature.__init__(self, *args, **kwargs)
        self.architecture = "circular"
        self.band = all_bands["all"]
        self.electrode_ids = list(range(16)) + [0]
        FeatureSpectralCoherence.coherences.append(self)

    def __len__(self):
        n_bands = len(ALL_BANDS_400_HZ)
        if self.architecture == "circular":
            return self.n_electrodes * n_bands
        else:
            return self.n_electrodes * (self.n_electrodes - 1) * n_bands / 2

    def config(self, architecture="circular", band="all"):
        self.architecture = architecture
        self.band = all_bands[band]
        return self

    def process(self, signals):
        features = np.empty(self.__len__(), dtype=float)
        n_bands = len(ALL_BANDS_400_HZ)
        if self.architecture == "circular":
            # autosp = AutospectralDensities(signals, self.fs)
            for i in range(self.n_electrodes):
                has_nan = checkDropOutsByChannel(signals[:, i])
                coherences = SpectralCoherence(
                    signals, self.electrode_ids[i], self.electrode_ids[i + 1], fs=self.fs)
                for j in range(n_bands):
                    features[i * n_bands + j] = coherences[ALL_BANDS_400_HZ[j]
                        ].mean() if not has_nan else np.nan
        elif self.architecture == "full":
            has_nan = np.zeros(self.n_electrodes, dtype=np.bool)
            for i in range(self.n_electrodes):
                has_nan[i] = checkDropOutsByChannel(signals[:, i])
            p = 0
            for i in range(self.n_electrodes):
                for k in range(self.n_electrodes):
                    if i < k:
                        coherences = SpectralCoherence(
                            signals, i, k, fs=self.fs)
                        for j in range(n_bands):
                            features[p * n_bands + j] = coherences[ALL_BANDS_400_HZ[j]
                                ].mean() if not (has_nan[i] or has_nan[k]) else np.nan
                        p += 1
        else:
            NotImplementedError()
        return features


class FeatureSet:

    def __init__(self, n_electrodes, fs=400):
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
