# -*- coding: utf-8 -*-

from utils import *
import numpy as np
from scipy.signal import coherence

COHERENCE_BINS = np.array([0.1, 4.0, 8.0, 12.0, 30.0, 70.0, 180.0])
DELTA_BINS_400_HZ = slice(1, 3)
THETA_BINS_400_HZ = slice(3, 6)
ALPHA_BINS_400_HZ = slice(6, 8)
BETA_BINS_400_HZ  = slice(8, 20)
GAMMA_BINS_400_HZ = slice(20, 45)
HIGH_GAMMA_BINS_400_HZ = slice(45, 116)

@np.vectorize
def elog(value):
    return np.log(value) if value > 0 else -9999

def ZCR(signal):
    return len(np.where(np.diff(np.signbit(signal)))[0])

def STE(signal):
    return np.sum(signal ** 2, axis = 0)

def PowerSpectrum(signal):
    fft = np.fft.fft(signal)
    return fft ** 2 / len(fft)

def ProbSpectralDensity(signal):
    spectrum = PowerSpectrum(signal).real
    return spectrum / np.sum(spectrum)

def PowerSpectralEntropy(signal):
    density = ProbSpectralDensity(signal)
    return - np.sum(density * elog(density))

def allCoherenceBins(N, fs):
    return coherence(np.random.rand(N), np.random.rand(N), fs = fs)[0]

def SpectralCoherence(signals, i, j, fs):
    return coherence(signals[:, i], signals[:, j], fs = fs)[1].mean()