# -*- coding: utf-8 -*-
# spectral.py
# author : Antoine Passemiers

from utils import *

import numpy as np
from scipy.signal import coherence, csd


COHERENCE_BINS = np.array([0.1, 4.0, 8.0, 12.0, 30.0, 70.0, 180.0])
DELTA_BINS_400_HZ = slice(1, 3)
THETA_BINS_400_HZ = slice(3, 6)
ALPHA_BINS_400_HZ = slice(6, 8)
BETA_BINS_400_HZ = slice(8, 20)
GAMMA_BINS_400_HZ = slice(20, 45)
HIGH_GAMMA_BINS_400_HZ = slice(45, 116)
ALL_BINS_400_HZ = slice(0, 120)
ALL_BANDS_400_HZ = [
    DELTA_BINS_400_HZ,
    THETA_BINS_400_HZ,
    ALPHA_BINS_400_HZ,
    BETA_BINS_400_HZ,
    GAMMA_BINS_400_HZ,
    HIGH_GAMMA_BINS_400_HZ]

all_bands = {"delta": DELTA_BINS_400_HZ, "theta": THETA_BINS_400_HZ,
             "alpha": ALPHA_BINS_400_HZ, "beta": BETA_BINS_400_HZ,
             "gamma": GAMMA_BINS_400_HZ, "high gamma": HIGH_GAMMA_BINS_400_HZ,
             "all": ALL_BINS_400_HZ}


@np.vectorize
def ezerolog(value):
    return np.log(value) if value > 0 else 0


def ZCR(signal):
    return len(np.where(np.diff(np.signbit(signal)))[0])


def STE(signal):
    return np.sum(signal ** 2, axis=0)


def PowerSpectrum(signal):
    fft = np.fft.fft(signal)
    return fft.real ** 2 / len(fft)


def ProbSpectralDensity(signal):
    spectrum = PowerSpectrum(signal)
    return spectrum / np.sum(spectrum)


def PowerSpectralEntropy(signal):
    density = ProbSpectralDensity(signal)
    return - np.sum(density * ezerolog(density))


def allCoherenceBins(N, fs):
    return coherence(np.random.rand(N), np.random.rand(N), fs=fs)[0]


def SpectralCoherence(signals, i, j, fs):
    return coherence(signals[:, i], signals[:, j], fs=fs)[1]


def AutospectralDensities(signals, fs):
    n_signals = signals.shape[1]
    autospectralDensities = []
    for i in range(n_signals):
        autospectralDensities.append(
            csd(signals[:, i], signals[:, i], fs=fs)[1])
    return autospectralDensities


def CustomSpectralCoherence(signals, autospectralDensities, i, j, fs):
    G_xy = np.abs(csd(signals[:, i], signals[:, j], fs=fs)[1])
    G_xx = autospectralDensities[i]
    G_yy = autospectralDensities[j]
    A = np.dot(G_xy, G_xy)
    B = np.vdot(G_xx, G_yy).real
    if A == 0:
        return 0
    elif B == 0:
        return 0
    return np.dot(G_xy, G_xy) / np.vdot(G_xx, G_yy).real


def LogSpectrum(signal, n_coef):
    N = len(signal)
    fft_freq = np.fft.fftfreq(N)
    spectrum = np.log(np.abs(np.fft.fft(signal).real))
    features = np.empty(n_coef, dtype=np.float32)
    max_freq = COHERENCE_BINS[-1]
    i = 0
    while fft_freq[i] < max_freq and i < N / 2:
        i += 1
    fft_freq = fft_freq[1:i]
    step = float(len(fft_freq)) / float(n_coef)
    begin = 0.0
    for i in range(n_coef):
        end = begin + step
        features[i] = spectrum[np.round(begin):np.round(end)].mean()
        begin = end
    return features
