# -*- coding: utf-8 -*-

from utils import *
import numpy as np
from scipy.signal import csd



def HaarDWT(signal):
    N = len(signal)
    output = np.zeros(N)
    length = N >> 1
    while True:
        for i in xrange(0, length):
            output[i] = signal[i * 2] + signal[i * 2 + 1]
            output[length + i] = signal[i * 2] - signal[i * 2 + 1]
        if length == 1:
            return output
        signal = output[:length << 1]
        length >>= 1

def AutospectralDensities(signals, fs):
    n_signals = signals.shape[1]
    autospectralDensities = []
    for i in range(n_signals):
        autospectralDensities.append(csd(signals[:, i], signals[:, i], fs = fs)[1])
    return autospectralDensities

def AlphaCoherence(signals,autospectralDensities, fs):
    n_signals = signals.shape[1]
    alpha = np.eye(n_signals, dtype = float)
    for i in range(n_signals):
        for j in range(n_signals):
            if i != j:
                G_xy = np.abs(csd(signals[:, i], signals[:, j], fs = fs)[1])
                G_xx = autospectralDensities[i]
                G_yy = autospectralDensities[j]
                alpha[i, j] = np.dot(G_xy, G_xy) / np.dot(G_xx, G_yy)
                print(i, j)
    return alpha