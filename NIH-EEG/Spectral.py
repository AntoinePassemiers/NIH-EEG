# -*- coding: utf-8 -*-

from utils import *
import numpy as np
from scipy.signal import csd


def STE(signal):
    return np.sum(signal ** 2, axis = 0)

def AutospectralDensities(signals, fs):
    n_signals = signals.shape[1]
    autospectralDensities = []
    for i in range(n_signals):
        autospectralDensities.append(csd(signals[:, i], signals[:, i], fs = fs)[1])
    return autospectralDensities

def AlphaCoherence(signals, autospectralDensities, i, j, fs):
    G_xy = np.abs(csd(signals[:, i], signals[:, j], fs = fs)[1])
    G_xx = autospectralDensities[i]
    G_yy = autospectralDensities[j]
    A = np.dot(G_xy, G_xy)
    B = np.vdot(G_xx, G_yy).real
    if A == 0:
        return 0
    elif B == 0:
        return 0
    return np.dot(G_xy, G_xy) / np.vdot(G_xx, G_yy).real