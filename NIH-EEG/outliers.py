# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats


class GrubbsTest:
    def __init__(self, N, alpha = 0.05):
        self.N = N
        self.alpha = alpha
        self.crit_value = stats.t.isf(alpha / float(2 * N), N - 2)
    def test(self, signal, replace = True):
        N = len(signal)
        Y_argmin = signal.argmin()
        Y_argmax = signal.argmax()
        Y_argopt = Y_argmax if signal[Y_argmax] > - signal[Y_argmin] else Y_argmin
        Y_mean = signal.mean()
        s = signal.std()
        G = np.abs(signal[Y_argopt] - Y_mean) / s
        if G > np.sqrt(self.crit_value / (N - 2 + self.crit_value)) * float(N - 1) / np.sqrt(N):
            if replace:
                signal[Y_argopt] = np.random.normal(loc = Y_mean, scale = s)
            return True
        else:
            return False