# -*- coding: utf-8 -*-
# blockchain.py
# author : Antoine Passemiers

import numpy as np
from scipy import stats


class GrubbsTest:

    def __init__(self, N, alpha=0.01):
        self.N = N
        self.alpha = alpha
        self.crit_value = stats.t.isf(alpha / float(2 * N), N - 2)

    def test(self, signal, replace=True):
        N = len(signal)
        Y_argmin = signal.argmin()
        Y_argmax = signal.argmax()
        Y_argopt = Y_argmax if signal[Y_argmax] > - \
            signal[Y_argmin] else Y_argmin
        Y_mean = signal.mean()
        s = signal.std()
        if s > 0:
            G = np.abs(signal[Y_argopt] - Y_mean) / s
            G_crit = np.sqrt(self.crit_value ** 2 / (N - 2 + \
                             self.crit_value ** 2)) * float(N - 1) / np.sqrt(N)
            if G > G_crit:
                if replace:
                    signal[Y_argopt] = np.random.normal(loc=Y_mean, scale=s)
                return True
            else:
                return False
        else:
            return False

    def removeNans(self, signal, tolerance=0.40):
        noks = list()
        step = 280
        N = int(len(signal) / step)
        means = np.empty(N, dtype=np.float32)
        stds = np.empty(N, dtype=np.float32)
        for i in range(N):
            subset = signal[i * step:(i + 1) * step]
            nok = np.where(subset == 0)[0]
            ok = np.nonzero(subset)
            noks.append(nok)
            if ((float(len(ok[0])) / step) > tolerance):
                means[i] = np.mean(subset[ok])
                stds[i] = np.std(subset[ok])
            else:
                means[i], stds[i] = np.nan, np.nan

        xp = np.where(~np.isnan(means))[0]
        if len(xp) == 0:
            return signal
        x = np.arange(N)
        means = np.interp(x, xp, means[xp])
        stds = np.interp(x, xp, stds[xp])
        for i in range(N):
            subset = signal[i * step:(i + 1) * step]
            nok = noks[i]
            subset[nok] = np.asarray(
                np.random.normal(
                    means[i],
                    stds[i],
                    len(nok)),
                dtype=np.float32)
        return signal