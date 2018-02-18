# -*- coding: utf-8 -*-
# configs.py
# author : Antoine Passemiers

import numpy as np

# https://github.com/AntoinePassemiers/ArchMM
from archmm.core import IOConfig


io_configs = list()

""" Patient 1 """

# 0.497874
config = IOConfig()
config.n_classifiers = 1
config.preictal_proportion = 0.5
config.n_examples = 256
config.n_states = 7
config.architecture = "linear"
config.n_iterations = 50
config.pi_learning_rate = 0.005
config.pi_nhidden = 200
config.pi_nepochs = 2
config.pi_activation = "sigmoid"
config.s_learning_rate = 0.005
config.s_nhidden = 200
config.s_nepochs = 2
config.s_activation = "sigmoid"
config.o_learning_rate = 0.005
config.o_nhidden = 200
config.o_nepochs = 2
config.o_activation = "sigmoid"
config.missing_value_sym = np.nan
io_configs.append(config)



""" Patient 2 """

config = IOConfig()
config.n_classifiers = 1
config.preictal_proportion = 0.5
config.n_examples = 128
config.n_states = 5
config.architecture = "ergodic"
config.n_iterations = 100
config.pi_learning_rate = 0.005
config.pi_nhidden = 200
config.pi_nepochs = 2
config.pi_activation = "sigmoid"
config.s_learning_rate = 0.005
config.s_nhidden = 200
config.s_nepochs = 2
config.s_activation = "sigmoid"
config.o_learning_rate = 0.005
config.o_nhidden = 200
config.o_nepochs = 2
config.o_activation = "sigmoid"
config.missing_value_sym = np.nan
io_configs.append(config)


""" Patient 3 """

config = IOConfig()
config.n_classifiers = 1
config.preictal_proportion = 0.5
config.n_examples = 128
config.n_states = 5
config.architecture = "linear"
config.n_iterations = 50
config.pi_learning_rate = 0.005
config.pi_nhidden = 200
config.pi_nepochs = 2
config.pi_activation = "sigmoid"
config.s_learning_rate = 0.005
config.s_nhidden = 200
config.s_nepochs = 2
config.s_activation = "sigmoid"
config.o_learning_rate = 0.005
config.o_nhidden = 200
config.o_nepochs = 2
config.o_activation = "sigmoid"
config.missing_value_sym = np.nan
io_configs.append(config)