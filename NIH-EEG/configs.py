# -*- coding: utf-8 -*-

import numpy as np

from HMM_Core import IOConfig

io_configs = list()

""" Patient 1 """

config = IOConfig()
config.n_examples = 128
config.n_iterations = 100
config.pi_learning_rate = 0.005
config.pi_nhidden = 200
config.pi_nepochs = 2
config.pi_activation = "sigmoid"
config.s_learning_rate  = 0.005
config.s_nhidden  = 200
config.s_nepochs = 2
config.s_activation = "sigmoid"
config.o_learning_rate  = 0.005
config.o_nhidden  = 200
config.o_nepochs = 2
config.o_activation = "sigmoid"
config.missing_value_sym = np.nan
io_configs.append(config)

""" Patient 2 """

config = IOConfig()
config.n_examples = 128
config.n_iterations = 100
config.pi_learning_rate = 0.005
config.pi_nhidden = 200
config.pi_nepochs = 2
config.pi_activation = "sigmoid"
config.s_learning_rate  = 0.005
config.s_nhidden  = 200
config.s_nepochs = 2
config.s_activation = "sigmoid"
config.o_learning_rate  = 0.005
config.o_nhidden  = 200
config.o_nepochs = 2
config.o_activation = "sigmoid"
config.missing_value_sym = np.nan
io_configs.append(config)

""" Patient 3 """

config = IOConfig()
config.n_examples = 128
config.n_iterations = 100
config.pi_learning_rate = 0.005
config.pi_nhidden = 100
config.pi_nepochs = 2
config.pi_activation = "sigmoid"
config.s_learning_rate  = 0.005
config.s_nhidden  = 100
config.s_nepochs = 2
config.s_activation = "sigmoid"
config.o_learning_rate  = 0.005
config.o_nhidden  = 100
config.o_nepochs = 2
config.o_activation = "sigmoid"
config.missing_value_sym = np.nan
io_configs.append(config)