# -*- coding: utf-8 -*-

import json, os, pickle, random
import warnings

warnings.filterwarnings('ignore')

with open('SETTINGS.json') as settings_file:
    SETTINGS = json.load(settings_file)
    
DATA_PATH = str(SETTINGS["DATA_PATH"])
MODEL_PATH = str(SETTINGS["MODEL_PATH"])