"""
-*- basic config file -*-
@author:    alexjaniak
@date:      6/3/20
@file:      config.py  
"""

data_cfg = {
    "CATEGORICAL_VARIABLES": 37,
    "TRAIN_SPLIT": 0.8,
    "FORMATED_DATA": 
        {"images": (-1,-1,3),
        "file_names": "file_name_123.jpg",
        "formated_fnames": "file name",
        "labels": range(0,36)}
}

model_cfg = {
    "EPOCHS": 20,
    "ADAM_OPTIMIZER": {
        "lr": 0.001,
        "beta_1": 0.9,
        "beta_2": 0.999
    },
    "VALIDATION_SPLIT": 0.1
}