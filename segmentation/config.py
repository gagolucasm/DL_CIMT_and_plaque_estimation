#!/usr/bin/env python
# coding: utf-8

DATABASE = 'CCA'  # 'BULB' or 'CCA'
PREDICT_ONLY_IM = False  # If True, only binary segmentation will be performed.
SINGLE_IMT_CLASS_BULB = True  # If True, IMT doubtful will be considered as IMT
BATCH_SIZE = 4
INPUT_SHAPE = (512, 512)

BACKBONE = 'efficientnetb0'  # Can be one of: 'mobilenetv2', 'efficientnetb{0 to 7}', 'inceptionv3', 'vgg16', 'resnet34'
LR = 0.0001
LOAD_PRETRAINED_MODEL = True  # If activated, the code will only evaluate

TRAINING_PARAMETERS = {
    'steps_per_epoch': 200,
    'epochs': 40,
    'validation_steps': 35,
    'use_multiprocessing': False,
    'max_queue_size': 10,
    'workers': 8,
    'verbose': 1}
