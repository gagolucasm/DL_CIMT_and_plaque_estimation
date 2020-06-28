#!/usr/bin/env python
# coding: utf-8

RESOLUTION = 23.5  # pixels/mm
IMT_THRESHOLD = 0.8  # For IMT estimation from segmented results
DATABASE = 'CCA'  # 'BULB' or 'CCA'
RANDOM_SEED = 21  # To ensure reproducibility of results
COMPARE_RESULTS = True  # Compare with https://doi.org/10.1016/j.artmed.2019.101784
DEBUG = False  # Prints and plots extra information for debugging purposes
SILENT_MODE = False  # Suppress all outputs # TODO: Implement
MIXED_PRECISION = True  # models run faster and use less memory, needs > compute capability 7.0

# NN IMT prediction parameters
TARGET_COLUMNS = {'imt_max': {'predict': True, 'weight': 1., 'loss': 'mean_squared_error'},
                  'imt_avg': {'predict': True, 'weight': 1., 'loss': 'mean_squared_error'},
                  'plaque': {'predict': True, 'weight': .5, 'loss': 'binary_crossentropy'}}
TRAIN = True
RESUME_TRAINING = False
INPUT_TYPE = 'img'  # img or mask
TRAIN_PERCENTAGE = 0.7
VAL_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.1
INPUT_SHAPE = (512, 512)  # original is (470, 445), masks depend on prediction size from previous step
LEARNING_RATE = 0.01  # CCA 0.001
BATCH_SIZE = 16
EPOCHS = 2000
EARLY_STOPPING_PATIENCE = 100
MAX_QUEUE_SIZE = 5
WORKERS = 6
DROPOUT_RATE = .25  # CCA .25 BULB .15
DATA_AUGMENTATION_PARAMS = {'width_shift_range': 0.05,
                            'height_shift_range': 0.05,
                            # 'vertical_flip': True,
                            'horizontal_flip': True,
                            'rotation_range': 1,
                            # 'zoom_range': 0.1,
                            'fill_mode': 'constant',
                            'cval': 0}
