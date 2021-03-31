# !/usr/bin/env python
# coding: utf-8

EXPERIMENT_SUFFIX = ''
RESOLUTION = 23.5  # pixels/mm
IMT_THRESHOLD = 0.8  # For IMT estimation from segmented results # TODO: Remove
DATABASE = 'BULB'  # 'BULB' or 'CCA'
if DATABASE == 'CCA':
    RANDOM_SEED = 29  # To ensure reproducibility of results
elif DATABASE == 'BULB':
    RANDOM_SEED = 1  # To ensure reproducibility of results
else:
    raise NotImplementedError()

DEBUG = False  # Prints and plots extra information for debugging purposes
SILENT_MODE = False  # Suppress all outputs
MIXED_PRECISION = True  # models run faster and use less memory, needs compute capability > 7.0
FORCE_GPU = True
SAVE_FIGURES = True

# NN IMT prediction parameters
TARGET_COLUMNS = {'imt_max': {'predict': True, 'weight': 1., 'loss': 'mean_squared_error'},
                  'imt_avg': {'predict': True, 'weight': 1., 'loss': 'mean_squared_error'},
                  'plaque': {'predict': True, 'weight': 0.5, 'loss': 'binary_crossentropy'}}

TRAIN = False
RESUME_TRAINING = False
INPUT_TYPE = 'img_and_mask'  # img or mask or img_and_mask
TRAIN_PERCENTAGE = 0.6
VAL_PERCENTAGE = 0.20
TEST_PERCENTAGE = 0.20
INPUT_SHAPE = (470, 445)  # original is (470, 445), masks depend on prediction size from previous step
LEARNING_RATE = 0.001  # CCA 0.001
BATCH_SIZE = 32
EPOCHS = 500
RLR_ON_PLATEAU_PATIENCE = 20
EARLY_STOPPING_PATIENCE = 40
MAX_QUEUE_SIZE = 10
WORKERS = 10
DROPOUT_RATE = .25
DATA_AUGMENTATION_PARAMS = {'width_shift_range': 0.05,
                            'height_shift_range': 0.05,
                            'vertical_flip': False,
                            'horizontal_flip': True,
                            'rotation_range': 1,
                            'brightness_range': [0.8, 1.0],
                            # 'zoom_range': 0.1,
                            'fill_mode': 'constant',
                            'cval': 0}
