#!/usr/bin/env python
# coding: utf-8

EXPERIMENT_PREFIX = 'SOFT_F1'
RESOLUTION = 23.5  # pixels/mm
IMT_THRESHOLD = 0.8  # For IMT estimation from segmented results
DATABASE = 'CCA'  # 'BULB' or 'CCA'
RANDOM_SEED = 21  # To ensure reproducibility of results
COMPARE_RESULTS = True  # Compare with https://doi.org/10.1016/j.artmed.2019.101784
DEBUG = False  # Prints and plots extra information for debugging purposes
SILENT_MODE = False  # Suppress all outputs
MIXED_PRECISION = True  # models run faster and use less memory, needs compute capability > 7.0
FORCE_GPU = True
SAVE_FIGURES = True

import tensorflow as tf
def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost

# NN IMT prediction parameters
TARGET_COLUMNS = {'imt_max': {'predict': False, 'weight': 1., 'loss': 'mean_squared_error'},
                  'imt_avg': {'predict': False, 'weight': 1., 'loss': 'mean_squared_error'},
                  'plaque': {'predict': True, 'weight': 1., 'loss': macro_double_soft_f1}} #  binary_crossentropy

TRAIN = False
RESUME_TRAINING = False
INPUT_TYPE = 'img_and_mask'  # img or mask or img_and_mask
TRAIN_PERCENTAGE = 0.7
VAL_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.1
INPUT_SHAPE = (470, 445)  # original is (470, 445), masks depend on prediction size from previous step
LEARNING_RATE = 0.001  # CCA 0.001
BATCH_SIZE = 32
EPOCHS = 2000
EARLY_STOPPING_PATIENCE = 100
MAX_QUEUE_SIZE = 10
WORKERS = 10
DROPOUT_RATE = .15  # CCA .25 BULB .15
DATA_AUGMENTATION_PARAMS = {'width_shift_range': 0.05,
                            'height_shift_range': 0.05,
                            #'vertical_flip': True,
                            'horizontal_flip': True,
                            'rotation_range': 1,
                            # 'zoom_range': 0.1,
                            'fill_mode': 'constant',
                            'cval': 0}
