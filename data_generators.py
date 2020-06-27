#!/usr/bin/env python
# coding: utf-8

import threading

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence


class aux_gen(Sequence):
    """
    Auxiliary generator to augment ImageDataGenerator capabilities in order to provide multiple outputs. It is thread
    safe.
    :param generator: ImageDataGenerator with a tuple with more than one value in the target column
    """

    def __init__(self, generator, n_outputs):
        self.generator = generator
        self.lock = threading.Lock()
        self.n_outputs = n_outputs
        self.n = generator.n
        self.batch_size = generator.batch_size

    def __len__(self):
        return int(np.ceil(len(self.generator.filepaths) / float(self.generator.batch_size)))

    def __getitem__(self, idx):
        with self.lock:
            gen_batch = self.generator.next()
            outputs = np.vstack(gen_batch[1])
            if self.n_outputs == 3:
                return np.array(gen_batch[0]), [outputs[:, 0], outputs[:, 1], outputs[:, 2]]
            elif self.n_outputs == 2:
                return np.array(gen_batch[0]), [outputs[:, 0], outputs[:, 1]]
            else:
                raise NotImplementedError


def data_generator(mode, dataframe, input_column, target_column, batch_size, input_shape, n_outputs,
                   data_augmentation_params=None):
    if data_augmentation_params is None:
        data_augmentation_params = {}

    assert mode in ['train', 'test', 'valid', 'complete'], 'Invalid mode'

    if mode == 'train':
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                  **data_augmentation_params)
        shuffle = True
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        shuffle = False

    generator = datagen.flow_from_dataframe(dataframe,
                                            x_col=input_column,
                                            y_col=target_column,
                                            batch_size=batch_size,
                                            class_mode='raw',
                                            color_mode='grayscale',
                                            target_size=input_shape,
                                            shuffle=shuffle
                                            )
    if n_outputs > 1:
        # If there is more than one output, use an auxiliary generator to add this functionality to flow_from_dataframe
        generator = aux_gen(generator, n_outputs=n_outputs)
    return generator
