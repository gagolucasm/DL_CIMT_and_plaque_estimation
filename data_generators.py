#!/usr/bin/env python
# coding: utf-8

import threading

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence


class aux_gen(Sequence):
    """
    Auxiliary generator to augment ImageDataGenerator capabilities in order to provide multiple inputs and outputs.
    It is thread safe.
    :param generator: ImageDataGenerator with a tuple with more than one value in the target column
    """

    def __init__(self, generators, n_outputs):
        self.n_inputs = len(generators)
        self.generators = generators
        self.lock = threading.Lock()
        self.n_outputs = n_outputs
        self.n = generators[0].n  # All generators should provide the same result
        self.batch_size = generators[0].batch_size
        self.samples = len(self.generators[0].filepaths)

    def __len__(self):
        return int(np.ceil(self.samples / float(self.batch_size)))

    def __getitem__(self, idx):
        with self.lock:
            gen_batch = []
            for i in range(self.n_inputs):
                gen_batch.append(self.generators[i].next())
            outputs = np.vstack(gen_batch[0][1])
            if self.n_outputs == 3:
                output_list = [outputs[:, 0], outputs[:, 1], outputs[:, 2]]
            elif self.n_outputs == 2:
                output_list = [outputs[:, 0], outputs[:, 1]]
            elif self.n_outputs == 1:
                output_list = outputs[:, 0]
            else:
                raise NotImplementedError

            if self.n_inputs == 1:
                input_list = np.array(gen_batch[0][0])
            elif self.n_inputs == 2:
                input_list = np.concatenate((np.array(gen_batch[0][0]), np.array(gen_batch[1][0])), axis=3)
            else:
                raise NotImplementedError
            return input_list, output_list


def data_generator(mode, dataframe, input_column, target_column, batch_size, input_shape, n_outputs, seed,
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
    generators = []
    if input_column == 'img_and_mask':
        generators.append(datagen.flow_from_dataframe(dataframe,
                                                      x_col='complete_path',
                                                      y_col=target_column,
                                                      batch_size=batch_size,
                                                      class_mode='raw',
                                                      color_mode='grayscale',
                                                      target_size=input_shape,
                                                      shuffle=False,
                                                      seed=seed
                                                      ))
        generators.append(datagen.flow_from_dataframe(dataframe,
                                                      x_col='mask_path',
                                                      y_col=target_column,
                                                      batch_size=batch_size,
                                                      class_mode='raw',
                                                      color_mode='grayscale',
                                                      target_size=input_shape,
                                                      shuffle=False,
                                                      seed=seed
                                                      ))
    else:
        generators.append(datagen.flow_from_dataframe(dataframe,
                                                      x_col=input_column,
                                                      y_col=target_column,
                                                      batch_size=batch_size,
                                                      class_mode='raw',
                                                      color_mode='grayscale',
                                                      target_size=input_shape,
                                                      shuffle=shuffle
                                                      ))

    generator = aux_gen(generators, n_outputs=n_outputs)

    return generator
