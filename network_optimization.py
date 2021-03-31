#!/usr/bin/env python
# coding: utf-8

import json
import os

import hiplot as hip
import numpy as np
import pandas as pd
import tensorflow as tf
from kerastuner import HyperModel
from kerastuner.tuners import BayesianOptimization
from tensorflow.python.keras.metrics import Recall

import config
import helpers
from data_generators import data_generator


class CNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=hp.Choice('num_filters_1', values=[16, 32, 64], default=32),
                                         kernel_size=(3, 3), input_shape=(self.input_shape[0], self.input_shape[1], 1),
                                         activation='relu'))
        model.add(tf.keras.layers.Conv2D(filters=hp.Choice('num_filters_2', values=[16, 32, 64], default=64),
                                         kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(
            tf.keras.layers.Conv2D(hp.Choice('num_filters_3', values=[16, 32, 64, 128], default=64), kernel_size=(3, 3),
                                   activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(filters=hp.Choice(
            'num_filters_4',
            values=[16, 32, 64, 128],
            default=32,
        ), kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            units=hp.Int(
                'units_1',
                min_value=16,
                max_value=128,
                step=16,
                default=32
            ),
            activation='relu'
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=hp.Float(
            'dropout_1',
            min_value=0.0,
            max_value=0.5,
            default=0.25,
            step=0.05
        )))
        model.add(tf.keras.layers.Dense(units=hp.Int(
            'units_2',
            min_value=16,
            max_value=128,
            step=16,
            default=32
        ),
            activation='relu'
        ))
        model.add(tf.keras.layers.Dropout(rate=hp.Float(
            'dropout_2',
            min_value=0.0,
            max_value=0.5,
            default=0.25,
            step=0.05
        )))
        model.add(tf.keras.layers.Dense(units=hp.Int(
            'units_3',
            min_value=8,
            max_value=64,
            step=8,
            default=32
        ),
            activation='relu'
        ))
        model.add(tf.keras.layers.Dense(1, activation=hp.Choice(
            'dense_activation',
            values=['relu', 'tanh', 'sigmoid'],
            default='relu'
        )))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss=hp.Choice('loss_function', values=['mean_squared_error', 'mae', 'mean_squared_logarithmic_error']),
            metrics=['mae', 'mean_squared_error']
        )
        return model


if __name__ == '__main__':
    # Load data from disk
    data = np.load(os.path.join('segmentation', 'complete_data_{}.npy'.format(config.DATABASE)))
    data = data[()]['data']

    # Convert to dataframe and filter invalid values
    df = pd.DataFrame.from_dict(data, orient='index')
    df = helpers.filter_dataframe(df, config.DATABASE)

    # Change index format #TODO: fix in previous step
    df.index = df.index.map(lambda x: x[4:-1])

    df['gt_plaque'] = df['gt_imt_max'].apply(lambda x: 1 if x >= 1.5 else 0)

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    if config.INPUT_TYPE == 'img':
        input_column = 'complete_path'
    elif config.INPUT_TYPE == 'mask':
        input_column = 'mask_path'

    # Shuffle dataframe
    df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)

    # TODO: Adapt for single value training
    selected_columns = ['gt_' + key for key, value in config.TARGET_COLUMNS.items() if value['predict']]
    n_outputs = len(selected_columns)
    if n_outputs == 1:
        target_column = selected_columns[0]
    else:

        raise NotImplementedError("Network optimization only implemented for single output")

    df_train, df_valid, df_test, df = helpers.train_validate_test_split(df, train_percent=config.TRAIN_PERCENTAGE,
                                                                        validate_percent=config.VAL_PERCENTAGE,
                                                                        test_percent=config.VAL_PERCENTAGE)

    # Define losses and weights depending on number of outputs
    losses = []
    metrics = {}
    loss_weights = []

    for key, value in config.TARGET_COLUMNS.items():
        if value['predict']:
            if key == 'plaque':
                metrics['plaque'] = [Recall(name='recall'), 'accuracy']
            losses.append(value['loss'])
            loss_weights.append(value['weight'])

    # Define data generators
    train_generator = data_generator(mode='train', dataframe=df_train, input_column=input_column,
                                     target_column=target_column, batch_size=config.BATCH_SIZE,
                                     data_augmentation_params=config.DATA_AUGMENTATION_PARAMS,
                                     input_shape=config.INPUT_SHAPE, seed=config.RANDOM_SEED
                                     , n_outputs=n_outputs)
    valid_generator = data_generator(mode='valid', dataframe=df_valid, input_column=input_column,
                                     target_column=target_column, batch_size=config.BATCH_SIZE,
                                     input_shape=config.INPUT_SHAPE, seed=config.RANDOM_SEED
                                     , n_outputs=n_outputs)
    test_generator = data_generator(mode='test', dataframe=df_test, input_column=input_column,
                                    target_column=target_column, batch_size=config.BATCH_SIZE,
                                    input_shape=config.INPUT_SHAPE, seed=config.RANDOM_SEED
                                    , n_outputs=n_outputs)
    TRAINING_STEPS = train_generator.n // train_generator.batch_size
    VALIDATION_STEPS = valid_generator.n // valid_generator.batch_size

    if config.DEBUG:
        helpers.test_generator_output(test_generator, n_images=2)

    hypermodel = CNNHyperModel(input_shape=(config.INPUT_SHAPE[0], config.INPUT_SHAPE[1], 1))

    tuner = BayesianOptimization(
        hypermodel,
        objective='val_loss',
        seed=42,
        max_trials=60,
        directory='bayesian_optimization',
        project_name='{}_{}'.format(config.DATABASE, target_column)
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.EARLY_STOPPING_PATIENCE,
                                                restore_best_weights=True)
    tuner.search(train_generator,
                 steps_per_epoch=TRAINING_STEPS,
                 validation_data=valid_generator,
                 validation_steps=VALIDATION_STEPS,
                 epochs=50,
                 max_queue_size=10,
                 workers=6,
                 verbose=2,
                 callbacks=[callback])

    vis_data = []
    rootdir = 'bayesian_optimization/{}_{}'.format(config.DATABASE, target_column)
    for subdirs, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith("trial.json"):
                with open(subdirs + '/' + file, 'r') as json_file:
                    data = json_file.read()
                vis_data.append(json.loads(data))

    # TODO: Automatically generate data
    data = [{'num_filters_1': vis_data[idx]['hyperparameters']['values']['num_filters_1'],
             'num_filters_2': vis_data[idx]['hyperparameters']['values']['num_filters_2'],
             'num_filters_3': vis_data[idx]['hyperparameters']['values']['num_filters_3'],
             'units': vis_data[idx]['hyperparameters']['values']['units'],
             'dropout_1': vis_data[idx]['hyperparameters']['values']['dropout_1'],
             'dropout_2': vis_data[idx]['hyperparameters']['values']['dropout_2'],
             'learning_rate': vis_data[idx]['hyperparameters']['values']['learning_rate'],
             'loss_function': vis_data[idx]['hyperparameters']['values']['loss_function'],

             'mae': vis_data[idx]['metrics']['metrics']['mae']['observations'][0]['value'],
             'val_mae': vis_data[idx]['metrics']['metrics']['val_mae']['observations'][0]['value'],
             } for idx in range(len(vis_data))]

    hip.Experiment.from_iterable(data).display()
