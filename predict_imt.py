#!/usr/bin/env python
# coding: utf-8
import inspect
import os
import time

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import Recall
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.optimizers import Adam

import config
import helpers
from data_generators import data_generator
from helpers import add_previous_results, filter_dataframe
from models import get_imt_prediction_model


def weighted_bce(y_true, y_pred):
    """
    Weighted binary cross-entropy loss for training of unbalanced plaque class classification
    :param y_true: tensor of gt
    :param y_pred: tensor of predicted values
    :return: weighted binary cross-entropy between gt and predictions
    """
    weights = (y_true * 10.) + 1.
    bce = keras_backend.binary_crossentropy(y_true, y_pred)
    weighted_bce = keras_backend.mean(bce * weights)
    return weighted_bce


def nn_predict_imt(img_path, mask_path, model, input_shape, target_columns):
    """
    Predicts the IMT for an image given its path
    :param img_path: path to the image to predict. If the original image is not an input, replace with None
    :param mask_path: path to the mask to predict. If the mask is not an input, replace with None
    :param model: tensorflow model for IMT prediction
    :param target_columns: name of the columns forming the output
    :param input_shape: shape of the input image
    :return: predicted IMT values, specific targets depends on the model
    """
    if img_path is not None:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.
        img = cv2.resize(img, (input_shape[1], input_shape[0]))
        input_data = img
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.
        mask = cv2.resize(mask, (input_shape[1], input_shape[0]))
        input_data = mask
    if img_path is not None and mask_path is not None:
        input_data = np.dstack((img, mask))
    prediction = model.predict(np.expand_dims(input_data, axis=0))

    result = {}
    count = 0
    for key, value in target_columns.items():
        if value['predict']:
            result[key] = np.squeeze(prediction[count])
            count += 1
    return result


def plot_predictions(model, generator, plot_images=False, loops=1):
    """
    Predicts batches from a specified generator to analyze the output.
    :param model: tensorflow model
    :param generator: generator to evaluate
    :param plot_images: boolean indicating if results should be plotted. If batch size is high, it could be impractical
    :param loops: number of batches to predict on
    """
    errors = []
    for x_batch, y_batch in generator:
        for i in range(generator.batch_size):
            if config.PREDICT_PLAQUE:  # TODO:Update
                gt = [y_batch[0][i], y_batch[1][i], y_batch[2][i]]
            else:
                gt = [y_batch[0][i], y_batch[1][i]]
            pred = model.predict(np.expand_dims(x_batch[i], axis=0))
            pred = np.squeeze(np.array(pred).tolist())
            error = gt - pred
            errors.append(error)
            if plot_images:
                plt.imshow(np.squeeze(x_batch[i]))
                plt.show()
            print('GT:        {}'.format(gt))
            print('Predicted: {}'.format(pred))
            print('error:     {}'.format(error))

        loops -= 1
        if not loops:
            break
    print('Mean error: {}'.format(sum(errors) / len(errors)))


def predict_complete_dataframe(model, dataframe, input_column, target_columns, input_shape, debug=False):
    """
    Evaluates model on train, validation and test data.
    :param target_columns: name of the columns forming the output
    :param debug: boolean indicating if extra information should be printed for debugging purposes
    :param input_shape: shape of the input image
    :param model: tensorflow model
    :param dataframe: dataframe containing information relevant to the experiment
    :param input_column: name of the column containing the paths to the input images
    :return:
    """
    start = time.time()
    # Can be optimized using a generator, but I found ordering errors in tf 2.2 complete generator with shuffle=False
    print('Predicting values from the complete dataframe, this could take a while')
    nn_predictions = []
    for index, row in dataframe.iterrows():
        img_path = None if input_column == 'mask_path' else row['complete_path']
        mask_path = None if input_column == 'complete_path' else row['mask_path']
        nn_predictions.append(
            nn_predict_imt(img_path=img_path, mask_path=mask_path, model=model, input_shape=input_shape,
                           target_columns=target_columns))
    dataframe['nn_prediction'] = nn_predictions
    for key, value in target_columns.items():
        if value['predict']:
            prediction_column_name = 'predicted_{}'.format(key)
            dataframe[prediction_column_name] = dataframe['nn_prediction'].apply(lambda x: x[key])
    print('Original method took {:.02f}'.format(time.time() - start))
    return dataframe


def predict_complete_dataframe_generator(model, complete_data_generator, dataframe, target_columns, batch_size,
                                         input_column, input_shape):
    """
    Evaluates model on train, validation and test data.
    :param target_columns: name of the columns forming the output
    :param debug: boolean indicating if extra information should be printed for debugging purposes
    :param input_shape: shape of the input image
    :param model: tensorflow model
    :param dataframe: dataframe containing information relevant to the experiment
    :param input_column: name of the column containing the paths to the input images
    :return:
    """
    start = time.time()

    # Can be optimized using a generator, but I found ordering errors in tf 2.2 complete generator with shuffle=False
    print('Predicting values from the complete dataframe, this could take a while')
    nn_predictions = model.predict(complete_data_generator, batch_size=batch_size, workers=0,
                                   steps=np.ceil(complete_data_generator.samples / complete_data_generator.batch_size))
    if np.array(nn_predictions).shape[1] == 1:
        nn_predictions = np.expand_dims(nn_predictions, axis=0)
    predicted_imt_max = []
    predicted_imt_avg = []
    predicted_plaque = []
    count = 0
    for key, value in target_columns.items():
        if value['predict']:
            if key == 'imt_max':
                predicted_imt_max = nn_predictions[count]
            elif key == 'imt_avg':
                predicted_imt_avg = nn_predictions[count]
            elif key == 'plaque':
                predicted_plaque = nn_predictions[count]
            else:
                raise NotImplementedError
            count += 1

    # TODO: Re-design
    # Sorry about this lines, but there seems to be an issue in TF about the order of the output of generators
    # https://github.com/keras-team/keras/issues/5048, or maybe there is a bug in my code that I am not able to find.
    # Anyway, I found that rolling batch size indexes fixes it, and the code is x10 faster using generators.
    # It should be changed in the future, it might be automatically fixed and then provoke a failure
    predicted_plaque = np.roll(predicted_plaque, batch_size)
    predicted_imt_avg = np.roll(predicted_imt_avg, batch_size)
    predicted_imt_max = np.roll(predicted_imt_max, batch_size)

    # Just a sanity check, we can compare the first values with manual predictions:
    values_to_compare = 20
    manual_nn_predictions = []
    counter = 0
    for index, row in dataframe.iterrows():
        img_path = None if input_column == 'mask_path' else row['complete_path']
        mask_path = None if input_column == 'complete_path' else row['mask_path']
        manual_nn_predictions.append(
            nn_predict_imt(img_path=img_path, mask_path=mask_path, model=model, input_shape=input_shape,
                           target_columns=target_columns))
        if counter >= values_to_compare:
            break
        counter += 1
    # for i, element in enumerate(manual_nn_predictions):
    #     for mode_key in element.keys():
    #         if mode_key == 'imt_max':
    #             if not np.isclose(element[mode_key], predicted_imt_max[i], atol=1.e-2): # The values can be slightly different due to the resize process
    #                 #raise Exception('The complete data generator is not working as expected')
    #
    #         if mode_key == 'imt_avg':
    #             if not np.isclose(element[mode_key], predicted_imt_avg[i], atol=1.e-2):
    #                 #raise Exception('The complete data generator is not working as expected')
    #
    #         if mode_key == 'plaque':
    #             if not np.isclose(element[mode_key], predicted_plaque[i], atol=1.e-2):
    #                 raise Exception('The complete data generator is not working as expected')

    for key, value in target_columns.items():
        if value['predict']:
            prediction_column_name = 'predicted_{}'.format(key)
            if key == 'imt_max':
                dataframe[prediction_column_name] = predicted_imt_max
            elif key == 'imt_avg':
                dataframe[prediction_column_name] = predicted_imt_avg
            elif key == 'plaque':
                dataframe[prediction_column_name] = predicted_plaque
            else:
                raise NotImplementedError
    print('New method took {:.02f}'.format(time.time() - start))

    return dataframe


def train_imt_predictor(database=config.DATABASE, input_type=config.INPUT_TYPE, input_shape=config.INPUT_SHAPE,
                        target_columns=config.TARGET_COLUMNS,
                        random_seed=config.RANDOM_SEED, learning_rate=config.LEARNING_RATE, debug=config.DEBUG,
                        train_model=config.TRAIN, use_mixed_precision=config.MIXED_PRECISION, epochs=config.EPOCHS,
                        batch_size=config.BATCH_SIZE, early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                        rlr_on_plateau_patience=config.RLR_ON_PLATEAU_PATIENCE,
                        n_workers=config.WORKERS, max_queue_size=config.MAX_QUEUE_SIZE,
                        data_augmentation_params=config.DATA_AUGMENTATION_PARAMS, train_percent=config.TRAIN_PERCENTAGE,
                        valid_percent=config.VAL_PERCENTAGE, test_percent=config.TEST_PERCENTAGE,
                        resume_training=config.RESUME_TRAINING, silent_mode=config.SILENT_MODE,
                        suffix=config.EXPERIMENT_SUFFIX, dropout_rate=config.DROPOUT_RATE):
    """
    Complete training pipeline. Values can be set on the config.py or directly on function call.

    :param rlr_on_plateau_patience:
    :param dropout_rate:
    :param database: string representing the database. Could be 'CCA' or 'BULB'
    :param input_type: string indicating if the input of the network should be the original image or the segmented mask
    :param input_shape: tuple containing the shape of the input image
    :param target_columns: name of the columns forming the output
    :param compare_results: boolean indicating if results should be compared to the ones in M.d.M Vila et al.
    :param random_seed: number used to ensure reproducibility of results
    :param learning_rate: starting learning rate value for the model
    :param debug: boolean indicating if extra information should be printed for debugging purposes
    :param train_model: boolean indicating if the network should be trained. If False, only the evaluation will be performed
    :param use_mixed_precision: boolean indicating if mixed precision is used. Compute capability >7.0 is required.
    :param epochs: number of passes through the complete data-set in the training process.
    :param batch_size: size of batches generated by the generator
    :param early_stopping_patience: max number of epochs without improvements in val_loss
    :param n_workers: number of CPU cores used in the training process. A value of -1 will use all available ones
    :param max_queue_size: maximum number of batches to pre-compute to avoid bottlenecks.
    :param data_augmentation_params: dict containing data augmentation for training. See tf ImageDataGenerator
    :param train_percent: percentage of values used for training
    :param valid_percent: percentage of values used for validation
    :param test_percent: percentage of values used for testing
    :param resume_training: boolean indicating if previous best performing model should be loaded before training
    :param silent_mode: boolean indicating if all outputs should be suppressed
    :param suffix: string to distinguish between experiments

    """

    # Define experiment id
    output_id = '_'.join([key.replace('imt_', '') for key, value in target_columns.items() if value['predict']])
    experiment_id = '{}_{}_{}_{}_{}'.format(database, input_type, input_shape[0], output_id, suffix)
    if experiment_id[-1] == '_':  # no sufix
        experiment_id = experiment_id[:-1]

    # Create a folder to save the results
    experiment_folder_path = os.path.join('experiments', experiment_id)
    if os.path.exists(experiment_folder_path) and train_model:
        experiment_repetitions = 0
        while os.path.exists(experiment_folder_path):
            experiment_repetitions += 1
            experiment_folder_path = os.path.join('experiments', experiment_id + '_' + str(experiment_repetitions))

    print("Experiment id: {}, saving results in {}".format(experiment_id, experiment_folder_path))
    if train_model:
        os.makedirs(experiment_folder_path, exist_ok=True)
        os.makedirs(os.path.join(experiment_folder_path, 'input'), exist_ok=True)
        os.makedirs(os.path.join(experiment_folder_path, 'training_logs'), exist_ok=True)
        os.makedirs(os.path.join(experiment_folder_path, 'results'), exist_ok=True)
        helpers.save_training_config(frame=inspect.currentframe(), folder=experiment_folder_path)

    # Set random seeds
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    # Load data from disk
    df = pd.read_csv(os.path.join('segmentation', 'complete_data_{}.csv'.format(database)), index_col=0)

    # Convert to dataframe and filter invalid values
    df = filter_dataframe(df, database)

    # Shuffle dataframe
    df = df.sample(frac=1, random_state=random_seed)
    # Change index format #TODO: fix in previous step (Done?)
    df.index = df.index.map(lambda x: x[:-1])
    print(df.head())

    # Add columns with results from https://doi.org/10.1016/j.artmed.2019.101784
    df_merged = add_previous_results(df.copy(), database=database)
    df = df_merged.reindex(df.index)
    df = df.dropna()
    print(df.head())
    # df_merged.to_csv('df_merged.csv')

    df['gt_plaque'] = df['gt_imt_max'].apply(lambda x: 1 if x >= 1.5 else 0)

    device_name = tf.test.gpu_device_name()
    if config.FORCE_GPU:
        if device_name != '/device:GPU:0':
            raise SystemError('GPU device not found')
        if not silent_mode:
            print('Found GPU at: {}'.format(device_name))

    if input_type == 'img':
        input_column = 'complete_path'
    elif input_type == 'mask':
        input_column = 'mask_path'
    elif input_type == 'img_and_mask':
        input_column = input_type
    else:
        raise NotImplementedError

    # TODO: Clean Bulb df
    if database == 'BULB':
        df['complete_path'] = df['complete_path'].apply(lambda x: x[1:])

    selected_columns = ['gt_' + key for key, value in target_columns.items() if value['predict']]
    n_outputs = len(selected_columns)
    if n_outputs == 1:
        target_column = selected_columns[0]
    else:
        df['target_column'] = df[selected_columns].values.tolist()
        df['target_column'] = df['target_column'].to_numpy()
        target_column = 'target_column'

    df_train, df_valid, df_test, df = helpers.train_validate_test_split(df, train_percent=train_percent,
                                                                        validate_percent=valid_percent,
                                                                        test_percent=test_percent)
    if train_model:
        helpers.save_input_data(experiment_folder_path, df_train, df_valid, df_test)
    model = get_imt_prediction_model(input_type=input_type, input_shape=input_shape, target_columns=target_columns,
                                     dropout_rate=dropout_rate)

    weights_path = os.path.join(experiment_folder_path, 'best_validation_weights.h5')
    # weights_path = weights_path.replace('__','_') # to make it compatible with old code

    if resume_training:
        if os.path.exists(weights_path):
            model.load_weights(weights_path)

    optimizer = Adam(lr=learning_rate)

    # Define losses and weights depending on number of outputs
    losses = []
    metrics = {}
    loss_weights = []

    for key, value in target_columns.items():
        if value['predict']:
            if key == 'plaque':
                metrics['plaque'] = [Recall(name='recall'), 'accuracy']
            losses.append(value['loss'])
            loss_weights.append(value['weight'])

    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

    # Define data generators
    complete_data_generator = data_generator(mode='complete', dataframe=df, input_column=input_column,
                                             target_column=target_column, batch_size=batch_size,
                                             data_augmentation_params=data_augmentation_params, input_shape=input_shape
                                             , n_outputs=n_outputs, seed=config.RANDOM_SEED)
    train_generator = data_generator(mode='train', dataframe=df_train, input_column=input_column,
                                     target_column=target_column, batch_size=batch_size,
                                     data_augmentation_params=data_augmentation_params, input_shape=input_shape
                                     , n_outputs=n_outputs, seed=config.RANDOM_SEED)
    valid_generator = data_generator(mode='valid', dataframe=df_valid, input_column=input_column,
                                     target_column=target_column, batch_size=batch_size, input_shape=input_shape
                                     , n_outputs=n_outputs, seed=config.RANDOM_SEED)
    test_generator = data_generator(mode='test', dataframe=df_test, input_column=input_column,
                                    target_column=target_column, batch_size=batch_size, input_shape=input_shape
                                    , n_outputs=n_outputs, seed=config.RANDOM_SEED)

    if debug:
        helpers.test_generator_output(test_generator, n_images=2)

    # Define callbacks
    if train_model:
        tensorboard_log_path = os.path.join(experiment_folder_path, 'training_logs', 'run_{}.h5'.format(experiment_id))
        callbacks = [ModelCheckpoint(filepath=weights_path, save_best_only=True, verbose=True, monitor='val_loss'),
                     TensorBoard(
                         log_dir=tensorboard_log_path, profile_batch=0, write_graph=False),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=rlr_on_plateau_patience, min_lr=1e-8,
                                       verbose=True),
                     EarlyStopping(monitor='val_loss', patience=early_stopping_patience)]

        # Mixed precision can speedup the training process and lower the memory usage, CC>7 required
        if use_mixed_precision:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)

        # Define number of steps for the generator to cover all the data
        TRAINING_STEPS = train_generator.n // train_generator.batch_size
        VALIDATION_STEPS = valid_generator.n // valid_generator.batch_size

        # Execute training
        history = model.fit(train_generator,
                            steps_per_epoch=TRAINING_STEPS,
                            validation_data=valid_generator,
                            validation_steps=VALIDATION_STEPS,
                            epochs=epochs,
                            max_queue_size=max_queue_size,
                            workers=n_workers,
                            use_multiprocessing=False,  # tf 2.2 recommends using tf.data for multiprocessing
                            callbacks=callbacks)
        # if not silent_mode:
        helpers.plot_training_history(history, experiment_id, experiment_folder_path)
    # Load the best performing weights for the validation set
    model.load_weights(weights_path, )

    model_path = os.path.join(experiment_folder_path, 'model_{}.h5'.format(experiment_id))

    # Evaluation # TODO: Move to another file

    if debug:
        plot_predictions(model, test_generator)
    # if not silent_mode:
    mode_list = [key for key, value in target_columns.items() if value['predict']]
    # df_old = predict_complete_dataframe(model=model, dataframe=df.copy(), input_column=input_column, input_shape=input_shape,
    #                                 debug=debug, target_columns=target_columns)
    df = predict_complete_dataframe_generator(model=model, dataframe=df.copy(), target_columns=target_columns,
                                              complete_data_generator=complete_data_generator, batch_size=batch_size,
                                              input_shape=input_shape, input_column=input_column)
    results_path = os.path.join(experiment_folder_path, 'results', 'complete_predictions.csv')
    df.to_csv(results_path)
    # results_old_path = os.path.join(experiment_folder_path, 'results', 'complete_predictions_old.csv')
    # df_old.to_csv(results_old_path)
    helpers.evaluate_performance(dataframe=df, mode_list=mode_list,
                                 exp_id=experiment_id, experiment_folder_path=experiment_folder_path, debug=debug)
    if not silent_mode:
        helpers.save_model(model, model_path)


if __name__ == '__main__':
    train_imt_predictor()
