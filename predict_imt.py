#!/usr/bin/env python
# coding: utf-8
import os
import time

import cv2
import numpy as np
import pandas as pd
import pingouin as pg
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
from helpers import add_previous_results, filter_dataframe, get_classification_results
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


def nn_predict_imt(path, model, input_shape, target_columns):
    """
    Predicts the IMT for an image given its path
    :param target_columns: name of the columns forming the output
    :param input_shape: shape of the input image
    :param path: path to the image to predict
    :param model: tensorflow model for IMT prediction
    :return: predicted IMT values, specific targets depends on the model
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.
    img = cv2.resize(img, input_shape)
    prediction = model.predict(np.expand_dims(img, axis=0))

    result = {}
    count = 0
    for key, value in target_columns.items():
        if value['predict']:
            result[key] = np.squeeze(prediction[count])
            count += 1
    return result


def get_metrics(df, exp_id, target_columns, subset=None, compare=True):
    """
    Generate a report of the performance of a model. A subset can be specified, and results can be compared with the ones
    in https://doi.org/10.1016/j.artmed.2019.101784 .
    :param target_columns: name of the columns forming the output
    :param exp_id: string representing the experiment
    :param df: pandas dataframe with paths to images of interest for the experiment
    :param subset: subset of data to evaluate, can be train, valid or test.
    :param compare: boolean indicating if results should be compared with the ones in M.d.M Vila et al.
    """
    mode_list = [key for key, value in target_columns.items() if value['predict']]
    for nn_prediction_mode in mode_list:
        print('\nPredictions of {}'.format(nn_prediction_mode))
        # TODO: repeated lines, convert into functions
        if subset is not None:
            df = df[df['training_group'] == subset]
            print('Filtering to {}'.format(subset))
        if nn_prediction_mode != 'plaque':
            print('Mean {} nn predictions error: {:.4f}'.format(nn_prediction_mode,
                                                                df['nn_predictions_error_{}'.format(
                                                                    nn_prediction_mode)].mean()))
            print('Mean {} nn predictions squared error: {:.4f}'.format(nn_prediction_mode,
                                                                        df['nn_predictions_squared_error_{}'.format(
                                                                            nn_prediction_mode)].mean()))
            print('{} Correlation coefficient: {:.4f}'.format(nn_prediction_mode,
                                                              df['gt_{}'.format(nn_prediction_mode)].corr(
                                                                  df['nn_predictions_{}'.format(nn_prediction_mode)])))
            if compare:
                print('MdM results for this subset:')
                print('Mean {} nn predictions error: {:.4f}'.format(nn_prediction_mode,
                                                                    df['mdm_nn_predictions_error_{}'.format(
                                                                        nn_prediction_mode)].mean()))
                print('Mean {} nn predictions squared error: {:.4f}'.format(nn_prediction_mode,
                                                                            df[
                                                                                'mdm_nn_predictions_squared_error_{}'.format(
                                                                                    nn_prediction_mode)].mean()))
                print('{} Correlation coeficient: {:.4f}'.format(nn_prediction_mode,
                                                                 df['gt_{}'.format(nn_prediction_mode)].corr(
                                                                     df['mdm_{}_est'.format(nn_prediction_mode)])))

            ax = pg.plot_blandaltman(df['nn_predictions_{}'.format(nn_prediction_mode)].to_numpy(),
                                     df['gt_{}'.format(nn_prediction_mode)].to_numpy())
            ax.set_xlabel('Average of CIMT values (mm)')
            ax.set_ylabel('Difference of CIMT values (mm)')
            # plt.title('Error {}, exp: {}'.format(nn_prediction_mode, exp_id))
            plt.show()
            ax = df.plot.scatter(x='gt_{}'.format(nn_prediction_mode),
                                 y='nn_predictions_{}'.format(nn_prediction_mode))
            ax.plot([0, 1], [0, 1], transform=ax.transAxes)
            plt.title('Scatter plot {}, exp: {}'.format(nn_prediction_mode, exp_id))
            plt.show()
            if nn_prediction_mode == 'imt_max':
                for i in range(10, 16):
                    print('\n Thr: ' + str(i / 10))
                    df['predicted_plaque'] = df['nn_predictions_{}'.format(nn_prediction_mode)].apply(
                        lambda x: 1 if x > i / 10 else 0)
                    get_classification_results(df['gt_plaque'].to_numpy(),
                                               df['predicted_plaque'].to_numpy())
                    if compare:
                        print('MdM result:')
                        df['mdm_predicted_plaque'] = df['mdm_{}_est'.format(nn_prediction_mode)].apply(
                            lambda x: 1 if x > i / 10 else 0)
                        get_classification_results(df['gt_plaque'].to_numpy(),
                                                   df['mdm_predicted_plaque'].to_numpy())
        else:
            for i in range(1, 11):
                print('\n Thr: ' + str(i / 10))
                df['predicted_plaque'] = df['nn_predictions_{}'.format(nn_prediction_mode)].apply(
                    lambda x: 1 if x > i / 10 else 0)
                get_classification_results(df['gt_plaque'].to_numpy(),
                                           df['predicted_plaque'].to_numpy())


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


def evaluate_model(model, dataframe, input_column, target_columns, input_shape, exp_id, debug=False,
                   compare_results=False):
    """
    Evaluates model on train, validation and test data.
    :param exp_id: string representing the experiment
    :param target_columns: name of the columns forming the output
    :param compare_results: boolean indicating if results should be compared to the ones in M.d.M Vila et al.
    :param debug: boolean indicating if extra information should be printed for debugging purposes
    :param input_shape: shape of the input image
    :param model: tensorflow model
    :param dataframe: dataframe containing information relevant to the experiment
    :param input_column: name of the column containing the paths to the input images
    :return:
    """
    if debug:
        print(nn_predict_imt(path=dataframe.complete_path.values[0], model=model, input_shape=input_shape,
                             target_columns=target_columns))

    # Can be optimized using a generator, but I found ordering errors in tf 2.2 complete generator with shuffle=False
    print('Predicting values from the complete dataframe, this could take a while')
    dataframe['nn_prediction'] = dataframe[input_column].apply(
        lambda x: nn_predict_imt(path=x, model=model, input_shape=input_shape,
                                 target_columns=target_columns))

    for key, value in target_columns.items():
        if value['predict']:
            prediction_column_name = 'nn_predictions_{}'.format(key)
            error_column_name = 'nn_predictions_error_{}'.format(key)
            squared_error_column_name = 'nn_predictions_squared_error_{}'.format(key)
            dataframe[prediction_column_name] = dataframe['nn_prediction'].apply(lambda x: x[key])
            dataframe[error_column_name] = dataframe['gt_{}'.format(key)] - dataframe[prediction_column_name]
            dataframe[squared_error_column_name] = dataframe[error_column_name] * dataframe[error_column_name]
            if compare_results and key != 'plaque':
                dataframe['mdm_' + error_column_name] = dataframe['gt_{}'.format(key)] - dataframe[
                    'mdm_{}_est'.format(key)]
                dataframe['mdm_' + squared_error_column_name] = dataframe['mdm_' + error_column_name] * dataframe[
                    'mdm_' + error_column_name]

    # get_metrics(dataframe)
    get_metrics(dataframe, exp_id=exp_id, subset='train', target_columns=target_columns, compare=compare_results)
    get_metrics(dataframe, exp_id=exp_id, subset='valid', target_columns=target_columns, compare=compare_results)
    get_metrics(dataframe, exp_id=exp_id, subset='test', target_columns=target_columns, compare=compare_results)


def train_imt_predictor(database=config.DATABASE, input_type=config.INPUT_TYPE, input_shape=config.INPUT_SHAPE,
                        target_columns=config.TARGET_COLUMNS, compare_results=config.COMPARE_RESULTS,
                        random_seed=config.RANDOM_SEED, learning_rate=config.LEARNING_RATE, debug=config.DEBUG,
                        train=config.TRAIN, use_mixed_precision=config.MIXED_PRECISION, epochs=config.EPOCHS,
                        batch_size=config.BATCH_SIZE, early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                        n_workers=config.WORKERS, max_queue_size=config.MAX_QUEUE_SIZE,
                        data_augmentation_params=config.DATA_AUGMENTATION_PARAMS, train_percent=config.TRAIN_PERCENTAGE,
                        valid_percent=config.VAL_PERCENTAGE, test_percent=config.TEST_PERCENTAGE,
                        resume_training=config.RESUME_TRAINING, silent_mode=config.SILENT_MODE):
    """
    Complete training pipeline. Values can be set on the config.py or directly on function call.

    :param database: string representing the database. Could be 'CCA' or 'BULB'
    :param input_type: string indicating if the input of the network should be the original image or the segmented mask
    :param input_shape: tuple containing the shape of the input image
    :param target_columns: name of the columns forming the output
    :param compare_results: boolean indicating if results should be compared to the ones in M.d.M Vila et al.
    :param random_seed: number used to ensure reproducibility of results
    :param learning_rate: starting learning rate value for the model
    :param debug: boolean indicating if extra information should be printed for debugging purposes
    :param train: boolean indicating if the network should be trained. If False, only the evaluation will be performed
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
    """

    # Define experiment id
    output_id = '_'.join([key.replace('imt_', '') for key, value in target_columns.items() if value['predict']])
    experiment_id = '{}_{}_{}_{}'.format(database, input_type, input_shape[0], output_id)
    print("Experiment id: {}".format(experiment_id))

    # Set random seeds
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    # Load data from disk
    data = np.load(os.path.join('segmentation', 'complete_data_{}.npy'.format(database)))
    data = data[()]['data']

    # Convert to dataframe and filter invalid values
    df = pd.DataFrame.from_dict(data, orient='index')
    df = filter_dataframe(df, database)

    # Change index format #TODO: fix in previous step
    df.index = df.index.map(lambda x: x[4:-1])

    if compare_results:
        # Add columns with results from https://doi.org/10.1016/j.artmed.2019.101784
        df = add_previous_results(df, database=database)

    df['gt_plaque'] = df['gt_imt_max'].apply(lambda x: 1 if x > 1.5 else 0)

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    if not silent_mode:
        print('Found GPU at: {}'.format(device_name))

    if input_type == 'img':
        input_column = 'complete_path'
    elif input_type == 'mask':
        input_column = 'mask_path'

    # Shuffle dataframe
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

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
    df['training_group'].value_counts()

    model = get_imt_prediction_model()

    weights_path = 'checkpoints/weights_{}.h5'.format(experiment_id)
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
    train_generator = data_generator(mode='train', dataframe=df_train, input_column=input_column,
                                     target_column=target_column, batch_size=batch_size,
                                     data_augmentation_params=data_augmentation_params, input_shape=input_shape
                                     , n_outputs=n_outputs)
    valid_generator = data_generator(mode='valid', dataframe=df_valid, input_column=input_column,
                                     target_column=target_column, batch_size=batch_size, input_shape=input_shape
                                     , n_outputs=n_outputs)
    test_generator = data_generator(mode='test', dataframe=df_test, input_column=input_column,
                                    target_column=target_column, batch_size=batch_size, input_shape=input_shape
                                    , n_outputs=n_outputs)

    if debug:
        helpers.test_generator_output(test_generator, n_images=2)

    # Define callbacks
    os.makedirs('checkpoints', exist_ok=True)

    if train:
        callbacks = [ModelCheckpoint(filepath=weights_path, save_best_only=True, verbose=True, monitor='val_loss'),
                     TensorBoard(
                         log_dir='logs/run_{}_{}.h5'.format(experiment_id, time.time())),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=True),
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
        if not silent_mode:
            helpers.plot_training_history(history, experiment_id)

    # Load the best performing weights for the validation set
    model.load_weights(weights_path)

    model_path = 'models/model_{}.h5'.format(experiment_id)

    if debug:
        plot_predictions(model, test_generator)
    if not silent_mode:
        evaluate_model(model=model, dataframe=df, input_shape=input_shape, input_column=input_column,
                       target_columns=target_columns, compare_results=compare_results, debug=debug,
                       exp_id=experiment_id)

    helpers.save_model(model, model_path)


if __name__ == '__main__':
    train_imt_predictor()
