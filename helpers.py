#!/usr/bin/env python
# coding: utf-8

import os
import time

import numpy as np
import pandas as pd
import pingouin as pg
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def filter_dataframe(dataframe, database):
    """

    :param dataframe: pandas dataframe with paths to images of interest for the experiment
    :param database: name of the database, can be CCA or BULB
    :return: filtered database
    """
    if database == 'CCA':
        CCA_filter_list = ['051171144_RCCA',  # the PLAQUE is in the bulb region
                           '11727784_LCCA',  # the PLAQUE is in the bulb region
                           '11741384_RCCA',  # the PLAQUE is in the bulb region
                           '22142360_RCCA',  # the PLAQUE is in the bulb region
                           '22189560_RCCA',  # the PLAQUE is in the bulb region
                           '22239360_LCCA',  # the PLAQUE is in the bulb region
                           '22138460_RCCA',  # the PLAQUE is in the bulb region
                           '22265660_RCCA',  # the PLAQUE is in the bulb region
                           '053098156_LCCA',  # the PLAQUE is in the right part of the image
                           '22153060_RCCA',  # not a Dicom image
                           '22251060_RCCA',  # different resolution
                           '22265660_LCCA',  # different resolution. Contains PLAQUE in the GT
                           '22406960_RCCA'  # bulb in center. Plaque in bulb not in GT
                           ]

        # TODO: Change name format to avoid conversion
        CCA_filter_list = ['img:' + i + 'g' for i in CCA_filter_list]

        dataframe = dataframe[~dataframe.index.isin(CCA_filter_list)]

    return dataframe


def get_classification_results(gt_plaque, pred_plaque):
    """
    Prints plaque classification results
    :param gt_plaque: numpy array with the gt values of plaque
    :param pred_plaque: numpy array with the predicted values of plaque
    """
    cm1 = confusion_matrix(gt_plaque, pred_plaque, labels=[1, 0])
    print('Confusion Matrix : \n tp   fp\n fn   tn\n', cm1)

    total1 = sum(sum(cm1))
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    sensitivity1 = cm1[0, 0] / (
            cm1[0, 0] + cm1[0, 1])  # true positive rate, the recall, or probability of detection
    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[
        1, 1])  # measures the proportion of actual negatives that are correctly identified as such
    print('Accuracy:    {:.4f}'.format(accuracy1))
    print('Sensitivity: {:.4f}'.format(sensitivity1))
    print('Specificity: {:.4f}'.format(specificity1))


def add_previous_results(dataframe, database):
    """

    :param dataframe: pandas dataframe with paths to images of interest
    :return: original dataframe with previous results appended
    """

    assert database in ['CCA', 'BULB'], 'No previous results for this database'

    if database == 'CCA':
        df_names = pd.read_csv('results_mdm_et_al/names_postprT56_1cm.csv', names=['file_name'], header=None)
        df_prev_results = pd.read_csv('results_mdm_et_al/imts_postprT56_1cm.csv',
                                      names=['mdm_imtmax_est',  # Max IMT estimated with Tiramisu56
                                             'gt_imtm_CCA',  # Max IMT from Ground Truth (AMC)
                                             'mdm_imtavg_est',  # Mean IMT estimated with Tiramisu56
                                             'gt_imta_CCA'],  # Mean IMT from Ground Truth (AMC)
                                      header=None)

        df_prev_results.index = df_names['file_name']
    elif database == 'BULB':
        df_names = pd.read_csv('results_mdm_et_al/names__postprT67_BUnewGT_4L.csv', names=['file_name'],
                               header=None)
        df_prev_results = pd.read_csv('results_mdm_et_al/imts_postprT67_BUnewGT_4L.csv',
                                      names=['mdm_imtmax_est',  # Max IMT estimated with Tiramisu56
                                             'gt_imtm_CCA',  # Max IMT from Ground Truth (AMC)
                                             'mdm_imtavg_est',  # Mean IMT estimated with Tiramisu56
                                             'gt_imta_CCA'],  # Mean IMT from Ground Truth (AMC)
                                      header=None)
        df_prev_results.index = df_names['file_name']

    df_prev_results = df_prev_results.rename(
        columns={"mdm_imtmax_est": "mdm_imt_max_est", "mdm_imtavg_est": "mdm_imt_avg_est"})
    df_merged = pd.concat([dataframe, df_prev_results], axis=1, sort=True)
    df_merged = df_merged.dropna()
    return df_merged


def train_validate_test_split(dataframe, train_percent,
                              validate_percent,
                              test_percent):
    """
    Split a dataframe into training, validation and testing using the given proportions. It also ads a column in the
    original dataframe indicating the subset selected for each element.
    :param dataframe: pandas dataframe with paths to images of interest for the experiment
    :param train_percent: percentage of data reserved for training
    :param validate_percent: percentage of data reserved for validation
    :param test_percent: percentage of data reserved for testing
    :return: three datasets containing subsets of the data and the original one with training_group column added
    """
    assert train_percent + test_percent + validate_percent == 1, 'Train, val and test % should add up 1'
    length_df = len(dataframe.index)
    train_end = int(train_percent * length_df)
    validate_end = int(validate_percent * length_df) + train_end
    train = dataframe.iloc[:train_end]
    validate = dataframe.iloc[train_end:validate_end]
    test = dataframe.iloc[validate_end:]
    dataframe['training_group'] = 'train'
    dataframe.loc[train_end:validate_end, 'training_group'] = 'valid'
    dataframe.loc[validate_end:, 'training_group'] = 'test'

    return train, validate, test, dataframe


def save_model(model, path):
    """
    Saves tensorflow model after user confirmation. Creates required folders
    :param model: tensorflow model
    :param path: path to save the model
    :return:
    """
    if os.path.exists(path):

        confirmation = input('Are you sure you want to overwrite the model? (y/n) ')
        if confirmation == 'y':
            model.save(path)
            print('Model saved')
        else:
            print('Saving canceled')
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)
        print('Model saved')


def test_generator_output(generator, n_images=10):
    """
    Plots the output of a given generator for testing purposes
    :param generator: generator to evaluate
    :param n_images: number of images to plot
    """
    for x_batch, y_batch in generator:
        for i in range(generator.batch_size):
            plt.imshow(np.squeeze(x_batch[i]))
            plt.show()
            # print(y_batch[0][i])
            # print(y_batch[1][i])
            # if config.PREDICT_PLAQUE:  # TODO: Update
            #    print(y_batch[2][i])
            if i > n_images:
                break
        break


def test_prediction(model, input_shape):
    """
    Predict a dummy image to test if the model is working.
    :param model: tensorflow model
    """
    start = time.time()
    model.predict(np.zeros([1, input_shape[0], input_shape[1], 1]))
    print('prediction took {}s'.format(time.time() - start))


def plot_training_history(history, experiment_id):
    """
    Plots training history, more information can be found in the tensorboard log
    :param history: tensorflow keras training history
    :param experiment_id: identifier of current experiment
    """
    plt.figure(figsize=(20, 10))
    plt.plot(history.history['loss'], label='Combined loss (training data)')
    plt.plot(history.history['val_loss'], label='Combined loss (validation data)')
    # TODO: Make dynamic
    # for key in history.history.keys():
    #    plt.plot(history.history[key], label=key)

    plt.title('Training results, {}'.format(experiment_id))
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


def get_metrics(df, exp_id, mode_list, subset=None, compare=True):
    """
    Generate a report of the performance of a model. A subset can be specified, and results can be compared with the ones
    in https://doi.org/10.1016/j.artmed.2019.101784 .
    :param mode_list: list of modes to predict, could be any combination of 'plaque', 'max' and 'avg'
    :param exp_id: string representing the experiment
    :param df: pandas dataframe with paths to images of interest for the experiment
    :param subset: subset of data to evaluate, can be train, valid or test.
    :param compare: boolean indicating if results should be compared with the ones in M.d.M Vila et al.
    """

    for mode in mode_list:
        print('\nPredictions of {}'.format(mode))
        # TODO: repeated lines, convert into functions
        if subset is not None:
            df = df[df['training_group'] == subset]
            print('Filtering to {}'.format(subset))
        if mode != 'plaque':
            print('Mean {} nn predictions error: {:.4f}'.format(mode,
                                                                df['predicted_{}_error'.format(
                                                                    mode)].mean()))
            print('Mean {} nn predictions squared error: {:.4f}'.format(mode,
                                                                        df['predicted_{}_squared_error'.format(
                                                                            mode)].mean()))
            print('{} Correlation coefficient: {:.4f}'.format(mode,
                                                              df['gt_{}'.format(mode)].corr(
                                                                  df['predicted_{}'.format(mode)])))
            if compare:
                print('MdM results for this subset:')
                print('Mean {} nn predictions error: {:.4f}'.format(mode,
                                                                    df['mdm_predicted_{}_error'.format(
                                                                        mode)].mean()))
                print('Mean {} nn predictions squared error: {:.4f}'.format(mode,
                                                                            df[
                                                                                'mdm_predicted_{}_squared_error'.format(
                                                                                    mode)].mean()))
                print('{} Correlation coeficient: {:.4f}'.format(mode,
                                                                 df['gt_{}'.format(mode)].corr(
                                                                     df['mdm_{}_est'.format(mode)])))

            ax = pg.plot_blandaltman(df['predicted_{}'.format(mode)].to_numpy(),
                                     df['gt_{}'.format(mode)].to_numpy())
            ax.set_xlabel('Average of CIMT values (mm)')
            ax.set_ylabel('Difference of CIMT values (mm)')
            # plt.title('Error {}, exp: {}'.format(nn_prediction_mode, exp_id))
            plt.show()
            ax = df.plot.scatter(x='gt_{}'.format(mode),
                                 y='predicted_{}'.format(mode))
            ax.plot([0, 1], [0, 1], transform=ax.transAxes)
            plt.title('Scatter plot {}, exp: {}'.format(mode, exp_id))
            plt.show()
            if mode == 'imt_max':
                for i in range(10, 16):
                    print('\n Thr: ' + str(i / 10))
                    df['predicted_plaque'] = df['predicted_{}'.format(mode)].apply(
                        lambda x: 1 if x > i / 10 else 0)
                    get_classification_results(df['gt_plaque'].to_numpy(),
                                               df['predicted_plaque'].to_numpy())
                    if compare:
                        print('MdM result:')
                        df['mdm_predicted_plaque'] = df['mdm_{}_est'.format(mode)].apply(
                            lambda x: 1 if x > i / 10 else 0)
                        get_classification_results(df['gt_plaque'].to_numpy(),
                                                   df['mdm_predicted_plaque'].to_numpy())
        else:
            for i in range(1, 11):
                print('\n Thr: ' + str(i / 10))
                df['predicted_plaque'] = df['predicted_{}'.format(mode)].apply(
                    lambda x: 1 if x > i / 10 else 0)
                get_classification_results(df['gt_plaque'].to_numpy(),
                                           df['predicted_plaque'].to_numpy())


def evaluate_performance(dataframe, mode_list, exp_id, compare_results=False):
    """
    Evaluates model on train, validation and test data.
    :param exp_id: string representing the experiment
    :param target_columns: name of the columns forming the output
    :param compare_results: boolean indicating if results should be compared to the ones in M.d.M Vila et al.
    :param dataframe: dataframe containing information relevant to the experiment
    :return:
    """

    for mode in mode_list:
        prediction_column_name = 'predicted_{}'.format(mode)
        error_column_name = 'predicted_{}_error'.format(mode)
        squared_error_column_name = 'predicted_{}_squared_error'.format(mode)
        dataframe[error_column_name] = dataframe['gt_{}'.format(mode)] - dataframe[prediction_column_name]
        dataframe[squared_error_column_name] = dataframe[error_column_name] * dataframe[error_column_name]
        if compare_results and mode != 'plaque':
            dataframe['mdm_' + error_column_name] = dataframe['gt_{}'.format(mode)] - dataframe[
                'mdm_{}_est'.format(mode)]
            dataframe['mdm_' + squared_error_column_name] = dataframe['mdm_' + error_column_name] * dataframe[
                'mdm_' + error_column_name]

    get_metrics(dataframe, exp_id=exp_id, subset=None, mode_list=mode_list, compare=compare_results)
    get_metrics(dataframe, exp_id=exp_id, subset='train', mode_list=mode_list, compare=compare_results)
    get_metrics(dataframe, exp_id=exp_id, subset='valid', mode_list=mode_list, compare=compare_results)
    get_metrics(dataframe, exp_id=exp_id, subset='test', mode_list=mode_list, compare=compare_results)
