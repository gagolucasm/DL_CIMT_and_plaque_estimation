#!/usr/bin/env python
# coding: utf-8

import inspect
import os
import time

import numpy as np
import pandas as pd
import pingouin as pg
from matplotlib import pyplot as plt
from sklearn import metrics


def filter_dataframe(dataframe, database):
    """

    :param dataframe: pandas dataframe with paths to images of interest for the experiment
    :param database: name of the database, can be CCA or BULB
    :return: filtered database
    """
    if database == 'CCA':
        filter_list = ['051171144_RCCAg',  # the PLAQUE is in the bulb region
                       '11727784_LCCAg',  # the PLAQUE is in the bulb region
                       '11741384_RCCAg',  # the PLAQUE is in the bulb region
                       '22142360_RCCAg',  # the PLAQUE is in the bulb region
                       '22189560_RCCAg',  # the PLAQUE is in the bulb region
                       '22239360_LCCAg',  # the PLAQUE is in the bulb region
                       '22138460_RCCAg',  # the PLAQUE is in the bulb region
                       '22265660_RCCAg',  # the PLAQUE is in the bulb region
                       '053098156_LCCAg',  # the PLAQUE is in the right part of the image
                       '22153060_RCCAg',  # not a Dicom image
                       '22251060_RCCAg',  # different resolution
                       '22265660_LCCAg',  # different resolution. Contains PLAQUE in the GT
                       '22406960_RCCAg'  # bulb in center. Plaque in bulb not in GT
                       '22115660_LCCAg',  # Possible mismatch in IMT and image assignation
                       '22115660_RCCAg',  # "" to end
                       '221156705_LCCAg',
                       '221156705_RCCAg',
                       '22327260_LCCAg',
                       '22327260_RCCAg',
                       '223272060_LCCAg',
                       '223272060_RCCAg',
                       '22254460_LCCAg',
                       '22254460_RCCAg',
                       '222544060_LCCAg',
                       '222544060_RCCAg'
                       ]

    elif database == 'BULB':
        filter_list = ['22115660_LBULg',  # Possible mismatch in IMT and image assignation
                       '22115660_RBULg',  # "" to end
                       '221156705_LBULg',
                       '221156705_RBULg',
                       '22327260_LBULg',
                       '22327260_RBULg',
                       '223272060_LBULg',
                       '223272060_RBULg',
                       '22254460_LBULg',
                       '22254460_RBULg',
                       '222544060_LBULg',
                       '222544060_RBULg'
                       ]
    else:
        raise NotImplementedError()
    dataframe = dataframe[~dataframe.index.isin(filter_list)]
    dataframe = dataframe.dropna()

    return dataframe


def save_training_config(frame, folder):
    args, _, _, values = inspect.getargvalues(frame)
    with open(os.path.join(folder, 'config.txt'), 'w') as fp:
        for i in args:
            fp.write("%s = %s \n" % (i, values[i]))


def get_classification_results(gt_plaque, pred_plaque):
    """
    Prints plaque classification results
    :param gt_plaque: numpy array with the gt values of plaque
    :param pred_plaque: numpy array with the predicted values of plaque
    """
    tn, fp, fn, tp = metrics.confusion_matrix(gt_plaque, pred_plaque, labels=[0, 1]).ravel()  # tn, fp, fn, tp
    print("""Confusion Matrix :
                            pred_neg       pred_pos
           actual_neg        tn: {}         fp: {}
           actual_pos        fn: {}         tp: {}  """.format(tn, fp, fn, tp))

    precision = tp / (tp + fp)  # closeness of the measurements to each other

    specificity = tn / (tn + fp)  # (also called the true negative rate) measures the proportion of actual negatives
    # that are correctly identified as such (e.g., the percentage of healthy people who are correctly identified as not
    # having the condition).

    accuracy = (tn + tp) / (tn + tp + fn + fp)  # closeness of the measurements to a specific value

    sensitivity = tp / (tp + fn)  # (also called the true positive rate, the epidemiological/clinical sensitivity,
    # the recall, or probability of detection[1] in some fields) measures the proportion of actual positives that
    # are correctly identified as such (e.g., the percentage of sick people who are correctly identified
    # as having the condition)

    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    print('Accuracy:    {:.4f}'.format(accuracy))
    print('Precision:   {:.4f}'.format(precision))
    print('Sensitivity: {:.4f}'.format(sensitivity))
    print('Specificity: {:.4f}'.format(specificity))
    print('F1 Score:    {:.4f}'.format(f1_score))

    return tn, fp, fn, tp, accuracy, precision, sensitivity, specificity, f1_score


def add_previous_results(dataframe, database):
    """

    :param database: name of database
    :param dataframe: pandas dataframe with paths to images of interest
    :return: original dataframe with previous results appended
    """

    assert database in ['CCA', 'BULB'], 'No previous results for this database'

    if database == 'CCA':
        df_names = pd.read_csv('results_vila_et_al/names_postprT56_1cm.csv', names=['file_name'], header=None)
        df_prev_results = pd.read_csv('results_vila_et_al/imts_postprT56_1cm.csv',
                                      names=['mdm_imtmax_est',  # Max IMT estimated with Tiramisu56
                                             'gt_imtm',  # Max IMT from Ground Truth (AMC)
                                             'mdm_imtavg_est',  # Mean IMT estimated with Tiramisu56
                                             'gt_imta'],  # Mean IMT from Ground Truth (AMC)
                                      header=None)

        # Load M.d.M postprocessing for our segmentations
        df_mdm_postprocessing = pd.read_csv('results_vila_et_al/CCA_end2end_postprocessing.csv',
                                            usecols=['namesimg', 'imt_max', 'GT_maxIMT', 'imt_mean', 'GT_meanIMT'])

    elif database == 'BULB':
        df_names = pd.read_csv('results_vila_et_al/names__postprT67_BUnewGT_4L.csv', names=['file_name'],
                               header=None)
        df_prev_results = pd.read_csv('results_vila_et_al/imts_postprT67_BUnewGT_4L.csv',
                                      names=['mdm_imtmax_est',  # Max IMT estimated with Tiramisu56
                                             'gt_imtm',  # Max IMT from Ground Truth (AMC)
                                             'mdm_imtavg_est',  # Mean IMT estimated with Tiramisu56
                                             'gt_imta'],  # Mean IMT from Ground Truth (AMC)
                                      header=None)
        # Load M.d.M postprocessing for our segmentations
        df_mdm_postprocessing = pd.read_csv('results_vila_et_al/BULB_end2end_postprocessing.csv',
                                            usecols=['namesimg', 'imt_max', 'GT_maxIMT', 'imt_mean', 'GT_meanIMT'])

    df_prev_results.index = df_names['file_name']
    df_mdm_postprocessing.index = df_mdm_postprocessing['namesimg']
    df_prev_results = df_prev_results.rename(
        columns={"mdm_imtmax_est": "mdm_imt_max_est", "mdm_imtavg_est": "mdm_imt_avg_est"})
    df_mdm_postprocessing = df_mdm_postprocessing.rename(
        columns={"imt_max": "mdm_post_imt_max_est", "imt_mean": "mdm_post_imt_avg_est"})
    df_merged = pd.concat([dataframe, df_prev_results], axis=1, sort=True)
    df_merged = df_merged.dropna()
    # df_merged = pd.concat([df_merged, df_mdm_postprocessing], axis=1, sort=True)
    # df_merged = df_merged.dropna()
    # print(sum(df_merged['gt_imt_max'].eq(df_merged['gt_imtm']))/len(df_merged)) # TODO: GT not matching
    # print(sum(df_merged['gt_imt_max'].eq(df_merged['GT_maxIMT']))/len(df_merged)) # TODO: GT not matching
    # print(sum(df_merged['gt_imt_avg'].eq(df_merged['gt_imta']))/len(df_merged)) # TODO: GT not matching
    # print(sum(df_merged['gt_imt_avg'].eq(df_merged['GT_meanIMT']))/len(df_merged)) # TODO: GT not matching
    df_merged['gt_imt_avg'] = df_merged['gt_imta']
    df_merged['gt_imt_max'] = df_merged['gt_imtm']
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
    test_end = int(test_percent * length_df) + train_end
    train = dataframe.iloc[:train_end]
    test = dataframe.iloc[train_end:test_end]
    validate = dataframe.iloc[test_end:]
    dataframe['training_group'] = 'train'
    dataframe.loc[train_end:test_end, 'training_group'] = 'test'
    dataframe.loc[test_end:, 'training_group'] = 'valid'

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


def plot_training_history(history, experiment_id, experiment_path):
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
    plt.savefig(os.path.join(experiment_path, 'training_losses.png'))
    pd.DataFrame(history.history).to_csv(os.path.join(experiment_path, 'training_history.csv'))


def get_metrics(dataframe, mode_list, experiment_folder_path, debug):
    """
    Generate a report of the performance of a model. A subset can be specified, and results can be compared with the ones
    in https://doi.org/10.1016/j.artmed.2019.101784 .
    :param mode_list: list of modes to predict, could be any combination of 'plaque', 'max' and 'avg'
    :param exp_id: string representing the experiment
    :param dataframe: pandas dataframe with paths to images of interest for the experiment
    """
    results_df = pd.DataFrame(
        columns=['model', 'mode', 'subset', 'mean_error', 'std_error', 'MAE', 'MSE', 'CC', 'tn', 'fp', 'fn', 'tp',
                 'accuracy',
                 'precision', 'sensitivity', 'specificity', 'f1_score'])
    if 'plaque' in mode_list:
        optimal_thr = get_optimal_thr(dataframe[dataframe['training_group'] == 'valid']['gt_plaque'].to_numpy(),
                                      dataframe[dataframe['training_group'] == 'valid']['predicted_plaque'].to_numpy(),
                                      experiment_folder_path,
                                      debug=debug)

    for subset in ['complete', 'train', 'valid', 'test']:
        df = dataframe.copy()
        if subset != 'complete':
            df = df[df['training_group'] == subset]
            print('\n --- Filtering to {} ---'.format(subset))
        for mode in mode_list:
            print('\nPredictions of {}'.format(mode))
            if mode != 'plaque':
                # TODO:
                #  Change to mean average error and variance
                mean_error = df['predicted_{}_error'.format(mode)].mean()
                mean_absolute_error = df['predicted_{}_error'.format(mode)].abs().mean()
                error_std = df['predicted_{}_error'.format(mode)].std()
                squared_error = df['predicted_{}_squared_error'.format(mode)].mean()
                pearson_cc = df['gt_{}'.format(mode)].corr(df['predicted_{}'.format(mode)])
                print('Mean {} nn predictions error: {:.4f}, std: {:.4f}'.format(mode, mean_error, error_std))
                print('Mean {} nn predictions squared error: {:.4f}, MAE: {:.4f}'.format(mode, squared_error,
                                                                                         mean_absolute_error))
                print('{} Correlation coefficient: {:.4f}'.format(mode, pearson_cc))

                mdm_mean_error = df['mdm_predicted_{}_error'.format(mode)].mean()
                mdm_mean_absolute_error = df['mdm_predicted_{}_error'.format(mode)].abs().mean()
                mdm_error_std = df['mdm_predicted_{}_error'.format(mode)].std()
                mdm_squared_error = df['mdm_predicted_{}_squared_error'.format(mode)].mean()
                mdm_pearson_cc = df['gt_{}'.format(mode)].corr(df['mdm_{}_est'.format(mode)])

                # mdm_post_mean_error = df['mdm_post_predicted_{}_error'.format(mode)].mean()
                # mdm_post_mean_absolute_error = df['mdm_post_predicted_{}_error'.format(mode)].abs().mean()
                # mdm_post_error_std = df['mdm_post_predicted_{}_error'.format(mode)].std()
                # mdm_post_squared_error = df['mdm_post_predicted_{}_squared_error'.format(mode)].mean()
                # mdm_post_pearson_cc = df['gt_{}'.format(mode)].corr(df['mdm_post_{}_est'.format(mode)])

                print('MdM results for this subset:')
                print(
                    'Mean {} nn predictions error: {:.4f}, std: {:.4f}'.format(mode, mdm_mean_error, mdm_error_std))
                print('Mean {} nn predictions squared error: {:.4f}, MAE: {:.4f}'.format(mode, mdm_squared_error,
                                                                                         mdm_mean_absolute_error))
                print('{} Correlation coeficient: {:.4f}'.format(mode, mdm_pearson_cc))

                # print('MdM postprocessing results for this subset:')
                # print(
                #     'Mean {} nn predictions error: {:.4f}, std: {:.4f}'.format(mode, mdm_post_mean_error, mdm_post_error_std))
                # print('Mean {} nn predictions squared error: {:.4f}, MAE: {:.4f}'.format(mode,
                #                                                                          mdm_post_squared_error,
                #                                                                          mdm_post_mean_absolute_error))
                # print('{} Correlation coeficient: {:.4f}'.format(mode, mdm_post_pearson_cc))

                ax = pg.plot_blandaltman(df['predicted_{}'.format(mode)].to_numpy(),
                                         df['gt_{}'.format(mode)].to_numpy())
                ax.set_xlabel('Average of CIMT values (mm)')
                ax.set_ylabel('Difference of CIMT values (mm)')
                # plt.title('Error {}, exp: {}'.format(nn_prediction_mode, exp_id))
                plt.savefig(os.path.join(experiment_folder_path, 'results', 'bland_altman_{}.png'.format(subset)))
                ax = df.plot.scatter(x='gt_{}'.format(mode),
                                     y='predicted_{}'.format(mode))
                ax.plot([0, 1], [0, 1], transform=ax.transAxes)
                plt.title('Scatter plot')
                plt.savefig(os.path.join(experiment_folder_path, 'results', 'scatter_{}.png'.format(subset)))
                if mode == 'imt_max':
                    thr = 1.5  # Manheim consensus
                    df['predicted_plaque_thr'] = df['predicted_{}'.format(mode)].apply(
                        lambda x: 1 if x >= thr else 0)
                    tn, fp, fn, tp, accuracy, precision, sensitivity, specificity, f1_score = get_classification_results(
                        df['gt_plaque'].to_numpy(),
                        df['predicted_plaque_thr'].to_numpy())

                    print('MdM result:')
                    df['mdm_predicted_plaque'] = df['mdm_{}_est'.format(mode)].apply(
                        lambda x: 1 if x >= thr else 0)
                    mdm_tn, mdm_fp, mdm_fn, mdm_tp, mdm_accuracy, mdm_precision, mdm_sensitivity, mdm_specificity, mdm_f1_score = get_classification_results(
                        df['gt_plaque'].to_numpy(),
                        df['mdm_predicted_plaque'].to_numpy())

                    # print('MdM postprocessing result:')
                    # df['mdm_post_predicted_plaque'] = df['mdm_post_{}_est'.format(mode)].apply(
                    #     lambda x: 1 if x >= thr else 0)
                    # mdm_post_tn,  mdm_post_fp, mdm_post_fn, mdm_post_tp, mdm_post_accuracy, mdm_post_precision, mdm_post_sensitivity, mdm_post_specificity, mdm_post_f1_score = get_classification_results(
                    #     df['gt_plaque'].to_numpy(),
                    #     df['mdm_post_predicted_plaque'].to_numpy())
                elif mode == 'imt_avg':
                    mdm_tn, mdm_fp, mdm_fn, mdm_tp, mdm_accuracy, mdm_precision, mdm_sensitivity, mdm_specificity, mdm_f1_score = [
                                                                                                                                      None] * 9
                    # mdm_post_tn, mdm_post_fp, mdm_post_fn, mdm_post_tp, mdm_post_accuracy, mdm_post_precision, mdm_post_sensitivity, mdm_post_specificity, mdm_post_f1_score = [None] * 9
                    tn, fp, fn, tp, accuracy, precision, sensitivity, specificity, f1_score = [None] * 9
            elif mode == 'plaque':
                mean_error, error_std, mean_absolute_error, squared_error, pearson_cc = [None] * 5
                df['predicted_plaque_thr'] = df['predicted_{}'.format(mode)].apply(
                    lambda x: 1 if x >= 0.5 else 0)
                tn, fp, fn, tp, accuracy, precision, sensitivity, specificity, f1_score = get_classification_results(
                    df['gt_plaque'].to_numpy(),
                    df['predicted_plaque_thr'].to_numpy())
            else:
                raise NotImplementedError()

            results_df = results_df.append({'model': 'End2End DL',
                                            'mode': mode,
                                            'subset': subset,
                                            'mean_error': mean_error,
                                            'std_error': error_std,
                                            'MAE': mean_absolute_error,
                                            'MSE': squared_error,
                                            'CC': pearson_cc,
                                            'tn': tn,
                                            'fp': fp,
                                            'fn': fn,
                                            'tp': tp,
                                            'accuracy': accuracy,
                                            'precision': precision,
                                            'sensitivity': sensitivity,
                                            'specificity': specificity,
                                            'f1_score': f1_score},
                                           ignore_index=True)
            if mode != 'plaque':
                results_df = results_df.append({'model': 'M.d.M et al. 2020',
                                                'mode': mode,
                                                'subset': subset,
                                                'mean_error': mdm_mean_error,
                                                'std_error': mdm_error_std,
                                                'MAE': mdm_mean_absolute_error,
                                                'MSE': mdm_squared_error,
                                                'CC': mdm_pearson_cc,
                                                'tn': mdm_tn,
                                                'fp': mdm_fp,
                                                'fn': mdm_fn,
                                                'tp': mdm_tp,
                                                'accuracy': mdm_accuracy,
                                                'precision': mdm_precision,
                                                'sensitivity': mdm_sensitivity,
                                                'specificity': mdm_specificity,
                                                'f1_score': mdm_f1_score},
                                               ignore_index=True)

                # results_df = results_df.append({'model': 'M.d.M et al. 2020 only post-processing',
                #                                 'mode': mode,
                #                                 'subset': subset,
                #                                 'mean_error': mdm_post_mean_error,
                #                                 'std_error': mdm_post_error_std,
                #                                 'MAE': mdm_post_mean_absolute_error,
                #                                 'MSE': mdm_post_squared_error,
                #                                 'CC': mdm_post_pearson_cc,
                #                                 'tn': mdm_post_tn,
                #                                 'fp': mdm_post_fp,
                #                                 'fn': mdm_post_fn,
                #                                 'tp': mdm_post_tp,
                #                                 'accuracy': mdm_post_accuracy,
                #                                 'precision': mdm_post_precision,
                #                                 'sensitivity': mdm_post_sensitivity,
                #                                 'specificity': mdm_post_specificity,
                #                                 'f1_score': mdm_post_f1_score},
                #                                ignore_index=True)
    results_df.to_csv(os.path.join(experiment_folder_path, 'results', 'results.csv'))


def get_optimal_thr(y_true, y_pred, experiment_path, debug=False):
    """
    Selects the thr that maximizes the F1-Score. Should be computed with validation data. Only for binary classification.
    :param y_true: Ground truth
    :param y_pred: Predicted values
    :return: Threshold that maximizes the F1-Score
    """
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    # max = np.max(f1_scores)
    max_value = np.max(recall)
    # max_thresh = thresholds[np.argmax(f1_scores)]
    max_thresh = thresholds[np.argmax(recall)]

    if debug:
        print('Max score: {:.4f} with a thr of {:.4f}'.format(max_value, max_thresh))
    with open(os.path.join(experiment_path, 'results', 'optimal_plaque_thr.txt'), 'w') as fp:
        fp.write("%s" % max_thresh)
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for validation data')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(os.path.join(experiment_path, 'results', 'precision-recall curve.png'))
    return max_thresh


def save_input_data(experiment_folder_path, df_train, df_valid, df_test):
    os.makedirs(os.path.join(experiment_folder_path, 'input'), exist_ok=True)
    df_train.to_csv(os.path.join(experiment_folder_path, 'input', 'train_data.csv'))
    df_valid.to_csv(os.path.join(experiment_folder_path, 'input', 'validation_data.csv'))
    df_test.to_csv(os.path.join(experiment_folder_path, 'input', 'test_data.csv'))


def evaluate_performance(dataframe, mode_list, exp_id, experiment_folder_path, debug):
    """
    Evaluates model on train, validation and test data.
    :param exp_id: string representing the experiment
    :param target_columns: name of the columns forming the output
    :param dataframe: dataframe containing information relevant to the experiment
    :return:
    """

    for mode in mode_list:
        prediction_column_name = 'predicted_{}'.format(mode)
        error_column_name = 'predicted_{}_error'.format(mode)
        squared_error_column_name = 'predicted_{}_squared_error'.format(mode)
        dataframe[error_column_name] = dataframe['gt_{}'.format(mode)] - dataframe[prediction_column_name]
        dataframe[squared_error_column_name] = dataframe[error_column_name] * dataframe[error_column_name]
        if mode != 'plaque':
            dataframe['mdm_' + error_column_name] = dataframe['gt_{}'.format(mode)] - dataframe[
                'mdm_{}_est'.format(mode)]
            dataframe['mdm_' + squared_error_column_name] = dataframe['mdm_' + error_column_name] * dataframe[
                'mdm_' + error_column_name]
            # dataframe['mdm_post_' + error_column_name] = dataframe['gt_{}'.format(mode)] - dataframe[
            #     'mdm_post_{}_est'.format(mode)]
            # dataframe['mdm_post_' + squared_error_column_name] = dataframe['mdm_post_' + error_column_name] * dataframe[
            #     'mdm_post_' + error_column_name]

    get_metrics(dataframe, mode_list=mode_list,
                experiment_folder_path=experiment_folder_path, debug=debug)
