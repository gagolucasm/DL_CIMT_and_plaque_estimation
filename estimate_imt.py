#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from skimage.measure import label

import config
import helpers


def get_column_width(mask, thr, initial_range_columns, final_range_columns):
    """
    
    :param mask: segmentation result of the IM area
    :param thr: threshold for binarization of the segmentation
    :param initial_range_columns: #TODO: Check if this parameter is correct
    :param final_range_columns: 
    :return: #TODO: Define output
    """
    bin_mask = mask > thr
    result = last_nonzero(bin_mask) - first_nonzero(bin_mask)
    result = np.trim_zeros(result)
    if final_range_columns > 0:
        # Maintain only values between 0.4 and 1.5 mm, as stated in https://doi.org/10.1016/j.artmed.2019.101784
        result[:-final_range_columns] = np.clip(result[:-final_range_columns],
                                                int(0.4 * config.RESOLUTION),
                                                int(1.5 * config.RESOLUTION))
    return result


def last_nonzero(mask):
    """
    For each column get last value that is not zero
    :param mask: segmentation result of the IM area
    :return: list of position of last non-zero value if any, else -1
    """
    val = mask.shape[0] - np.flip(mask, axis=0).argmax(axis=0) - 1
    return np.where(mask.any(axis=0), val, -1)


def first_nonzero(mask):
    """
    For each column, get first value that is not zero
    :param mask: segmentation result of the IM area
    :return: list of position of first non-zero value if any, else -1
    """
    return np.where(mask.any(axis=0), mask.argmax(axis=0), -1)


def getLargestCC(segmentation):
    """
    
    :param segmentation: segmentation result of the IM area
    :return: label of largest connected component
    """
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def calculate_imt(mask_path, thr=config.IMT_THRESHOLD, database=config.DATABASE):
    """
    Calculates IMT from the path of a segmented image
    :param mask_path: path of the segmented image
    :param thr: threshold for binarization of the segmentation
    :param database: name of the database, can be CCA or BULB
    :return: mean and max IMT values
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.
    _, mask = cv2.threshold(mask, thr, 1, cv2.THRESH_BINARY)

    h, w = mask.shape
    mask = getLargestCC(mask)
    # TODO: comment more this section
    if database == 'CCA':
        start_imt = np.argmax(first_nonzero(mask > thr) > 1)
        mask = mask[:, int(start_imt):int(start_imt + 10 * config.RESOLUTION)]
        initial_range_columns = 0
        if int(start_imt + 10 * config.RESOLUTION) > int(w - (3 * config.RESOLUTION)):
            final_range_columns = int(start_imt + 10 * config.RESOLUTION) - int(w - (3 * config.RESOLUTION))
        else:
            final_range_columns = 0
    elif database == 'BULB':
        initial_range_columns = int(3 * config.RESOLUTION)
        final_range_columns = int(w - 3 * config.RESOLUTION)

    imt_per_column_pixels = get_column_width(mask, thr, initial_range_columns,
                                             final_range_columns) / config.RESOLUTION
    if np.sum(imt_per_column_pixels) == 0:
        return 999, 999  # error code

    return np.mean(imt_per_column_pixels), np.max(imt_per_column_pixels)


def explore_result(df, key, thr=config.IMT_THRESHOLD, predict_imt=True):
    """
    Plots an image with its predicted mask and IMT values
    :param df: pandas dataframe with paths to images of interest for the experiment
    :param key: unique identifier for each image
    :param thr: threshold for imt estimation from predicted mask
    :param predict_imt: boolean indicating whether to predict the IMT
    """
    mask_path = df.at[key, 'mask_path']
    mask = cv2.imread(mask_path)
    start_imt = np.argmax(first_nonzero(mask > thr) > 1)
    print('Image id: {}'.format(key))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.imread(df.at[key, 'complete_path']))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    print(mask.shape)
    # TODO: make plot dependant on database
    plt.axvline(x=start_imt, color='k', linestyle='--')
    plt.axvline(x=int(start_imt + 10 * config.RESOLUTION), color='r')
    plt.axvline(x=int(mask.shape[1] - (3 * config.RESOLUTION)), color='b')
    plt.title('Predicted mask and ROI for id: {}'.format(key))
    plt.show()
    if predict_imt:
        print('Predicted IMT: {:.4f} max {:.4f} avg'.format(df.at[key, 'predicted_imt_max'],
                                                            df.at[key, 'predicted_imt_avg']))
        print('GT IMT:        {:.4f} max {:.4f} avg\n'.format(df.at[key, 'gt_imt_max'], df.at[key, 'gt_imt_avg']))


if __name__ == '__main__':

    # Load data from disk
    data = np.load('segmentation/complete_data_{}.npy'.format(config.DATABASE))
    data = data[()]['data']

    # Convert to dataframe and filter invalid values
    df = pd.DataFrame.from_dict(data, orient='index')
    df = helpers.filter_dataframe(df, config.DATABASE)

    # Change index format #TODO: fix in previous step
    df.index = df.index.map(lambda x: x[4:-1])

    if config.COMPARE_RESULTS:
        # Add columns with results from https://doi.org/10.1016/j.artmed.2019.101784
        df = helpers.add_previous_results(dataframe=df, database=config.DATABASE)

    # Calculate IMT from mask
    print('Calculating CIMT from segmentation results, this could take a while')
    df['predicted_imt_tuple'] = df['mask_path'].apply(lambda x: calculate_imt(x))
    df['predicted_imt_avg'] = df['predicted_imt_tuple'].apply(lambda x: x[0])
    df['predicted_imt_max'] = df['predicted_imt_tuple'].apply(lambda x: x[1])
    df = df.drop(columns=['predicted_imt_tuple'])

    # Calculate errors
    df['imt_error_max'] = df['gt_imt_max'] - df['predicted_imt_max']
    df['imt_error_avg'] = df['gt_imt_avg'] - df['predicted_imt_avg']
    df['gt_plaque'] = df['gt_imt_max'].apply(lambda x: 1 if x > 1.5 else 0)

    ax = df['predicted_imt_max'].hist()
    plt.title('Predicted imt max histogram')
    plt.show()

    df['imt_squared_error_max'] = np.square(df['imt_error_max'])
    df['imt_squared_error_avg'] = np.square(df['imt_error_avg'])
    print('Mean of max error:         {}'.format(df['imt_error_max'].mean()))
    print('Mean of max squared error: {}'.format(df['imt_squared_error_max'].mean()))

    df['imt_error_max'].hist()
    print('Mean of avg error: {}'.format(df['imt_error_avg'].mean()))
    print('Mean of avg squared error: {}'.format(df['imt_squared_error_avg'].mean()))

    df['imt_error_avg'].hist()
    df_results = df.sort_values(by=['imt_error_avg'])

    for key in df_results.index[:3]:
        explore_result(df_results, key)
    f, ax = plt.subplots(1, figsize=(8, 5))
    sm.graphics.mean_diff_plot(df_results['predicted_imt_max'].to_numpy(),
                               df_results['gt_imt_max'].to_numpy(), ax=ax)
    plt.title('Error max')
    plt.show()

    f, ax = plt.subplots(1, figsize=(8, 5))
    sm.graphics.mean_diff_plot(df_results['predicted_imt_avg'].to_numpy(),
                               df_results['gt_imt_avg'].to_numpy(), ax=ax)
    plt.title('Error avg')
    plt.show()

    df_results['gt_imt_max'].corr(df_results['predicted_imt_max'])

    df_results['gt_imt_avg'].corr(df_results['predicted_imt_avg'])

    ax = df_results.plot.scatter(x='gt_imt_max', y='predicted_imt_max')
    plt.title('Scatter plot IMT max')
    plt.show()

    ax = df_results.plot.scatter(x='gt_imt_avg', y='predicted_imt_avg')
    plt.title('Scatter plot IMT average')
    plt.show()

    df_results['predicted_plaque'] = df_results['predicted_imt_max'].apply(lambda x: 1 if x > 1.5 else 0)
    df_results['gt_plaque'].value_counts()
    df_results['predicted_plaque'].value_counts()

    helpers.get_classification_results(df_results['gt_plaque'].to_numpy(),
                                       df_results['predicted_plaque'].to_numpy())
