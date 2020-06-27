#!/usr/bin/env python
# coding: utf-8

import os
import sys

import cv2
import numpy as np
# TODO: reformat file
import pandas as pd
import segmentation_models as sm
import tensorflow as tf
import tqdm

from segmentation import config

tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# ## Load Model

# TODO: convert into function
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

preprocess_input = sm.get_preprocessing(config.BACKBONE)

n_classes = 1 if config.PREDICT_ONLY_IM else 6
activation = 'sigmoid' if config.PREDICT_ONLY_IM else 'softmax'

# create model
model = sm.Unet(config.BACKBONE, classes=n_classes, activation=activation)

best_weights_path = 'best_model_{}_unet_ef0_weights.h5'.format(config.DATABASE)
model.load_weights(best_weights_path)

# ## Evaluation of IMT
# ---

# TODO: duplicated code, convert into function
if config.DATABASE == 'CCA':
    """
    1-"EstudiDon": Is the subject identification (to find the corresponding image)
    2-"imtm_lcca_s”: Is the maximum value from IMT in left side of the CA
    3-"imtm_rcca_s": Is the maximum value from IMT in right side of the CA
    4-"imta_lcca_s": Is the mean value from IMT in left side of the CA
    5- "imta_rcca_s": Is the mean value from IMT in right side of the CA
    """
    imts_regicor = pd.read_csv('datasets/CCA/REGICOR_4000/imt_data.csv', header=None)
    base_regicor_img_path = 'datasets/CCA/REGICOR_4000/CCA2_png'

elif config.DATABASE == 'BULB':
    """
    1-"estudiparti": Is the subject identification (to find the corresponding image)
    2-"imtm_lbul_s”: Is the maximum value from IMT in left side of the CA
    3-"imtm_rbul_s": Is the maximum value from IMT in right side of the CA
    4-"imta_lbul_s": Is the mean value from IMT in left side of the CA
    5- "imta_rbul_s": Is the mean value from IMT in right side of the CA
    """
    imts_regicor = pd.read_csv('datasets/BULB/REGICOR_3000/imt_dataBULB.csv', header=None)
    base_regicor_img_path = 'datasets/BULB/REGICOR_3000/BULB2_png'

else:
    raise Exception('Database not recognized')

imts_regicor.columns = ['EstudiDon', 'imtm_lcca_s', 'imtm_rcca_s', 'imta_lcca_s', 'imta_rcca_s']

regicor_imgs_path = os.listdir(base_regicor_img_path)
prediction_folder = os.path.join('segmentation_results', config.DATABASE)
os.makedirs(prediction_folder, exist_ok=True)


def predict_all_images(imts_regicor):
    data = {}
    for index, row in tqdm.tqdm(imts_regicor.iterrows()):
        images_paths = [i for i in regicor_imgs_path if str(int(row['EstudiDon'])) in i]

        # Right side
        image_right_path = [i for i in images_paths if 'r{}g'.format(config.DATABASE.lower()[:3]) in i.lower()]
        if len(image_right_path) > 0:
            complete_right_path = os.path.join(base_regicor_img_path, image_right_path[0])
            image = cv2.imread(complete_right_path)
            image = cv2.resize(image, config.INPUT_SHAPE)
            prediction = np.squeeze(model.predict(np.expand_dims(image, axis=0)))
            if config.DATABASE == 'BULB':
                prediction = prediction[:, :, 4] + prediction[:, :, 3] + prediction[:, :, 2]
            else:
                prediction = prediction[:, :, 4]

            prediction = cv2.resize(prediction, (445, 470))  # Original shape
            prediction = prediction * 255.

            prediction_path = os.path.join(prediction_folder, image_right_path[0])
            prediction_path = prediction_path.replace('jpg', 'png')
            cv2.imwrite(prediction_path, prediction)

            data['img:' + image_right_path[0][:-4]] = {'complete_path': complete_right_path,
                                                       'mask_path': prediction_path,
                                                       'gt_imt_max': row['imtm_rcca_s'],
                                                       'gt_imt_avg': row['imta_rcca_s'], 'side': 'right'}

        # Left side
        image_left_path = [i for i in images_paths if 'l{}g'.format(config.DATABASE.lower()[:3]) in i.lower()]
        if len(image_left_path) > 0:
            complete_left_path = os.path.join(base_regicor_img_path, image_left_path[0])
            image = cv2.imread(complete_left_path)
            image = cv2.resize(image, config.INPUT_SHAPE)
            prediction = np.squeeze(model.predict(np.expand_dims(image, axis=0)))
            if config.DATABASE == 'BULB':
                prediction = prediction[:, :, 4] + prediction[:, :, 3] + prediction[:, :, 2]
            else:
                prediction = prediction[:, :, 4]
            prediction = cv2.resize(prediction, (445, 470))  # Original shape
            prediction_path = os.path.join(prediction_folder, image_left_path[0])
            prediction_path = prediction_path.replace('jpg', 'png')
            prediction = prediction * 255.

            cv2.imwrite(prediction_path, prediction)

            data['img:' + image_left_path[0][:-4]] = {'complete_path': complete_left_path,
                                                      'mask_path': prediction_path,
                                                      'gt_imt_max': row['imtm_lcca_s'],
                                                      'gt_imt_avg': row['imta_lcca_s'], 'side': 'left'}

    return data


print('[INFO] Predicting all images from {}'.format(config.DATABASE))
results_data = predict_all_images(imts_regicor)
print('[INFO] Prediction done')
filename = 'complete_data_{}.npy'.format(config.DATABASE)
print('[INFO] Saving results to disk as {}'.format(filename))
np_data = {'data': results_data}
np.save(filename, np_data)
print('[INFO] Saving done, checking integrity')
data2 = np.load(filename)
data3 = data2[()]['data']
assert sys.getsizeof(results_data) == sys.getsizeof(data3), 'Files do not match'
print('[INFO] Done, everything OK. Exiting')
