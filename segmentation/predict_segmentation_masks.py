#!/usr/bin/env python
# coding: utf-8

import os

import cv2
import numpy as np
import pandas as pd
import segmentation_models as sm
import tqdm

from segmentation import config, helpers


def get_images_from_id(image_id: str, regicor_imgs_path):
    possible_cases = [image_id + '156' + '_',
                      image_id + '156' + '_',
                      image_id + '84' + '_',
                      image_id + '60' + '_',
                      image_id + '705' + '_',
                      image_id + '060' + '_']
    images_paths = []
    for path in regicor_imgs_path:
        for case in possible_cases:
            if case in path:
                images_paths.append(path)
    images_paths = set(images_paths)
    if len(images_paths) > 2:
        print('Incorrect number of paths ({}) retrieved from id {}'.format(len(images_paths), image_id))

    # assert len(images_paths) < 3, 'Incorrect number of paths ({}) retrieved from id {}'.format(len(images_paths),image_id)
    return images_paths


def predict_all_images(base_regicor_img_path, regicor_imgs_path, prediction_folder):
    df = pd.DataFrame(columns=['img_id', 'complete_path', 'mask_path'])
    for image_path in tqdm.tqdm(regicor_imgs_path):
        complete_path = os.path.join(base_regicor_img_path, image_path)
        image = cv2.imread(complete_path)
        image = cv2.resize(image, config.INPUT_SHAPE)
        prediction = np.squeeze(model.predict(np.expand_dims(image, axis=0)))
        if config.DATABASE == 'BULB' and config.SINGLE_IMT_CLASS_BULB:
            prediction = prediction[:, :, 4] + prediction[:, :, 3] + prediction[:, :, 2]
        else:
            prediction = prediction[:, :, 4]

        prediction = cv2.resize(prediction, (445, 470))  # Original shape
        prediction = prediction * 255.

        prediction_path = os.path.join(prediction_folder, image_path)
        prediction_path = prediction_path.replace('jpg', 'png')
        cv2.imwrite(prediction_path, prediction)
        df = df.append({'img_id': image_path[:-4],
                        'complete_path': complete_path,
                        'mask_path': os.path.join('segmentation', prediction_path)},
                       ignore_index=True)
    return df


def predict_all_images_old(dataframe, regicor_imgs_path):
    data = {}
    for index, row in tqdm.tqdm(dataframe.iterrows()):
        # images_paths = [i for i in regicor_imgs_path if str(int(row['EstudiDon'])) == i[1:6]]
        images_paths = get_images_from_id(image_id=str(int(row['EstudiDon'])),
                                          regicor_imgs_path=regicor_imgs_path)
        # Right side
        image_right_path = [i for i in images_paths if 'r{}g'.format(config.DATABASE.lower()[:3]) in i.lower()]
        if len(image_right_path) > 0:
            complete_right_path = os.path.join(base_regicor_img_path, image_right_path[0])
            image = cv2.imread(complete_right_path)
            image = cv2.resize(image, config.INPUT_SHAPE)
            prediction = np.squeeze(model.predict(np.expand_dims(image, axis=0)))
            if config.DATABASE == 'BULB' and not config.SINGLE_IMT_CLASS_BULB:
                prediction = prediction[:, :, 4] + prediction[:, :, 3] + prediction[:, :, 2]
            else:
                prediction = prediction[:, :, 4]

            prediction = cv2.resize(prediction, (445, 470))  # Original shape
            prediction = prediction * 255.

            prediction_path = os.path.join(prediction_folder, image_right_path[0])
            prediction_path = prediction_path.replace('jpg', 'png')
            cv2.imwrite(prediction_path, prediction)

            data['img:' + image_right_path[0][:-4]] = {'complete_path': complete_right_path,
                                                       'mask_path': os.path.join('segmentation', prediction_path),
                                                       'gt_imt_max': row['imtm_rcca_s'],
                                                       'gt_imt_avg': row['imta_rcca_s'], 'side': 'right'}

        # Left side
        image_left_path = [i for i in images_paths if 'l{}g'.format(config.DATABASE.lower()[:3]) in i.lower()]
        if len(image_left_path) > 0:
            complete_left_path = os.path.join(base_regicor_img_path, image_left_path[0])
            image = cv2.imread(complete_left_path)
            image = cv2.resize(image, config.INPUT_SHAPE)
            prediction = np.squeeze(model.predict(np.expand_dims(image, axis=0)))
            if config.DATABASE == 'BULB' and not config.SINGLE_IMT_CLASS_BULB:
                prediction = prediction[:, :, 4] + prediction[:, :, 3] + prediction[:, :, 2]
            else:
                prediction = prediction[:, :, 4]
            prediction = cv2.resize(prediction, (445, 470))  # Original shape
            prediction_path = os.path.join(prediction_folder, image_left_path[0])
            prediction_path = prediction_path.replace('jpg', 'png')
            prediction = prediction * 255.

            cv2.imwrite(prediction_path, prediction)

            data['img:' + image_left_path[0][:-4]] = {'complete_path': complete_left_path,
                                                      'mask_path': os.path.join('segmentation', prediction_path),
                                                      'gt_imt_max': row['imtm_lcca_s'],
                                                      'gt_imt_avg': row['imta_lcca_s'], 'side': 'left'}

    return data


if __name__ == '__main__':

    preprocess_input = sm.get_preprocessing(config.BACKBONE)

    if config.PREDICT_ONLY_IM:
        n_classes = 1
    elif config.DATABASE == 'CCA':
        n_classes = 6
    elif config.DATABASE == 'BULB':
        n_classes = 6
    else:
        raise Exception('Invalid database')

    activation = 'sigmoid' if config.PREDICT_ONLY_IM else 'softmax'

    # create model
    model = sm.Unet(config.BACKBONE, classes=n_classes, activation=activation, input_shape=(512, 512, 3))

    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='TB')

    best_weights_path = 'weights/best_model_{}_unet_ef0_weights.h5'.format(config.DATABASE)
    model.load_weights(best_weights_path)

    # ## Evaluation of IMT

    imts_regicor, base_regicor_img_path = helpers.get_regicor_data(config.DATABASE)

    regicor_imgs_path = os.listdir(base_regicor_img_path)
    prediction_folder = os.path.join('segmentation_results', config.DATABASE)
    os.makedirs(prediction_folder, exist_ok=True)

    print('[INFO] Predicting all images from {}'.format(config.DATABASE))
    df = predict_all_images(base_regicor_img_path, regicor_imgs_path, prediction_folder)
    print('[INFO] Prediction done')
    filename = 'complete_data_{}.csv'.format(config.DATABASE)
    print('[INFO] Saving results to disk as {}'.format(filename))
    df.to_csv(filename, index=False)
    print('[INFO] Done, everything OK. Exiting')
