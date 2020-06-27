#!/usr/bin/env python
# coding: utf-8

import os
import random
import subprocess

import cv2
import keras
import keras.backend as K
import numpy as np
import pandas as pd
import segmentation_models as sm
import tensorflow as tf
from matplotlib import pyplot as plt
from natsort import natsorted

from segmentation import config
from segmentation.data_generators import DataGenerator, get_data_augmentation_pipeline


def load_images_from_directory(file_paths, ext, base_path):
    """
    Loads and resize all the images in a list of paths with a given extension
    :param file_paths: list of paths containing images
    :param ext: image file extension to look for
    :param base_path: base path to append to file_paths
    :return: a list of loaded and resized images
    """
    return [cv2.resize(cv2.imread(base_path + img_path + ext), config.INPUT_SHAPE) for img_path in file_paths]


def preprocess_masks(masks, binarize=False):
    """
    Preprocess masks before training
    :param masks: numpy array with the masks
    :param binarize: boolean indicating if only one class will be used for training
    :return:
    """
    masks = np.array(masks)
    if binarize:
        # Make it binary
        masks[masks != 4] = 0
        masks[masks == 4] = 1
    if config.DATABASE == 'BULB':
        # classes start at 1, this generates an extra dim when using to categorical
        masks = [cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) - 1 for mask in masks]
    else:
        masks = [cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) for mask in masks]

    return keras.utils.to_categorical(masks)


def prepare_mask_for_plotting(img):
    """
    Auxiliary function to convert a mask into something more visually appealing
    :param img: original mask
    :return: colored mask
    """
    if img.shape[2] > 1:
        img = np.argmax(img, axis=2)
    else:
        img = img.squeeze()
    img = np.stack((img,) * 3, axis=-1)
    if config.DATABASE == 'BULB':
        img = img + 1
    color_dict = {(0, 0, 0): [0, 0, 255],  # NOT USED
                  (1, 1, 1): [0, 128, 255],  # Blue, Near Wall
                  (2, 2, 2): [0, 255, 255],  # Light Blue, Lumen
                  (3, 3, 3): [128, 255, 128],  # Light Green, IMT LEFT doubtful
                  (4, 4, 4): [255, 255, 0],  # Yellow, IMT reliable
                  (5, 5, 5): [255, 128, 0],  # Orange, IMT RIGHT doubtful
                  (6, 6, 6): [128, 0, 255]}  # Purple, Far Wall
    # light blue: CCA lumen, green: Bulb IMR, yellow: CCA IMR)
    for key in color_dict:
        img[np.where((img == key).all(axis=2))] = color_dict[key]
    return img


def plot_img_and_mask(images, masks, index=None, result=None, paths=None):
    """
    Plots images alongside with respective masks
    :param images: numpy array of images
    :param masks: numpy array of masks
    :param index: if not None, plots an specific index
    :param result: if not None, it should be the prediction for the selected images and masks
    :param paths: if not None, a path indicating the origin of the image used for the title of the plot
    """
    if index is None:
        index = random.randint(0, len(images) - 1)
    img = images[index]
    mask = masks[index]
    if paths is not None:
        path = paths[index]

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(img, interpolation='none')
    if paths is not None:
        plt.title(str(index) + ' / ' + path)
    plt.subplot(1, 3, 2)
    plt.imshow(prepare_mask_for_plotting(mask), interpolation='none')
    plt.subplot(1, 3, 3)
    if result is None:
        plt.imshow(img, interpolation='none')
        plt.imshow(prepare_mask_for_plotting(mask), 'jet', interpolation='none', alpha=0.5)
    else:
        plt.imshow(prepare_mask_for_plotting(result), interpolation='none')

    # plt.contour(plot_mask(mask), [0.5], linewidths=1.2, colors='y')

    plt.show()


def dice_coef(y_true, y_pred, smooth=1.0):
    """
    Performs the dice coefficient metric between two tensors
    :param y_true: gt tensor
    :param y_pred: predicted tensor
    :param smooth: smooth parameter
    :return: dice coefficient
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def generate_mosaic(images, paths, masks, results, n_items=None):
    """
    Generates a mosaic to easily compare original images, ground truth and predictions
    :param images: numpy array containing original images
    :param paths: list of paths to original images
    :param masks: numpy array containing gt masks
    :param results: numpy array containing predicted masks
    :param n_items: optional, if set can limit the number of elements in the mosaic
    """
    if n_items is not None:
        n_items = min(n_items, len(images))
    else:
        n_items = len(images)

    fig, ax = plt.subplots(n_items, 3, figsize=(20, 5 * n_items))

    for index in range(0, n_items):
        img = images[index]
        mask = masks[index]
        result = results[index]
        plt.subplot(n_items, 3, 3 * index + 1)
        plt.imshow(img, interpolation='none')
        plt.title(paths[index])
        plt.subplot(n_items, 3, 3 * index + 2)
        plt.imshow(prepare_mask_for_plotting(mask), interpolation='none')
        plt.title('Ground truth')
        plt.subplot(n_items, 3, 3 * index + 3)
        plt.imshow(prepare_mask_for_plotting(result), interpolation='none')
        plt.title('Prediction. Dice: {:.2f}, IoU: {:.2f}'.format(K.get_value(dice_coef(mask, result)),
                                                                 K.get_value(iou_metric(mask, result))))

        # plt.contour(plot_mask(mask), [0.5], linewidths=1.2, colors='y')
    plt.savefig('results_{}_unet_efficientnet.png'.format(config.DATABASE))
    plt.show()


def get_total_IMT_value(data, database=config.DATABASE):
    """
    Calculates the total pixels labeled as intima media in a given set of masks
    :param data: numpy array with masks
    :param database: database of origin of the masks, can be BULB or CCA
    :return: total IM value of all masks
    """
    total_sum = 0
    if database == 'BULB':
        for element in data:
            _, element = cv2.threshold(element, 0.5, 1, cv2.THRESH_BINARY)
            total_sum += np.sum(element[:, :, 4] + element[:, :, 3] + element[:, :, 2])
    if database == 'CCA':
        for element in data:
            _, element = cv2.threshold(element, 0.5, 1, cv2.THRESH_BINARY)
            total_sum += np.sum(element[:, :, 4])
    return total_sum


if __name__ == '__main__':

    if config.DATABASE == 'BULB':
        X_train_path = '../datasets/BULB/TRAINTEST/ORIGINALS/ORI_BULBtrain/'
        X_test_path = '../datasets/BULB/TRAINTEST/ORIGINALS/ORI_BULBtest/'
        y_train_path = '../datasets/BULB/TRAINTEST/GT/GT_BULBtrain/'
        y_test_path = '../datasets/BULB/TRAINTEST/GT/GT_BULBtest/'
    elif config.DATABASE == 'CCA':
        X_train_path = '../datasets/CCA/TRAINTEST/ORIGINALS/TRAIN_jpg/'
        X_test_path = '../datasets/CCA/TRAINTEST/ORIGINALS/TEST_jpg/'
        y_train_path = '../datasets/CCA/TRAINTEST/GT/train_gt/'
        y_test_path = '../datasets/CCA/TRAINTEST/GT/test18_gt/'
    else:
        raise Exception('Unrecognized database {}'.format(config.DATABASE))

    X_train_paths = natsorted(os.listdir(X_train_path))
    X_test_paths = natsorted(os.listdir(X_test_path))
    y_train_paths = natsorted(os.listdir(y_train_path))
    y_test_paths = natsorted(os.listdir(y_test_path))

    # Clean paths to keep elements present both in image and in mask path
    train_paths = natsorted(list(set([os.path.splitext(x)[0] for x in X_train_paths]).intersection(
        [os.path.splitext(x)[0] for x in y_train_paths])))
    test_paths = natsorted(list(set([os.path.splitext(x)[0] for x in X_test_paths]).intersection(
        [os.path.splitext(x)[0] for x in y_test_paths])))

    if config.DATABASE == 'CCA':
        X_train = load_images_from_directory(train_paths, '.jpg', X_train_path)
        X_test = load_images_from_directory(test_paths, '.jpg', X_test_path)
    elif config.DATABASE == 'BULB':
        X_train = load_images_from_directory(train_paths, '.png', X_train_path)
        X_test = load_images_from_directory(test_paths, '.png', X_test_path)

    y_train = load_images_from_directory(train_paths, '.png', y_train_path)
    y_test = load_images_from_directory(test_paths, '.png', y_test_path)

    assert len(X_train) == len(y_train), 'The length of x and y train should be equal'
    assert len(X_test) == len(y_test), 'The length of x and y test should be equal'

    y_train = preprocess_masks(y_train, binarize=config.PREDICT_ONLY_IM)
    y_test = preprocess_masks(y_test, binarize=config.PREDICT_ONLY_IM)
    n_channels = y_train.shape[3]

    plot_img_and_mask(X_test, y_test, paths=test_paths)

    data_augmentation_pipeline = get_data_augmentation_pipeline()
    images_aug, segmaps_aug = data_augmentation_pipeline(images=np.array(X_train),
                                                         segmentation_maps=np.array(y_train, dtype=np.int32))

    plot_img_and_mask(images_aug, segmaps_aug)

    train_gen = DataGenerator(X_train, y_train, mode='train', batch_size=config.BATCH_SIZE)
    valid_gen = DataGenerator(X_train, y_train, mode='valid', batch_size=config.BATCH_SIZE)
    test_gen = DataGenerator(X_test, y_test, mode='test', batch_size=config.BATCH_SIZE)

    # Check if GPU is available
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    CLASSES = list(range(n_channels))
    preprocess_input = sm.get_preprocessing(config.BACKBONE)

    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES))  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    # create model
    model = sm.Unet(config.BACKBONE, classes=n_classes, activation=activation)

    os.makedirs('weights')
    best_weights_path = 'weights/best_model_{}_unet_ef0_weights.h5'.format(config.DATABASE)

    if config.LOAD_PRETRAINED_MODEL:
        model.load_weights(best_weights_path)

    total_loss = sm.losses.binary_focal_dice_loss

    optimizer = keras.optimizers.Adam()

    iou_metric = sm.metrics.IOUScore(threshold=0.5)
    f_score_metric = sm.metrics.FScore(threshold=0.5)
    metrics = [iou_metric,
               f_score_metric,
               dice_coef,
               'accuracy',
               tf.keras.metrics.Recall()]

    model.compile(optimizer, total_loss, metrics)

    callbacks = [
        keras.callbacks.ModelCheckpoint(best_weights_path, save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
        keras.callbacks.EarlyStopping(patience=20, monitor='val_loss')
    ]

    # Temporal fix due to tensorflow bug. See https://github.com/tensorflow/tensorflow/issues/24828
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    if not config.LOAD_PRETRAINED_MODEL:
        subprocess.Popen(
            "timeout 120 nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1 | sed s/%//g > GPU-stats.log",
            shell=True)

        history = model.fit(train_gen,
                            validation_data=valid_gen,
                            callbacks=callbacks,
                            **config.TRAINING_PARAMETERS
                            )

    if not config.LOAD_PRETRAINED_MODEL:
        # Plot training & validation iou_score values
        plt.figure(figsize=(30, 5))
        plt.subplot(131)
        plt.plot(history.history['iou_score'])
        plt.plot(history.history['val_iou_score'])
        plt.title('Model iou_score')
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(132)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Focal loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(133)
        plt.plot(history.history['dice_coef'])
        plt.plot(history.history['val_dice_coef'])
        plt.title('Model Dice coefficient')
        plt.ylabel('Dice_coef')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    if not config.LOAD_PRETRAINED_MODEL:
        gpu = pd.read_csv("GPU-stats.log")
        gpu.plot(figsize=(15, 7))

    model.load_weights(best_weights_path)

    # Evaluate results
    print('Training metrics')
    print(model.metrics_names)
    print(model.evaluate(np.array(X_train), y_train))

    print('Testing metrics')
    print(model.metrics_names)
    print(model.evaluate(np.array(X_test), y_test))

    predicted_masks_train = model.predict(np.array(X_train))
    generate_mosaic(X_train, X_train_paths, y_train, predicted_masks_train, n_items=10)

    predicted_masks_test = model.predict(np.array(X_test))
    generate_mosaic(X_test, test_paths, y_test, predicted_masks_test)

    print('Mean difference in imt pixels in train: {}'.format((get_total_IMT_value(y_train) -
                                                               get_total_IMT_value(predicted_masks_train)) / len(
        y_train)))
    print('Mean difference in imt pixels in test:  {}'.format((get_total_IMT_value(y_test) -
                                                               get_total_IMT_value(predicted_masks_test)) / len(
        y_test)))
