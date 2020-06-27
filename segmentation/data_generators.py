#!/usr/bin/env python
# coding: utf-8

import random

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from keras.utils import Sequence


def get_data_augmentation_pipeline():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([sometimes(iaa.Affine(
        scale={"x": (1, 1.3), "y": (1, 1.3)},  # scale images to 80-120% of their size, individually per axis
        # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
        # rotate=(-45, 45),  # rotate by -45 to +45 degrees
        # shear=(-16, 16),  # shear by -16 to +16 degrees
        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
        mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    )),
        iaa.SomeOf((0, 5), [
            iaa.ElasticTransformation(alpha=10, sigma=1),
            iaa.Add(value=(-55, 55), per_channel=True),
            iaa.JpegCompression(compression=random.randint(0, 90)),
            iaa.MotionBlur(k=7),
            iaa.MultiplyHueAndSaturation(mul=random.random() * 2),
            iaa.Grayscale(alpha=random.random()),
            iaa.AllChannelsHistogramEqualization(),
            iaa.PerspectiveTransform(),
            iaa.PiecewiseAffine(),
            iaa.CropAndPad(),
            iaa.Crop(),
            iaa.Fog(),
            iaa.Clouds()
        ], random_order=True)],
        random_order=True)

    seq_det = seq.to_deterministic()
    return seq_det


class DataGenerator(Sequence):

    def __init__(self, images, masks, batch_size=8, dim=(512, 512), n_channels=5, mode='train', val_split=0.2):
        'Initialization'
        self.dim = dim
        self.images = np.array(images, dtype=np.uint8)
        self.masks = np.array(masks)
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_imgs = len(self.images)

        if mode not in ['train', 'valid', 'test']:
            raise SystemError('Mode should be train or valid')

        if mode == 'train':
            self.min_i = 0
            self.max_i = int((self.n_imgs - 1) * (1 - val_split))
            self.augmentation = True
        elif mode == 'valid':
            self.min_i = int((self.n_imgs - 1) * (1 - val_split)) + 1
            self.max_i = self.n_imgs - 1
            self.augmentation = False
        elif mode == 'test':
            self.min_i = 0
            self.max_i = self.n_imgs - 1
            self.augmentation = False
        if self.augmentation:
            self.seq = get_data_augmentation_pipeline()
        print("DataGenerator for {} ready, {} real images. Data augmentation status: {}".format(mode, self.n_imgs,
                                                                                                self.augmentation))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.max_i - self.min_i // self.batch_size

    def __getitem__(self, i):
        'Generate one batch of data'

        # Generate data
        X, y = self.__data_generation()

        return X, y

    def __data_generation(self):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 3))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i in range(self.batch_size):
            index = random.randint(self.min_i, self.max_i)
            image = self.images[index]
            mask = self.masks[index]

            # Store sample
            X[i] = image

            # Store class
            y[i] = mask

        if self.augmentation:
            X, y = self.__data_augmentation(np.array(X, dtype=np.uint8), np.array(y))

        return np.array(X, dtype=np.uint8), np.array(y, dtype=np.uint8)

    def __data_augmentation(self, images, masks):

        images_aug, segmaps_aug = self.seq(images=images, segmentation_maps=np.array(masks, dtype=np.int32))
        return images_aug, segmaps_aug
