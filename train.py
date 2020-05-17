#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""

"""
import os
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageEnhance
import tensorflow as tf
import numpy as np
import random
from preprocess import random_crop, crop
from model import ModelSAR


import time


def train():
    """
        Train the model
        **input: **
            *dataset: (String) Dataset folder to used
            *ckpt: (String) [Optional] Path to the ckpt file to restore
            *output: (String) [Optional] Path to the output folder to used. ./outputs/ by default
    """
    ckpt = './outputs/checkpoints/SAR'
    output_folder = None

    train_data_dir = './data/Train'
    validation_data_dir = './data/Train'
    img_width, img_height = 100, 100
    NB_LABELS = 10
    nb_train_samples = 2747
    batch_size = 16
    lr = 0.0002

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=True)

    valid_generator = valid_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=nb_train_samples,
        class_mode='sparse',
        shuffle=True)

    # Init model
    model = ModelSAR("SAR", output_folder, NB_LABELS)
    if ckpt is not None:
        model.init()
    else:
        model.load(ckpt)

    # Training pipeline
    for batch_id in range(40000):
        x_batch, y_batch = train_generator.next()
        x_batch = random_crop(x_batch)

        ### Training
        cost, acc = model.optimize(np.expand_dims(x_batch[:,:,:,1],axis=3), y_batch, lr)
        print("Batch ID = %s, loss = %s, acc = %s" % (batch_id, cost, acc))

        ### Validation
        if batch_id % 20 == 0: # Plot the last results
            X_valid, y_valid = valid_generator.next()
            X_valid = crop(X_valid)
            print("Evaluate full validation dataset ...")
            loss, acc, _ = model.evaluate_dataset(np.expand_dims(X_valid[:,:,:,1],axis=3), y_valid)
            print("Current loss: %s Current acc: %s" % (loss, acc))
            model.save()

if __name__ == '__main__':
    train()
