#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Test the model

Usage:
  test.py <ckpt> <dataset>

Options:
  -h --help     Show this help.
  <dataset>     Dataset folder
"""

import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
import numpy as np
import random
import pickle
import os

import matplotlib.colors as col
from preprocess import random_crop, crop, additive_noise, multiplicative_noise, occlusion
from model import ModelSAR
from data_handler import get_data
from keras.preprocessing.image import ImageDataGenerator
import os

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    startcolor = '#FFFFFF'  # 红色，读者可以自行修改
    midcolor = '#2E8B57'  # 绿色，读者可以自行修改
    endcolor = '#696969'  # 蓝色
    cmap2 = col.LinearSegmentedColormap.from_list('own2', [startcolor,  endcolor])

    plt.imshow(cm, interpolation="nearest", cmap=cmap2)
    #plt.title(title, fontdict={'family': 'Times New Roman', 'size': 12})
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontproperties = 'Times New Roman', size = 10)
    plt.yticks(tick_marks, classes, fontproperties = 'Times New Roman', size = 10)
    plt.xlim(-0.5, 9.5)
    plt.ylim(-0.5, 9.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="black" if cm[i, j] > thresh else "black", fontproperties = 'Times New Roman', size = 12)

    plt.tight_layout()
    plt.ylabel('True label', fontproperties = 'Times New Roman', size = 12)
    plt.xlabel('Predicted label', fontproperties = 'Times New Roman', size = 12)


def test():
    """
        Train the model
        **input: **
            *dataset: (String) Dataset folder to used
            *ckpt: (String) [Optional] Path to the ckpt file to restore
    """
    ckpt='./outputs/checkpoints/SAR'
    validation_data_dir = './data/Test'
    img_width, img_height = 100, 100

    # Load name of id

    classname = ['2S1', 'BMP-2', 'BRDM-2', 'BTR-70', 'BTR-60', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU234']
    # Get Test dataset

    model = ModelSAR("SAR", output_folder=None)
    # Load the model
    model.load(ckpt)

    for i in range(1):
        test_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=0)
        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_height, img_width),
            batch_size=2425,
            class_mode='sparse')
        filenames = validation_generator.class_indices
        print(filenames)
        X_test, y_test = validation_generator.next()
        X_test = crop(X_test)
        #X_test = blur(X_test, pow(10, 1 + i * 0.2))
        #X_test = oclusion(X_test, 0.06*(1 + i ))
        #X_test = noise(X_test,0.01*(i+1))
        # Evaluate all the dataset
        loss, acc, predicted_class = model.evaluate_dataset(np.expand_dims(X_test[:, :, :, 1], axis=3), y_test)
        print("Accuracy = ", acc)
        print("Loss = ", loss)

        # Get the confusion matrix
        cnf_matrix = confusion_matrix(y_test, predicted_class)
        # Plot the confusion matrix
        plt.figure(figsize=(5.2, 5))
        plot_confusion_matrix(cnf_matrix, classes=classname, title='Confusion matrix, without normalization')
        plt.show()


if __name__ == '__main__':

    test()
