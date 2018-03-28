from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
from keras.utils import to_categorical

print(type(train_images))

print(type(train_labels))

print(train_labels)

print(train_images)
