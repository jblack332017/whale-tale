from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from prep_data import training_data, test_data
from model import createModel
from trainer import trainModel
from graphics import plotHistory
from labelMaker import chooseLabels

output = sys.argv[1]
training = training_data(sys.argv[2:])
testing = test_data()
train_images = training['train_images']
train_labels = training['train_labels']
test_images = testing['test_images']
test_imagenames = testing['test_names']

classes = np.unique(train_labels)
nClasses = len(classes)
print('Training data shape : ', train_images.shape, train_labels.shape)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# Find the shape of input images and create the variable input_shape
nRows,nCols,nDims = train_images.shape[1:]
train_data = train_images.reshape(train_images.shape[0], nRows, nCols, nDims)
input_shape = (nRows, nCols, nDims)

# Training parameters
epochs = 20;
batch_size = 32;

model, history = trainModel(createModel(input_shape, nClasses), train_data, train_labels, epochs, batch_size);
plotHistory(history);
classes_done = model.predict(test_images, batch_size=batch_size)

chooseLabels(classes_done, classes, test_imagenames, output);

