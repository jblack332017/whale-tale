from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from prep_data import training_data, test_data
from model import createModel
import residualmodel
from resnet import ResnetBuilder
from trainer import trainModel
from graphics import plotHistory
from labelMaker import chooseLabels

output = sys.argv[1]
epochs = int(sys.argv[2])
training = training_data(sys.argv[4:])
testing = test_data(sys.argv[3])
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
input_shape = (nDims, nRows, nCols)

# Training parameters
batch_size = 32;

model = ResnetBuilder.build_resnet_34(input_shape, nClasses)
model, history = trainModel(model, train_data, train_labels, epochs, batch_size);
plotHistory(history);
classes_done = model.predict(test_images, batch_size=batch_size)

chooseLabels(classes_done, classes, test_imagenames, output);

