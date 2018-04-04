from __future__ import print_function
import numpy as np
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.preprocessing import LabelBinarizer

def trainModel(model, train_data, train_labels, epochs, batch_size):
  # Change the labels from integer to categorical data
  encoder = LabelBinarizer()
  train_labels_one_hot = encoder.fit_transform(train_labels)
  
  # Display the change for category label using one-hot encoding
  print('Original label 0 : ', train_labels[0])
  print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[0])
  
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()
  history = model.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1)
  return (model, history);
