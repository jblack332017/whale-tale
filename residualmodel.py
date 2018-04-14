from __future__ import print_function
import numpy as np
import keras
import sys
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, add, BatchNormalization

def createModel(input_shape, training_shape, nClasses):
  print(training_shape)
  calibInput = Input(shape=(128, 128, 32))
  block1_bn1 = BatchNormalization()(calibInput)
  block1_a1 = Activation('relu')(block1_bn1)
  block1_w1 = Conv2D(32, (3, 3), padding='same', activation='relu')(block1_a1)
  block1_bn2 = BatchNormalization()(block1_w1)
  block1_a2 = Activation('relu')(block1_bn2)
  block1_w2 = Conv2D(32, (3, 3), padding='same', activation='relu')(block1_a2)
  block1_output = add([block1_w2, calibInput])

  block2_bn1 = BatchNormalization()(block1_output)
  block2_a1 = Activation('relu')(block2_bn1)
  block2_w1 = Conv2D(64, (3, 3), padding='same', activation='relu')(block2_a1)
  block2_bn2 = BatchNormalization()(block2_w1)
  block2_a2 = Activation('relu')(block2_bn2)
  block2_w2 = Conv2D(64, (3, 3), padding='same', activation='relu')(block2_a2)
  block2_output = add([block2_w2, block1_output])

  block3_bn1 = BatchNormalization()(block2_output)
  block3_a1 = Activation('relu')(block3_bn1)
  block3_w1 = Conv2D(128, (3, 3), padding='same', activation='relu')(block3_a1)
  block3_bn2 = BatchNormalization()(block3_w1)
  block3_a2 = Activation('relu')(block3_bn2)
  block3_w2 = Conv2D(128, (3, 3), padding='same', activation='relu')(block3_a2)
  block3_output = add([block3_w2, block2_output])

  

  model = Model(inputs=calibInput, outputs=block3_output)
  return model
