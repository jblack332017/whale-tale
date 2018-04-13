from __future__ import print_function
import numpy as np
import keras
import sys
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from keras import applications
from keras import optimizers
from keras.applications.vgg16 import VGG16



def createModel(input_shape, nClasses):
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    input = Input(shape=input_shape,name = 'image_input')

    output_vgg16_conv = model_vgg16_conv(input)
    #Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(nClasses, activation='softmax', name='predictions')(x)

    my_model = Model(input=input, output=x)



    # base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    #
    # add_model = Sequential()
    # add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    # add_model.add(Dense(256, activation='relu'))
    # add_model.add(Dense(nClasses, activation='softmax'))
    #
    # model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    # model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    #               metrics=['accuracy'])

    # model = Sequential()
    # # The first two layers with 32 filters of window size 3x3
    # model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(nClasses, activation='softmax'))

    return model
