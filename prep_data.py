import sys
import pandas as pd
import numpy as np
from glob import glob
import cv2
import os
from PIL import Image


def ImportImage(filename):
    img = cv2.imread(filename)
    return img

def training_data(training_sets):
    training_images = []
    image_label_dict = {}
    for training_set in training_sets:
        training_images = training_images + glob(f'{training_set}/train/*jpg')
        df = pd.read_csv(f'{training_set}/train.csv')
        df["Image"] = df["Image"].map( lambda x : f'{training_set}/train/{x}')
        image_label_dict = { **image_label_dict, **dict( zip( df["Image"], df["Id"])) }



    train_imgs = np.array([ImportImage( img) for img in training_images])
    train_labels = list(map(image_label_dict.get, training_images))
    return {'train_images': train_imgs, 'train_labels': np.array(train_labels)}

training_data(sys.argv[1:])
