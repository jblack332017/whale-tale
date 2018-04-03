import sys
import pandas as pd
import numpy as np
from glob import glob
import cv2
import os
from os import listdir
from PIL import Image

SIZE = 128
def ImportImage(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img ,(int(SIZE),int(SIZE)))
    return img

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def training_data(training_sets):
    training_images = []
    image_label_dict = {}
    for training_set in training_sets:
        training_images = training_images + glob(training_set + '/train/*jpg')
        df = pd.read_csv(training_set + '/train.csv')
        df["Image"] = df["Image"].map( lambda x : training_set+ '/train/' + x)
        image_label_dict = merge_two_dicts(image_label_dict, dict( zip( df["Image"], df["Id"])))


    train_imgs = np.array([ImportImage( img) for img in training_images])
    train_labels = list(map(image_label_dict.get, training_images))
    return {'train_images': train_imgs, 'train_labels': np.array(train_labels)}


def test_data():
    testing_images = glob(f'test/*jpg')
    testing_list = listdir('test')
    testing_imgs = np.array([ImportImage( img) for img in testing_images])
    testing_names = np.array(testing_list)
    return {'test_images': testing_imgs, 'test_names': testing_names}

training_data(sys.argv[1:])
