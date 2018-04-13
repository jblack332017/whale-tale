import pandas as pd
import numpy as np
from glob import glob
import cv2
import os
from os import listdir
from PIL import Image

from keras.preprocessing.image import img_to_array

SIZE = 256
def ImportImage(filename):
    img = Image.open(filename).convert('LA').resize((SIZE,SIZE))
    img_arr = img_to_array(img)
    img_arr = img_arr.astype(int)
    return img_arr

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
    testing_images = glob('test/*jpg')
    testing_list = listdir('test')
    testing_imgs = np.array([ImportImage( img) for img in testing_images])
    testing_names = np.array(testing_list)
    return {'test_images': testing_imgs, 'test_names': testing_names}
