import sys
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
#import cv2
import os
from PIL import Image


def ImportImage(filename):
    # img = cv2.imread(filename)
    print(filename);
    img = plt.imread(filename);
    return img

def training_data(training_sets):
    # Initialize empty dataframe
    df = None;
    for training_set in training_sets:
      # Remove ending /
      if training_set[-1] == "/":
        training_set = training_set[:-1];
      # Append train.csv for each set into the dataframe
      train = pd.read_csv(training_set + '/train.csv');
      train['Image'] = (training_set + '/train/') + train['Image'].astype(str);
      if df:
        df = df.append(train);
      else:
        df = train;
    
    image_paths = df['Image'].tolist();
    print(image_paths);
    train_imgs = np.array([ImportImage(path) for path in image_paths])
    train_labels = np.array(df['Id'].tolist());
    return {'train_images': train_imgs, 'train_labels': train_labels}

training_data(sys.argv[1:])
