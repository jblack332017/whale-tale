import math
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from PIL import Image

from tqdm import tqdm

from keras.preprocessing.image import (
    random_rotation, random_shift, random_shear, random_zoom,
    random_channel_shift, transform_matrix_offset_center, img_to_array)

INPUT_DIR = './input'

def plot_images_for_filenames(filenames, labels, rows=4):
    imgs = [plt.imread(f'{INPUT_DIR}/train/{filename}') for filename in filenames]

    return plot_images(imgs, labels, rows)


def plot_images(imgs, labels, first_image, rows=4):
    # Set figure to 13 inches x 8 inches
    figure, big_axes = plt.subplots( figsize=(13, 15) , nrows=rows+1, ncols=1, sharey=True)

    row_labels = ["Original Image", "Rotation", "Shear", "Zoom", "Greyscale", "All Techniques"]
    for row, big_ax in enumerate(big_axes, start=0):
        big_ax.set_title(row_labels[row], fontsize=16)
        # Turn off axis lines and ticks of the big subplot
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    cols = 5

    subplot = figure.add_subplot(rows+1, cols, 3)
    subplot.axis('Off')
    plt.imshow(first_image, cmap='gray')

    for i in range(1,len(imgs)+1):
        subplot = figure.add_subplot(rows+1, cols, i + 5)
        subplot.axis('Off')
        if labels:
            subplot.set_title(labels[i], fontsize=16)
        plt.imshow(imgs[i-1], cmap='gray')


np.random.seed(42)

train_df = pd.read_csv('./input/train.csv')
train_df.head()
file_name = "7f3495f3.jpg"
img_arr = cv2.imread( 'input/train/' + file_name)

img = Image.open('input/train/' + file_name).convert('RGB')
img_arr = img_to_array(img)
img_arr = img_arr.astype(int)
print(img_arr)
plt.imshow(img_arr)
imgs = []

imgs = imgs +  [
    random_rotation(img_arr, 30, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    for _ in range(5)]
# plot_images(imgs, None, rows=1)

# imgs = imgs +  [
#     random_shift(img_arr, wrg=0.1, hrg=0.3, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
#     for _ in range(5)]
# # plot_images(imgs, None, rows=1)

imgs = imgs + [
    random_shear(img_arr, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    for _ in range(5)]
# plot_images(imgs, None, rows=1)

imgs = imgs + [
    random_zoom(img_arr, zoom_range=(1.5, 0.7), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    for _ in range(5)]
# plot_images(imgs, None, rows=1)


import random

def random_greyscale(img, p):
    if random.random() < p:
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])

    return img

imgs = imgs +  [
    random_greyscale(img_arr, 0.5)
    for _ in range(5)]

# plot_images(imgs, None, rows=1)


def augmentation_pipeline(img_arr):
    img_arr = random_rotation(img_arr, 18, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_shear(img_arr, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_zoom(img_arr, zoom_range=(0.9, 2.0), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_greyscale(img_arr, 0.4)

    return img_arr

imgs = imgs +  [augmentation_pipeline(img_arr) for _ in range(5)]
plot_images(imgs, None, img_arr, rows=5)

plt.show()
