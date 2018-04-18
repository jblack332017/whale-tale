# Whale Tale

#### All Code for our submissions for the Kaggle Humpback Whale Identification Challenge https://www.kaggle.com/c/whale-categorization-playground

![Alt text](./charts/whales_sample.png?raw=true "Whale Image Samples")

## Folders

### Training sets

```
training_set 
|   train.csv - image labels
│
└───train
│   │   whale011.jpg
│   │   whale012.jpg
│   │   ...
```

* input - original images from Kaggle
* copied-input - copies of original images to increase image count
* input-sans-new-whale - images without the new_whale tag
* augmented-correct-gray - randomly gray augmented images
* augmented-correct-rotation - randomly rotated augmented images
* augmented-correct-shear - randomly sheared augmented images
* augmented-correct-zoom - randomly zoomed augmented images
* augmented-correct-colors - randomly augmented images using all techniques
* augmented-correct-large - large randomly augmented images using all techniques
* small, test, baby - used for testing

### Other Folders

charts - contains all charts used in final report
logs - contains logs from model training 
epoch_tests - contains output and results from epoch testing
output_skew_test - contains output and results from skew testing with new_whale
submissions - contains submissions

## Scripts


### create_image_diagrams.py
Creates augmented image diagrams
```
$ python create_image_diagrams.py
```

### image-augmenter.py
Create augmented image sets
```
python image-augmenter.py <INPUT_DIR> <OUTPUT_DIR>
```

### image-copier.py
Create copied image sets
```
python image-copier.py <INPUT_DIR> <OUTPUT_DIR>
```

## Train Model and Predict Test Images

```
$ python whale-tale.py <OUTPUT_PREDICTION_CSV> <TRAINING_SET1> <TRAINING_SET2> ... > <OUTPUT_LOG>
```
