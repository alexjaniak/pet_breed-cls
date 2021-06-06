"""
-*- trains & tests a ResNet34 model -*-
@author:    alexjaniak
@date:      6/2/20
@file:      train_34.py  

to run module issue the command: 

    python tools/train_34.py --train_path PATH/TO/train.pkl --test_path PATH/TO/test.pkl --save_path PATH/TO/models/model1
    
    output:
        model -> trained ResNet34 model saved at PATH/TO/models/model1:
        test -> model test performance
"""

# imports
# TODO: only import necessities
import argparse
from tqdm import tqdm

from tensorflow import image as tf_image
from tensorflow import keras
from pandas import DataFrame, read_pickle  
import matplotlib.pyplot as plt

from ResNet import resnet34
from config import data_cfg, model_cfg

def main(train_path, test_path, save_path):
    """
    trains & tests a ResNet-34 model 

    :param train_path: the pickled training data path
    :param test_path: the pickled testing data path
    :param save_path: the save path
    :return: returns nothing
    """
    # unpickle training data
    print("[INFO] Loading data ...")
    train = read_pickle(train_path)
    test = read_pickle(test_path)

    # train model
    print("[INFO] Training Model")
    resized_train_images = resize_images(train.images)

    ## init model
    # TODO: imput config into Adam optimizer
    rn34 = resnet34(output_nodes = data_cfg["CATEGORICAL_VARIABLES"])
    rn34.compile(loss=keras.losses.sparse_categorical_crossentropy,
                 optimizer=keras.optimizers.Adam(),
                 metrics=keras.metrics.sparse_categorical_accuracy)

    ## training
    # TODO: implement early stopping & model checkpoints
    history = rn34.fit(resized_train_images, train.labels, epochs=model_cfg["EPOCHS"], validation_split=model_cfg["VALIDATION_SPLIT"])
    plot_history(history)

    # test model
    print("[INFO] Evaluating Model")
    resized_test_images = resize_images(test.images)
    rn34.evaluate(resized_test_images, test.labels)

    # save model
    rn34.save(save_path)

def plot_history(history):
    """
    plots training metrics

    :param history: training history (keras model output)
    :return: returns nothing
    """
    DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

# TODO: try cut and resize tf
# TODO: add as preprocessing layer
def resize_images(images, shape=(224,224)):
    """
    resizes list of images

    :param images: pixel vals of images
    :param shape: shape images are resized to
    :return: list of resized images
    """
    resized_images = [] 
    for image in tqdm(images, desc="Resizing Images"): # progress bar
        resized_images.append(tf_image.resize(image, shape)) # resizes images (streches)
    return resized_images

def init_args():
    """
    initializes command line args

    :return: args
    """
    parser = argparse.ArgumentParser(description="Trains ResNet-34 model")
    parser.add_argument('--train_path', type=str, help='The pickled training data path')
    parser.add_argument('--test_path', type=str, help='The pickled test data path')
    parser.add_argument('--save_path', type=str, help='The model checkpoint')
    return parser.parse_args()

if __name__ == "__main__":
    # executes when train_34.py is run directly 
    args = init_args() # initializes command line args
    main(train_path=args.train_path,
         test_path=args.test_path,
         save_path=args.save_path)
     