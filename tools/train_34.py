"""
-*- trains & tests a ResNet34 model -*-
@author:    alexjaniak
@date:      7/20/21
@file:      train_34.py  

to run module issue the command: 

    python tools/train_34.py --train_path PATH/TO/train.pkl --test_path PATH/TO/test.pkl --save_path PATH/TO/models/model1
    
    output:
        model -> trained ResNet34 model saved at PATH/TO/models/model1:
        test -> model test performance
"""

import format_data as fd 
import argparse 
import confuse
import glob
import os
from ResNet import resnet34
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

def main(image_dir, save_path, plot_history = True):
    """
    trains & evaluates a ResNet-34 model

    :param image_dir: /PATH/TO/IMAGE/DIR
    :param save_path: /PATH/TO/MODEL/DIR
    :param plot_history: bool for plotting training metrics
    :return: returns nothing
    """
    ## load data
    file_paths = glob.glob(os.path.join(image_dir,'*.jpg')) # gets file paths from dir
    data, label_key = read_data(file_paths) # read data from file paths
    data.images = fd.resize_images(data.images, shape=(224,224)) # resize images

    # shuffle & split
    X_train, X_test, y_train, y_test = train_test_split(data.images,
                                                        data.labels,
                                                        train_split = config['data']['train_split'], 
                                                        random_state=42)

    ## train model
    rn34, history = train_rn34(X_train, y_train)

    # plot training metrics 
    if plot_history: plot_history(history)

    ## eval model
    rn34.evaluate(X_test, y_test)

    #TODO: add better save names
    ## save model
    rn34.save(save_path)
    


def train_rn34(images, labels):
    """
    trains a ResNet-34 model

    :param images: list of resized image values
    :param labels: list of image labels
    :returns: trained keras ResNet34 model
    """
    ## init model
    # TODO: imput config into Adam optimizer
    rn34 = resnet34(output_nodes = config['data']['pets']['labels'])
    rn34.compile(loss=keras.losses.sparse_categorical_crossentropy,
                 optimizer=keras.optimizers.Adam(),
                 metrics=keras.metrics.sparse_categorical_accuracy)

    ## training
    # TODO: implement early stopping & model checkpoints
    history = rn34.fit(images,
                       labels,
                       epochs=config['model']['epochs'],
                       validation_split=config['model']['val_split'])

    return rn34, history 

def plot_history(history):
    """
    plots training metrics

    :param history: training history (keras model output)
    :return: returns nothing
    """
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

def read_data(file_paths):
    """
    reads image data from file paths

    :param file_paths: list of file (image) paths
    :return: pandas data frame with images & encoded labels
    :return: label key (dict)
    """
    images = []
    file_names = []
    for file_path in file_paths:
        images.append(fd.get_image_vals(file_path)) # get image value 
        file_names.append(fd.format_file_name(file_path)) # get formatted file name 
    labels, label_key = fd.encode_labels(file_names)

    return pd.DataFrame({'images': images, 'labels': labels}), label_key



# TODO: add optional arguments -> config 
# TODO: add plot setting
# TODO: image_dir 
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
    config = confuse.Configuration('ml_skeleton-cls') # reads config file  
    main(train_path=args.train_path,
         test_path=args.test_path,
         save_path=args.save_path)