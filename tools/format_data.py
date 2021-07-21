"""
-*- functions for formating data -*-

@author:    alexjaniak
@date:      7/19/21
@file:      format_data.py  
"""

import os 
import numpy as np 
from PIL import Image 
from tensorflow import image as tf_image

# TODO: real expression
def format_file_name(file_name):
    """
    formats file name from 'file_name123.jpg' to 'file name'

    :param files: file name string
    :return: formatted file name string
    """

    formated_file_name = os.path.splitext(file_name)[0].replace("_", " ") # file_name123.jpg -> file name123
    formated_file_name = ''.join(c for c in formated_file_name if not c.isdigit()) # only letters
    return formated_file_name.rstrip() # remove white space

def get_image_vals(file_path):
    """
    returns pixel image vals 

    :param image: file path
    :return: numpy array of pixel vals
    """

    with Image.open(file_path) as image:
        image_vals = image.get_data().reshape(image.size[0], image.size[1])
        return np.array(image_vals, -1)


def encode_labels(labels):
    """
    creates labels for categorical variables using label encoding

    :param labels: list of categorical labels 
    :return: list of encoded labels & key
    """
    
    # get list of categorical vars
    cat_vars = []
    for i in labels:
        if i not in cat_vars:
            cat_vars.append(i)
    
    # create label encoding key
    cat_key = {}
    for i in range(len(cat_vars)):
        cat_key[cat_vars[i]] = i

    labels = [cat_key[i] for i in labels] # encode labels
    return labels, cat_key


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
    for image in images:
        resized_images.append(tf_image.resize(image, shape)) # resizes images (streches)
    return resized_images