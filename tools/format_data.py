"""
-*- reads, formats, & pickles image data -*-

@author:    alexjaniak
@date:      6/2/20
@file:      format_data.py  

to run module issue the command: 

    python format_data.py --image_dir PATH/TO/THE/IMAGES_DIR --save_path PATH/TO/SAVE_PATH
    
    output:
        train.pkl & test.pkl -> pickled pandas DataFrames saved at PATH/TO/SAVE_PATH:
            images: (-1, height, width, channels)
            labels: (-1, 1) -> int
            file_names: (-1, 1) -> 'file_name123.jpg'  
            formated_fnames: (-1, 1) -> 'file name'
"""
# imports 
# TODO: only include necessary inputs i.e from numpy import array
import os
import argparse
from tqdm import tqdm

from PIL import Image
import numpy as np 
import pandas as pd

from config import data_cfg

# TODO: include timer for non-progress bar processing
# @main
def format_data(image_dir, save_path):
    """
    reads, formats, & pickles image data

    :param image_dir: the image directory
    :param save_path: the save path
    :return: returns nothing
    """
    # generate data data frame 
    data = pd.DataFrame(pull_raw_image_data(image_dir))
    data["formated_fnames"] = format_file_names(data.file_names)
    data["labels"] = encode_labels(data.formated_fnames)

    # shuffle & split data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle
    idx = int(np.floor(data_cfg["TRAIN_SPLIT"]*data.shape[0])) # split index
    test, train = data.iloc[:idx],data.iloc[idx:] # split data into training & test
 
    # pickle data frames
    train_file_path = os.path.join(save_path, "train.pkl")
    test_file_path = os.path.join(save_path,"test.pkl")

    print("[INFO] Pickling to {} ...".format(train_file_path))
    train.to_pickle(train_file_path)
    print("[INFO] Pickling to {} ...".format(test_file_path))
    test.to_pickle(test_file_path)
 
#TODO: use glob.glob()
def pull_raw_image_data(image_dir):
    """
    reads images from image_dir

    :param image_dir: the image directory
    :return: dict containing image pixel vals and file names
    """
    imgs, fnames = [], []
    for fname in tqdm(os.listdir(image_dir), desc="Loading Files"): # progress bar
        split = os.path.splitext(fname) # splits file_name.ext to [file_name, ext]
        if split[1].lower() == '.jpg': # only opens .jpg files
            with Image.open(os.path.join(image_dir, fname)) as img:
                imgs.append(get_image_vals(img))
            fnames.append(fname) 
    return {'images':imgs, 'file_names':fnames}

#TODO: use real expressions
def format_file_names(file_names):
    """
    formats file names from 'file_name123.jpg' to 'file name'

    :param files: list of file names 
    :return: list of formated file names
    """
    formated_fnames = []
    for f in tqdm(file_names, desc="Formating Files"): # progress bar
        formated = os.path.splitext(f)[0].replace("_", " ")
        formated = ''.join(c for c in formated if not c.isdigit())
        formated_fnames.append(formated.rstrip())
    
    return formated_fnames

def get_image_vals(image):
    """
    converts PIL image to array of pixel vals

    :param image: PIL image object
    :return: array of pixel vals
    """
    return np.array(image.getdata()).reshape(image.size[0], image.size[1], -1)

def encode_labels(labels):
    """
    creates labels for categorical variables using label encoding

    :param labels: list of categorical labels 
    :return: list of encoded labels
    """
    print("[INFO] Encoding Labels ...")
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
    return labels

def init_args():
    """
    initializes command line args

    :return: args
    """
    parser = argparse.ArgumentParser(description="Formats image data")
    parser.add_argument('--image_dir', type=str, help='The image dir')
    parser.add_argument('--save_path', type=str, help='The save path')
    return parser.parse_args()

if __name__ == "__main__":
    args = init_args() #init 
    format_data(args.image_dir, args.save_path) # calls main func
