import os 
import numpy as np 
import glob 
from PIL import Image 

# TODO: real expression
def format_file_name(file_name):
    """
    formats file name from 'file_name123.jpg' to 'file name'

    :param files: file name string
    :return: formatted file name string
    """
    formated_file_name = os.path.splitext(file_name)[0].replace("_", " ")
    formated_file_name = ''.join(c for c in formated_file_name if not c.isdigit())
    return formated_file_name.rstrip()

def get_image_vals(file_path):
    """
    gets pixel image vals 

    :param image: PIL image object
    :return: array of pixel vals
    """
    return np.array(image.getdata()).reshape(image.size[0], image.size[1], -1)

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