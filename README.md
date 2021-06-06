# Body Type Classifier 
Classifies cat & dog breeds using a CNN with the [ResNet](https://arxiv.org/abs/1512.03385) architecture and the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). Currently only the 34-layer variant is implemented with other variants in future updates. The project's end goal is to serve as a body type classifier in the finest-ml pipeline. Alternatively, it could also serve as a template for other CV projects. 

## Installation
This project has only been tested using *python 3.9.5* & *pip 21.1.2*. You can install the required dependencies by running: 
```bash
pip install -r requirements.txt
```

## Data
Currently, the project only supports the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). The dataset contains 7349 images of 37 cat & dog breeds. Each breed has roughly 200 images with large variations in scale, pose and lighting.

Once you download the dataset you can format the data by running: 

```bash
python tools/format_data.py 
--image_dir PATH/TO/THE/IMAGES_DIR
--save_path PATH/TO/SAVE_PATH
```

This will save a pickled pandas DataFrame of the formated data into *PATH/TO/SAVE_PATH/train.pkl* & *PATH/TO/SAVE_PATH/test.pkl*

## Train Model
Once you format the data for the model you can train the 34-layer ResNet by running:

```bash
python tools/train_34.py 
--train_path PATH/TO/train.pkl
--test_path PATH/TO/test.pkl
--save_path PATH/TO/models/model1
```

This will save the model to PATH/TO/models/model1, plot the training loss & metrics, and output the loss & accuracy of the model on the test set.

## Config
You can edit the training split, validation split, epochs, as well as the parameters for the Adam optimizer in config.py. 


