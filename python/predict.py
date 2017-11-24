
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import pretrained_models, custom_metrics

try:
    import configparser
except ImportError: # Python 2.*
    import ConfigParser as configparser 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os, sys
import time

config = configparser.ConfigParser()
config.read('config.ini')
#Read Basic Metadata from config file
train_data_dir, validation_data_dir, results_dir, models_dir, log_dir = map(lambda x :
                                                                   x[1], 
                                                                   config.items("base"))

if len(sys.argv)>1:
    model_section = sys.argv[1]
    try:
        elements = config.items(model_section)
    except configparser.NoSectionError:
        print("Model %s not defined." % model_section)
        print("See defined sections : %s" % config.items())
        elements = config.items('default')
    weights_file_path = sys.argv[2]
else:
    elements = config.items('default')

img_width, img_height,\
batch_size, epochs, \
training_steps_per_epoch, validation_steps_per_epoch,\
positive_weight, model_name = map(lambda x :
                                         int(x[1]) if
                                         x[1].isdigit()
                                         else x[1],
                                         elements)

model = pretrained_models.model_definition(model_name, (img_width, img_height, 3))
 
x = model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

model_finale.load_weights(weights_file_path)

# compile the model 
model_final.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy", custom_metrics.precision, custom_metrics.recall])

