
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


## TODO : Custom layers depending on the task
#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy", custom_metrics.precision, custom_metrics.recall])

# prepare the tensorboard
timestamp = time.time()
tbCallBack = TensorBoard(log_dir=log_dir + '/' + model_name + '/' + str(int(timestamp)), histogram_freq=0,
    write_graph=True, write_images=True)

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=180)

test_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=180)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size, 
    class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    class_mode = "categorical")

## TODO : See how to check val accuracy evolution depending on the instances
## and the model 
# Save the model according to the conditions  
checkpoint = ModelCheckpoint(models_dir + '/' + model_name + ".h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


## TODO : Check sample variables, especially early
# Train the model 

model_final.fit_generator(
    train_generator,
    steps_per_epoch = training_steps_per_epoch,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps_per_epoch,
    verbose=1,
    class_weight = {0 : 1., 1 : positive_weight},
    callbacks = [tbCallBack, checkpoint, early])
