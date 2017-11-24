
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import pretrained_models, custom_metrics

from generators import generate_arrays_from_bottleneck_folder

try:
    import configparser
except ImportError: # Python 2.*
    import ConfigParser as configparser 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os, sys
import time
import numpy as np

def train(model_final, config, model_section):

    img_width, img_height,\
    batch_size, epochs, \
    training_steps_per_epoch, validation_steps_per_epoch,\
    positive_weight, model_name = get_metadata_model(config, model_section)

    #Read Basic Metadata from config file
    train_data_dir, validation_data_dir, test_data_dir,\
    results_dir, models_dir, log_dir = map(lambda x : x[1], 
                                            config.items("base"))

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
        fill_mode = "nearest")

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size, 
        class_mode = "categorical")

    custom_train_generator = generate_arrays_from_bottleneck_folder(train_data_dir,
        batch_size=batch_size, target_size=(img_height, img_width))

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_height, img_width),
        class_mode = "categorical")

    # prepare the tensorboard
    timestamp = time.time()
    tbCallBack = TensorBoard(log_dir=log_dir + '/' + model_name + '/' + str(int(timestamp)), histogram_freq=0,
        write_graph=True, write_images=True)

    # Save the model according to the conditions
    file_name = models_dir + '/' + model_section + ".h5"
    checkpoint = ModelCheckpoint(file_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

    # Train the model 
    print('Training model %s and saving snapshot at %s' %(model_section, file_name))
    model_final.fit_generator(
        custom_train_generator,
        steps_per_epoch = training_steps_per_epoch,
        epochs = epochs,
        validation_data = validation_generator,
        validation_steps = validation_steps_per_epoch,
        verbose=1,
        class_weight = {0 : 1., 1 : positive_weight},
        callbacks = [tbCallBack, checkpoint, early])

def evaluate(model_final, config):

    img_width, img_height,\
    batch_size, epochs, \
    training_steps_per_epoch, validation_steps_per_epoch,\
    positive_weight, model_name = get_metadata_model(config, model_section)

    train_data_dir, validation_data_dir, test_data_dir,\
    results_dir, models_dir, log_dir = map(lambda x : x[1], 
                                            config.items("base"))

    test_datagen = ImageDataGenerator(
        rescale = 1./255,
        fill_mode = "nearest")

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_height, img_width),
        class_mode = "categorical")

    print(model_final.evaluate_generator(
        validation_generator,
        10000,
        workers=8,
        use_multiprocessing=False))

    print('Evaluation done.')


def predict(model_final, config, model_file_name):

    img_width, img_height,\
    batch_size, epochs, \
    training_steps_per_epoch, validation_steps_per_epoch,\
    positive_weight, model_name = get_metadata_model(config, model_section)

    train_data_dir, validation_data_dir, test_data_dir,\
    results_dir, models_dir, log_dir = map(lambda x : x[1], 
                                            config.items("base"))

    test_datagen = ImageDataGenerator(
        rescale = 1./255,
        fill_mode = "nearest")

    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size = (img_height, img_width),
        class_mode = "categorical")

    predictions = model_final.predict_generator(
        validation_generator,
        10000,
        verbose=1,
        workers=8,
        use_multiprocessing=False)

    print('Prediction done.')
    print(predictions[:10])
    np.save(results_dir + '/' + model_file_name, predictions)

def load_data(config):
    train_data_dir, validation_data_dir, test_data_dir,\
    results_dir, models_dir, log_dir = map(lambda x : x[1], 
                                            config.items("base"))
    x = np.array([np.array(Image.open(fname)) for fname in filelist])


def load_model(config, model_section=None, weights_file=None):

    img_width, img_height,\
    batch_size, epochs, \
    training_steps_per_epoch, validation_steps_per_epoch,\
    positive_weight, model_name = get_metadata_model(config, model_section)

    model = pretrained_models.model_definition(model_name, (img_width, img_height, 3))

    #Adding custom Layers 
    x = model.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)

    # creating the final model 
    model_final = Model(input = model.input, output = predictions)

    if weights_file is not None:
        model_final.load_weights(weights_file)

    # compile the model 
    model_final.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy", custom_metrics.precision, custom_metrics.recall])
    return model_final

def get_metadata_model(config, model_section):
    if model_section is not None:
        try:
            elements = config.items(model_section)
        except configparser.NoSectionError:
            print("Model %s not defined." % model_section)
            print("See defined sections : %s" % config.items())
            elements = config.items('default')
    else:
        elements = config.items('default')
    
    return map(lambda x : int(x[1]) if
                         x[1].isdigit()
                         else x[1],
                         elements)

if __name__ == "__main__":

    mode = sys.argv[1]
    if len(sys.argv)>2:
        model_section = sys.argv[2]
        if len(sys.argv)>3:
            weights_file = sys.argv[3]
        else:
            weights_file = None
    else:
        model_section = 'default'
        weights_file = None

    # Load configs
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Load model
    model_final = load_model(config, model_section=model_section, weights_file=weights_file)

    if mode == 'train':
        train(model_final, config, model_section)
    elif mode == 'predict':
        predict(model_final, config, model_section)
    elif mode == 'evaluate':
        evaluate(model_final, config)
    else:
        print('unknown mode.')

