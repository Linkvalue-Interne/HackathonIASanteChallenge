from keras import applications, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils.training_utils import multi_gpu_model

import pretrained_models, custom_metrics
from make_parallel import make_parallel

from generators import generate_arrays_from_bottleneck_folder, load_set

try:
    import configparser
except ImportError: # Python 2.*
    import ConfigParser as configparser
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os, sys
import time
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x for x in local_device_protos if x.device_type == 'GPU'])

def train(model_final, config, model_section):
    img_width, img_height,\
    batch_size, epochs, \
    training_steps_per_epoch, validation_steps_per_epoch,\
    positive_weight, model_name = get_metadata_model(config, model_section)

    #Read Basic Metadata from config file
    train_data_dir, validation_data_dir, test_data_dir,\
    results_dir, models_dir, log_dir = map(lambda x : x[1],
                                            config.items("base"))

    X_train, Y_train = load_set(train_data_dir, target_size=(img_height, img_width), data_aug_range=[0,2,4])

    X_val, Y_val = load_set(validation_data_dir, target_size=(img_height, img_width))

    # prepare the tensorboard
    timestamp = time.time()
    tbCallBack = TensorBoard(log_dir=log_dir + '/' + model_name + '/' + str(int(timestamp)), histogram_freq=0,
        write_graph=True, write_images=True)

    # Save the model according to the conditions
    file_name = models_dir + '/' + model_section + "_onelayer.h5"
    checkpoint = ModelCheckpoint(file_name, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    # early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
    metrics = custom_metrics.Metrics()

    # Train the model
    print('Training model %s and saving snapshot at %s' %(model_section, file_name))

    model_final.fit(
        X_train,
        Y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = (X_val, Y_val),
        verbose=1,
        class_weight = {0 : 1., 1 : positive_weight},
        callbacks = [
            metrics,
            tbCallBack,
            checkpoint,
            #early
            ])

def evaluate(model_final, config):

    img_width, img_height,\
    batch_size, epochs, \
    training_steps_per_epoch, validation_steps_per_epoch,\
    positive_weight, model_name = get_metadata_model(config, model_section)

    train_data_dir, validation_data_dir, test_data_dir,\
    results_dir, models_dir, log_dir = map(lambda x : x[1],
                                            config.items("base"))

    X_val, Y_val = load_set(validation_data_dir, target_size=(img_height, img_width))

    print(model_final.evaluate(
        X_val,
        Y_val,
        batch_size=batch_size,
        verbose=1))

    print('Evaluation done.')


def predict(model_final, config, model_file_name):

    img_width, img_height,\
    batch_size, epochs, \
    training_steps_per_epoch, validation_steps_per_epoch,\
    positive_weight, model_name = get_metadata_model(config, model_section)

    train_data_dir, validation_data_dir, test_data_dir,\
    results_dir, models_dir, log_dir = map(lambda x : x[1],
                                            config.items("base"))

    X_test, _, return_img_names = load_set(test_data_dir, target_size=(img_height, img_width), shuffle=False, return_img_name=True)

    predictions = model_final.predict(
        X_test,
        batch_size=batch_size,
        verbose=1
    )
    print('Prediction done.')
    print(predictions[:10])
    preds = pd.DataFrame({'name' : return_img_names, 'risk' : predictions[:,1]})
    preds.to_csv('%s/%s_%d.csv' % (results_dir, model_file_name, int(time.time())), index=False)

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
    x = Dense(1024, activation="relu",
            kernel_regularizer=regularizers.l2(0.01)
        )(x)
    # x = Dropout(0.5)(x)
    # x = Dense(1024, activation="relu",
    #     kernel_regularizer=regularizers.l2(0.01)
    #     )(x)
    predictions = Dense(2, activation="softmax")(x)

    # creating the final model
    model_final = Model(input = model.input, output = predictions)

    if weights_file is not None:
        model_final.load_weights(weights_file)

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

    # Get numbers of available gpus
    gpuNumber = get_available_gpus()
    # Load model
    model_raw = load_model(config, model_section=model_section, weights_file=weights_file)
    model_final = make_parallel(model_raw, gpuNumber)
    # compile the model 
    model_final.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9),
        metrics=["accuracy",
            custom_metrics.precision,
            custom_metrics.recall,
            #custom_metrics.average_precision_at_k
        ])

    if mode == 'train':
        train(model_final, config, model_section)
    elif mode == 'predict':
        predict(model_final, config, model_section)
    elif mode == 'evaluate':
        evaluate(model_final, config)
    else:
        print('unknown mode.')
