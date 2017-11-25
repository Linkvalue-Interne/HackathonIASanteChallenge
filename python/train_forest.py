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
import xgboost as xgb

class ModelParams :
    def __init__(self, model_section):
        self.model_section = model_section

    def get_metadata(self, config):
        self.img_width, self.img_height,\
        self.batch_size, self.epochs, \
        self.training_steps_per_epoch, self.validation_steps_per_epoch,\
        self.positive_weight, self.model_name = get_metadata_model(config,
                                                                   self.model_section)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x for x in local_device_protos if x.device_type == 'GPU'])

def features_extraction(models, config, model_section):


def train(models, config, model_section):
    models_param = []

    for ms in model_section :
        mp = ModelParams(ms)
        mp.get_metadata(config)
        models_param += mp

    #Read Basic Metadata from config file
    train_data_dir, validation_data_dir, test_data_dir,\
    results_dir, models_dir, log_dir = map(lambda x : x[1],
                                            config.items("base"))

    X_train, Y_train = load_set(train_data_dir, target_size=(img_height, img_width), data_aug_range=[1,3,5,7])

    X_val, Y_val = load_set(validation_data_dir, target_size=(img_height, img_width))

    metrics = custom_metrics.Metrics()

    cnn_outputs_train = []
    cnn_outputs_val = []

    for model,mp in zip(models, models_param): 
        cnn_outputs_train += model.predict(
            X_train,
            batch_size=mp.batch_size,
            verbose=1
        )
        cnn_outputs_val += model.predict(
            X_val,
            batch_size=mp.batch_size,
            verbose=1
        )

    forest_input_train = reduce(lambda x,y : x + y, map(lambda x : 0.5*x,
                                                        cnn_outputs_train))
    forest_input_val = reduce(lambda x,y : x + y, map(lambda x : 0.5*x,
                                                        cnn_outputs_val))
    
    xgb_train = xgb.DMatrix(forest_input_train, Y_train)
    xgb_val = xbg.DMatrix(forest_input_val, Y_val)

    param = {'max_depth' : 3, 'eta' : 0.1, 'objective' : 'binary:logistic',
             'seed' : 42}
    num_round = 50
    bst = xgb.train(param, xgb_train, num_round, [(xgb_val, 'val'), (xgb_train, 'train')])

    return bst


def predict(models, config, model_section, bst):

    models_param = []

    for ms in model_section :
        mp = ModelParams(ms)
        mp.get_metadata(config)
        models_param += mp

    #Read Basic Metadata from config file
    train_data_dir, validation_data_dir, test_data_dir,\
    results_dir, models_dir, log_dir = map(lambda x : x[1],
                                            config.items("base"))

    X_test, _, return_img_names = load_set(test_data_dir, target_size=(img_height, img_width), shuffle=False, return_img_name=True)

    metrics = custom_metrics.Metrics()

    cnn_outputs_tes = []

    for model,mp in zip(models, models_param): 
        cnn_outputs_test += model.predict(
            X_test,
            batch_size=mp.batch_size,
            verbose=1
        )

    forest_input_test = reduce(lambda x,y : x + y, map(lambda x : 0.5*x,cnn_outputs_test))
    xgb_test = xgb.DMatrix(forest_input_test)

    predictions = bst.predict(xgb_test)
    
    
    print('Prediction done.')
    print(predictions[:10])
    preds = pd.DataFrame({'name' : return_img_names, 'risk' : predictions[:,1]})
    preds.to_csv('%s/%s_%d.csv' % (results_dir, model_file_name, int(time.time())), index=False)

def load_data(config):
    train_data_dir, validation_data_dir, test_data_dir,\
    results_dir, models_dir, log_dir = map(lambda x : x[1],
                                            config.items("base"))
    x = np.array([np.array(Image.open(fname)) for fname in filelist])


def load_models(config, model_section=None, weights_file=None):

    models = []
    models_param = []

    for ms in model_section :
        mp = ModelParams(ms)
        mp.get_metadata(config)
        models_param += mp

    for mp,wf in zip(models_param, weights_file) :
        model = pretrained_models.model_definition(mp.model_name,
                                                   (mp.img_width, mp.img_height, 3))

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
        model_final = Model(input = model.input, output = x)
        model_final.load_weights(wf)

        models += model_final

    return models

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

    model_section = sys.argv[1].split(',')
        if len(sys.argv)>2:
            weights_file = sys.argv[2].split(',')
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
    models = load_models(config, model_section=model_section, weights_file=weights_file)
    #model_final = make_parallel(model_raw, gpuNumber)
    # compile the models
    for model in models: 
        model.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9),
            metrics=["accuracy",
                custom_metrics.precision,
                custom_metrics.recall,
                #custom_metrics.average_precision_at_k
            ])

    bst = train(models, config, model_section)
    predict(models, config, model_section, bst)
