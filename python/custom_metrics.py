import numpy as np
from keras import backend as K
import tensorflow as tf
import keras
import sklearn.metrics as sklm

def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def average_precision_at_k(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    return tf.metrics.sparse_average_precision_at_k(y_true, y_pred, 2)


class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
        self.kappa = []
        self.auc = []
        self.ap = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))[:,1]
        targ_all = self.validation_data[1]
        targ = targ_all[:,1]

        ap = sklm.average_precision_score(targ_all, score)
        self.ap.append(ap)
        print('Average Precision : %s' % ap)

        auc = sklm.roc_auc_score(targ_all, score)
        self.auc.append(auc)
        print('AUC : %s' % auc)

        conf = sklm.confusion_matrix(targ, predict)
        self.confusion.append(conf)
        print('Confusion Matrix : %s' % conf)

        precision = sklm.precision_score(targ, predict)
        self.precision.append(precision)
        print('Precision : %s' % precision)
        
        recall = sklm.recall_score(targ, predict)
        self.recall.append(precision)
        print('Confusion Matrix : %s' % recall)
        
        f1s = sklm.f1_score(targ, predict)
        self.f1s.append(f1s)
        print('F1 : %s' % f1s)

        return