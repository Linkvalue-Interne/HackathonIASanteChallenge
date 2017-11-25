import os
import numpy as np
import random
import scipy
import sys
import json
import PIL, PIL.Image
from PIL import Image as pil_image
import imageio

import time

from scipy.ndimage import imread
from functools import partial

from multiprocessing import Pool

def generate_arrays_from_bottleneck_folder(path, batch_size=32, target_size=(224,224)):
    '''
    Generator that reads the precomputed weights from the files
    and gives it to the trainer.
    
    It shuffles the entries on every epoch.
    '''
    labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    labels.sort()
    
    labels_map = {}
    for i, label in enumerate(labels):
        labels_map[label] = i
    all_images = []
    
    for i, label in enumerate(labels):
        images = [str(a) for a in os.listdir(os.path.join(path, label))]
        for image in images:
            all_images.append((label, image))
    while 1:
        
        random.shuffle(all_images)
        X = np.zeros((batch_size, target_size[0], target_size[1], 3))
        Y = np.zeros((batch_size, len(labels)))
        for i in range(batch_size):
            entry = all_images[i]
            label = entry[0]
            image = entry[1]
            
            x = imread(os.path.join(path, label, image))
            x = scipy.misc.imresize(x, target_size)
            x = x.astype('float32') / 255.
            X[i] = x
            Y[i] = labels_map[label]

        yield (X, Y)

def read_image(img, path='' ,target_size=(224,224)):
    img_name = img[1]
    label = img[0] 
    path = os.path.join(path, label, img_name)
    x = imread(path)
    x = scipy.misc.imresize(x, target_size)
    x = x.astype('float32') / 255.
    return x

def load_set(path, target_size=(224,224), data_aug_range=None, shuffle=True, return_img_name=False):
    if data_aug_range is not None:
        labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) \
            and int(d.split('_')[-1].split('.')[0]) in data_aug_range]
    else:
        labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    labels.sort()
    
    labels_map = {}
    for i, label in enumerate(labels):
        labels_map[label] = i
    all_images = []
    
    for i, label in enumerate(labels):
        images = [str(a) for a in os.listdir(os.path.join(path, label))]
        for image in images:
            all_images.append((label, image))
    if shuffle:
        random.shuffle(all_images)
    print('Reading %s images' % L)
    # L = 5000
    X = np.zeros((L, target_size[0], target_size[1], 3))
    Y = np.zeros((L, len(labels)))
    st = time.time()
    pool = Pool(12)
    X_list = pool.map(partial(read_image, target_size=target_size, path=path), all_images[shift:L])
    pool.close() #we are not adding any more processes
    pool.join()
    print('Images loaded in memory in %s sec' % (int(time.time()-st)))
    for i, x in enumerate(X_list):
        X[i] = x
        Y[i, labels_map[all_images[i][0]]] = 1.

    if return_img_name:
        return X, Y, [a[1] for a in all_images[:L]]
    return X, Y


# gen = generate_arrays_from_bottleneck_folder('/sharedfiles/challenge_data/data/train', batch_size=32, target_size=(224,224))

# for x, y in gen:
#     print(x.shape)
#     print(y.shape)

