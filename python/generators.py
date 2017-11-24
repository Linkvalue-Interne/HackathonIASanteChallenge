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
            
            x = imread(os.path.join(path, label, image),
                # flatten=True, mode='RGB'
                )
            # print(x.size)
            # if x.size != (target_size[0], target_size[1],3):
            #     resample = pil_image.NEAREST
            #     x = x.resize((target_size[0], target_size[1],3), resample)
            x = scipy.misc.imresize(x, target_size)
            # x = np.expand_dims(x, axis=2)
            x = x.astype('float32') / 255.
            X[i] = x
#            X[i] = x[:target_size[0], :target_size[1], 3]
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

def load_set(path, target_size=(224,224)):
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
    random.shuffle(all_images)
    L = len(all_images)
    # L = 5000
    X = np.zeros((L, target_size[0], target_size[1], 3))
    Y = np.zeros((L, len(labels)))
    st = time.time()
    pool = Pool(12)
    X_list = pool.map(partial(read_image, target_size=target_size, path=path), all_images[:L])
    pool.close() #we are not adding any more processes
    pool.join()
    print('Images loaded in memory in %s sec' % (int(time.time()-st)))
    for i, x in enumerate(X_list):
        X[i] = x
        Y[i, labels_map[all_images[i][0]]] = 1.

    return X, Y
# gen = generate_arrays_from_bottleneck_folder('/sharedfiles/challenge_data/data/train', batch_size=32, target_size=(224,224))

# for x, y in gen:
#     print(x.shape)
#     print(y.shape)

