import os
import numpy as np
import random

import sys
import json
import PIL, PIL.Image
import imageio

from scipy.ndimage import imread

def generate_arrays_from_bottleneck_folder(path):
    '''
    Generator that reads the precomputed weights from the files
    and gives it to the trainer.
    
    It shuffles the entries on every epoch.
    '''
    labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    labels.sort()
    
    labels_map = {}
    for i in range(len(labels)):
        label = labels[i]
        labels_map[label] = i
    print(labels)
    all_images = []
    
    for i in range(len(labels)):
        label = labels[i]
        images = [str(a) for a in os.listdir(os.path.join(path, label))]
        print(len(images))
        #images.sort()
        for image in images:
            all_images.append((label, image))
    print(len(all_images))
    while 1:
        
        random.shuffle(all_images)
        for i in range(len(all_images)):
            entry = all_images[i]
            label = entry[0]
            image = entry[1]
            
            x = imread(os.path.join(path, label, image))
            y = np.zeros((1, len(labels)))
            y[0, labels_map[label]] = 1

            yield (x, y)

