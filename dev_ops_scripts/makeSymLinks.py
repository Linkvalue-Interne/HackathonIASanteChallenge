import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

def getImagesFromFolder(_path):
    types = ['benign', 'malignant']
    images = [[], []] # 0: benign, 1: malignant

    for index, t in enumerate(types):
        path = str(_path) + str(t)
        for (dirpath, dirnames, filenames) in os.walk(path):
            images[index].extend(filenames)
            images[index] = np.array(images[index])
            break
    return images

def splitTrainTest(_path):
    images = getImagesFromFolder(_path)
    split = np.zeros((2, len(images)), dtype=object)

    for index, t in enumerate(images):
        X = images[index]
        X_train, X_test = train_test_split(X, test_size=0.20) #random_state=42
        split[0][index] =  X_train
        split[1][index] = X_test
    return split

def makeSymLinks(_pathFrom, _pathTo):
    os.makedirs(_pathTo, exist_ok=True)
    os.makedirs(_pathTo + 'train/benign', exist_ok=True)
    os.makedirs(_pathTo + 'train/malignant', exist_ok=True)
    os.makedirs(_pathTo + 'test/benign', exist_ok=True)
    os.makedirs(_pathTo + 'test/malignant', exist_ok=True)

    splitData = splitTrainTest(_pathFrom)

    for index, split in enumerate(splitData):
        path_part_1 = 'train/' if index == 0 else 'test/'
        for index2, images in enumerate(list(split)):
            path_part_2 = 'benign/' if index2 == 0 else 'malignant/'
            for image in images:
                source = _pathFrom + path_part_2 + image
                target = _pathTo + path_part_1 + path_part_2 + image

                if (os.path.isfile(target)):
                    os.remove(target)

                os.symlink(source, target)
                print(source, ' -> ', target)

    return splitData

args = sys.argv
if (len(args) < 3):
    print('invalid args - need input dir and output target')
    sys.exit()

makeSymLinks(args[1], args[2])
