import os
import sys
import csv
import numpy as np
from sklearn.model_selection import train_test_split


def readCsvFile(filename):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=";")
    rownum = 0
    data = []
    for row in reader:
        data.append (row)
        rownum += 1
    ifile.close()
    del data[0]
    return data

def splitByClass(data):
    malignant = list(filter(lambda x: x[1] == 'malignant', data))
    benign = list(filter(lambda x: x[1] == 'benign', data))
    return [ benign, malignant ] # [ image, className ]

def splitTrainTest(splitData):
    split = np.zeros((2, len(splitData)), dtype=object)
    for index, classType in enumerate(splitData):
        X_train, X_test = train_test_split(classType, test_size=0.20) #random_state=42
        split[0][index] =  X_train
        split[1][index] = X_test
    return split

def makeSymLinks(_splitData, _pathFrom, _pathTo):
    os.makedirs(_pathTo, exist_ok=True)
    os.makedirs(_pathTo + 'train/benign', exist_ok=True)
    os.makedirs(_pathTo + 'train/malignant', exist_ok=True)
    os.makedirs(_pathTo + 'test/benign', exist_ok=True)
    os.makedirs(_pathTo + 'test/malignant', exist_ok=True)

    for index, split in enumerate(_splitData):
        path_part_1 = 'train/' if index == 0 else 'test/'
        for index2, images in enumerate(list(split)):
            path_part_2 = 'benign/' if index2 == 0 else 'malignant/'
            for image in images:
                source = _pathFrom + image[0]
                target = _pathTo + path_part_1 + path_part_2 + image[0]
                try:
                    if (os.path.isfile(target)):
                        os.remove(target)

                    os.symlink(source, target)
                    print(source, ' -> ', target)
                except:
                    print('could not create symlink for ', source, sys.exc_info()[0])

def run(csv, sourcePath, targetPath):
    data = readCsvFile(csv)
    splitData = splitByClass(data)
    trainTest = splitTrainTest(splitData)
    makeSymLinks(trainTest, sourcePath, targetPath)

args = sys.argv
if (len(args) < 3):
    print('invalid args - need csv path - input dir - output target dir')
    sys.exit()

run(args[1], args[2], args[3])
