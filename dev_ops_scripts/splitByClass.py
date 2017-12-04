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

def makeSymLinks(_splitData, _pathFrom, _pathTo):
    os.makedirs(_pathTo, exist_ok=True)
    os.makedirs(_pathTo + '/benign', exist_ok=True)
    os.makedirs(_pathTo + '/malignant', exist_ok=True)

    for index2, images in enumerate(list(_splitData)):
        path_part = 'benign/' if index2 == 0 else 'malignant/'
        for image in images:
            source = _pathFrom + image[0]
            target = _pathTo + path_part + image[0]
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
    makeSymLinks(splitData, sourcePath, targetPath)

args = sys.argv
if (len(args) < 3):
    print('invalid args - need csv path - input dir - output target dir')
    sys.exit()

run(args[1], args[2], args[3])
