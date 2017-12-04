# IA Sante Challenge - Team Linkvalue

[Challenge Home Page](http://iasantechallenge.bemyapp.com/)

## Environment

This repo was made to run with : 

* Python 3.5
* Keras 2.0
* Tensorflow 1.0.1
* 

## Usage

You first have to edit the [config file](python/config.ini) to set the basic parameters (location of your training data, future outputs...).
And the, the generic usage of `main.py` file is as follow : 

```
cd python/
python main.py <mode> <neural-net-name> (OPTIONAL : <weights-file.h5>) 
```

### Train

```
cd python/
python main.py train <neural-net-name> (OPTIONAL : <weights-file.h5>) 
```

**For instance**, to run a train with inception V3 using weights of the model trained on ImageNet, just run `python main.py train inception`

### Predict

To run a prediction, you need to pass a *weights file* as an argument.
```
cd python/
python main.py predict <neural-net-name> <weights-file.h5>
```

**For instance**, to run a predict with VGG using weights of the model you just trained, just run `python main.py predict vgg /sharedfiles/outputs/model/vgg.h5`

### Preprocessing (ROI & data augmentation)

Before runing the trains, we ran some preprocessing our images using the [preprocessing file](python/preprocessing.py) like this : 

```
cd python/
python preprocessing.py <input_folder> <output_folder> <im_size> <pool_size>
```

## Some sources that inspired us

Find some documentation useful to start with [here](docs/docs.md)

