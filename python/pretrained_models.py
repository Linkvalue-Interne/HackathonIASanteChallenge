from keras import applications
from sys import exit
def inception(input_shape) :
    return applications.InceptionV3(include_top=False,
                                     input_shape=input_shape,
                                     weights='imagenet')

def vgg(input_shape) :
    return applications.VGG19(include_top=False,
                                     input_shape=input_shape,
                                     weights='imagenet')
def vgg16(input_shape) :
    return applications.VGG16(include_top=False,
                                     input_shape=input_shape,
                                     weights='imagenet')

def resnet50(input_shape) :
    return applications.ResNet50(include_top=False,
                                     input_shape=input_shape,
                                     weights='imagenet')


def freezeFeaturesLearning(model):
    for layer in model.layers[:-1]:
        layer.trainable = False

    return model

def model_definition(model_name, input_shape) :
    switcher = {
        "inception" : inception,
        "vgg" : vgg,
        "resnet50" : resnet50,
        "vgg16" : vgg16
    }
    
    exit_func = lambda : exit("Model"+ model_name + "not provided")
    model = switcher.get(model_name, exit_func)
    return freezeFeaturesLearning(model(input_shape))
    
