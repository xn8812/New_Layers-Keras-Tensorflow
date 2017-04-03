from __future__ import print_function
import os
import pdb
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import Tensorflow
import tensorflow as tf

# Import Keras Layers
from keras import backend as K
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution1D

# New layer to test
from separable_RNN import Separable_SimpleRNN


# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
K.set_image_dim_ordering('tf')  # Tensorflow dimension ordering in this code


def BuildModel(dim):
    '''
    :param dim: input shape.
    
    The Separable_SimpleRNN can be added after linear convolution.
    Make sure the nb_filter in the Convolution layer is equivalent to output_dim in RNN
    '''
    inp = Input(shape=dim)
    #
    x_conv = Convolution1D(nb_filter=16, filter_length=5, border_mode='same')(inp)
    x_rnn = Separable_SimpleRNN(output_dim=16, activation='relu')(x_conv)
    #
    model = Model(input=inp, output=x_rnn)
    return model

# Load an example image.
height = 96
width = 96
image = Image.open("Lena.png")
image = image.resize((height, width))
image = np.asarray(image) / 255.
image = image.astype('float32')

# get a sequence of length 32
seq_1 = np.expand_dims(image[0, :32, :], axis=0)

# Build Model
model = BuildModel(dim=(32, 3))
out = model.predict(seq_1)
print('output: {}'.format(out))
