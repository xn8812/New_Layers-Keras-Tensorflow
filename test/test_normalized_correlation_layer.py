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

# New layer to test
from normalized_correlation_layer import Normalized_Correlation_Layer

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
K.set_image_dim_ordering('tf')  # Tensorflow dimension ordering in this code

def BuildModel(dim, patch_size=15, stride=(5, 5)):
    '''
    :param dim: input shape.
    :param patch_size: patch_size when do correlation.
    :param stride: stride.
    
    Number of channels in the output depends on patch_size and stride.
    
    As a simple example, calculate the correlation between two input images.
    Can also do correlation between two intermediate feature maps.
    Two inputs must have the same size, only border_mode = 'valid' is supported,
    so, if want to get output of same size as input, please do padding before feeding to the layer.
    '''

    input_a = Input(shape=dim)
    input_b = Input(shape=dim)
    #
    y_corr = Normalized_Correlation_Layer(patch_size=patch_size, stride=stride)([input_a, input_a])
    #
    model = Model(input=[input_a, input_b], output=y_corr)
    return model

# Load an example image.
height = 96
width = 96
image = Image.open("Lena.png")
image = image.resize((height, width))
image = np.asarray(image) / 255.
image = image.astype('float32')

image_1 = np.expand_dims(image, axis=0)
image_2 = np.expand_dims(image, axis=0)

# Build Model
model = BuildModel(dim=(height, width, 3))
out = model.predict([image_1, image_2])

print('Input shape is : {}'.format(image_1.shape))
print('Output shape is : {}'.format(out.shape))

plt.imshow(out[0,:,:,0])
plt.show()
