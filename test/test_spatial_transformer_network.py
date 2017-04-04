from __future__ import print_function
import os
import pdb
import numpy as np
from PIL import Image
import scipy.misc as sc
import matplotlib.pyplot as plt

# Import Tensorflow
import tensorflow as tf

# Import from Keras
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.layers import Dense, Activation, Flatten, Input, Convolution2D, MaxPooling2D

# Import STN
from spatial_transformer_network import STN

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"]='2'

#
plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['image.interpolation'] = 'nearest'

# Load an example image.
height = 96
width = 96
image = Image.open("Lena.png")
image = image.resize((height, width))
image = np.asarray(image) / 255.
image = image.astype('float32')

# To test the STN, we set up our predicted theta as we want,
# say now we want to flip the image, and downsample the image.
downsample = 1.
w = np.zeros((20, 6))
b = np.zeros((6,))
b[0] = -1.
b[1] = 0.
b[4] = 1.
print('Transformation matrix:\n{}'.format(b.reshape(2, 3)))

# --------------------------
# Build a simple model.
# To test the STN layer, simply set the transformation matrix manually.
# --------------------------
inputdata = Input(batch_shape=(1, height, width, 3))
fla = Flatten()(inputdata)
# Loc Networks.
dense1 = Dense(20)(fla)
dense2 = Dense(6, weights=[w, b])(dense1)  # Set the parameters
# STN Networks, give original image and theta as input.
trans = STN(downsample_factor=downsample)([inputdata, dense2])
# output:
# trans --> the image after transformation.
# dense2 --> the matrix we manually set.
model = Model(input=inputdata, output=[trans, dense2])

# --------------------------
#       Provide Data.
# --------------------------
A = np.expand_dims(image, axis=0)  # pad batch_size dimension.
[out, theta] = model.predict(A)
# pdb.set_trace()
print(theta[0])
print('Original image size : {}'.format(A.shape))
print('Transformed image size : {}'.format(out.shape))

f, axes = plt.subplots(1, 2, sharey=True)
axes[0].imshow(A[0])
axes[0].set_title('Original Image')
axes[1].imshow(out[0])
axes[1].set_title('Transformed Image')
plt.show()
