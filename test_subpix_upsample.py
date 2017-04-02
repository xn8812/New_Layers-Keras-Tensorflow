import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model
from subpix_upsample import Subpix_denseUP

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"]='2'

# Create Toy Data
height = 8
width = 8
# I is of size : 8x8x4
I = np.concatenate([np.ones((height, width, 1))*0, 20*np.ones((height, width, 1)),
                30*np.ones((height, width, 1)), 40*np.ones((height, width, 1))], axis=-1)
inp = np.zeros((2, height, width, 4))

inp[0] = I
inp[1] = I

# Build Model
inputdata = Input(batch_shape=(2, height, width, 4))

# ratio: refers to the up-sampling factor,
# need to guarantee that ratio**2 must be divisible by the nb_channels of the feature maps.
# example here :
# input shape is : batch_size x 8 x 8 x 4
# output shape will be : batch_size x 16 x 16 x 1  where (1 = 4 // ratio**2)
x = Subpix_denseUP(ratio=2)(inputdata)
model = Model(input=inputdata, output=x)

# Data Flow
y = model.predict(inp)
print('Shape before upsampling: {}'.format(inp.shape))
print('Shape after upsampling: {}'.format(y.shape))
y = np.squeeze(y)
plt.imshow(y[0])
plt.show()
