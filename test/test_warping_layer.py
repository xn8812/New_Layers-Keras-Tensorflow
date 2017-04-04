from __future__ import print_function
import os
import pdb
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import scipy.misc as sc
import matplotlib.pyplot as plt

# Import Keras Layers
from keras.models import Model
from keras.layers import Input

# Import Spatial Warping Layer
from spatial_warping_networks import SWN

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"]='2'

def buildModel(dim):
    # As a simple example, the flow field is provided to test the layer.
    # But this flow_field can be output from other streams.
    #
    inpdata = Input(shape=(dim[0], dim[1], 3))
    flow_field = Input(shape=(dim[0], dim[1], 2))

    # Warping Networks, give original image and flow field as input.
    trans = SWN()([inpdata, flow_field])
    #
    model = Model(input=[inpdata, flow_field], output=trans)
    return model

# Load an example image.
height = 96
width = 96
batch_size = 1
image = Image.open("Lena.png")
image = image.resize((height, width))
image = np.asarray(image) / 255.
image = image.astype('float32')

# Make a test flow_field.
image = np.expand_dims(image, axis=0)
flow_ = np.zeros((batch_size, height, width, 2))
flow_[:, :, :, 0] = -20
flow_[:, :, :, 1] = -20

model = buildModel(dim=(height, width, 3))

out = model.predict([image, flow_])

f, axes = plt.subplots(1, 2, sharey=True)
axes[0].imshow(image[0])
axes[0].set_title('Original Image')
axes[1].imshow(out[0])
axes[1].set_title('Transformed Image')
plt.show()
