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
import sys 
sys.path.append('../src')
from spatial_warping_network import SWN

from tensorflow.python import debug as tf_debug
import keras.backend as K
# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def buildModel(dim):
    # As a simple example, the flow field is provided to test the layer.
    # But this flow_field can be output from other streams.
    #
    inpdata = Input(shape=(dim[0], dim[1], 3))
    flow_field = Input(shape=(dim[0], dim[1], 2))

    # Warping Networks, give original image and flow field as input.
    trans = SWN()([inpdata, flow_field])
    #
    model = Model(inputs=[inpdata, flow_field], outputs=trans)
    return model

def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res 

# Load an example image.

#sess = K.get_session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#K.set_session(sess)

# height = 320
# width = 640
# batch_size = 1
# image = Image.open("lena.png")
# image = image.resize(( width, height))
# #pdb.set_trace()
# image = np.array(image) 

# flow_ = np.zeros((batch_size, height, width, 2))

# flow_[:, :, :, 0] = -180.
# flow_[:, :, :, 1] = -180.
# pdb.set_trace()
# out2 = warp_flow(image, flow_[0].astype('float32'))

# image = image.astype('float32')

# # Make a test flow_field.
# image = np.expand_dims(image, axis=0)


# Image.fromarray((out2).astype('uint8')).show()

# flow_[:, :, :, 0] = -180. / (width/2)
# flow_[:, :, :, 1] = -180. / (height/2)

# model = buildModel(dim=(height, width, 3))

# out = model.predict([image, flow_])

# Image.fromarray((image[0]).astype('uint8')).show()
# Image.fromarray((out[0]).astype('uint8')).show()

flow_path = '/mnt/ilcompf2d0/user/nxu/video_seg/tmp/flownet2/scripts'
flow_name = 'res.flo'
flow_ = readFlow(os.path.join(flow_path,flow_name))

height, width, _ = flow_.shape
batch_size = 1

im1 = Image.open(os.path.join(flow_path, '50.png'))
im1.show()
image = np.asarray(im1).astype('float32')
image = np.expand_dims(image, axis=0)

flow_ = np.expand_dims(flow_, axis=0)
flow_[:, :, :, 0] = flow_[:, :, :, 0] / width * 2
flow_[:, :, :, 1] = flow_[:, :, :, 1] / height * 2

model = buildModel(dim=(height, width, 3))
out = model.predict([image, flow_])
Image.fromarray((out[0]).astype('uint8')).show()