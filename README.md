# New_Layers-Keras-Tensorflow-

In this repository, I will add some layers from recent papers, 
they are tested on Keras-1.2.2.  

The layers include:

-- Spatial Transformer Networks (STN).

[1] Spatial Transformer Networks. 

https://arxiv.org/abs/1506.02025

-- Spatial Warping Layer.
This is based on STN, instead of predicting transformation matrices, this layer predict the x,y displacements.

-- Separable RNN Layer.

Seperate the RNN into two parts, convolution + recurrence.
This layer can accept input directly from linear convolution.

[2] https://openreview.net/forum?id=rJJRDvcex&noteId=rJJRDvcex

-- RNN with Layer Normalization.

[3] Layer Normalization. 

https://arxiv.org/abs/1607.06450

-- Subpixel Upsampling.

[4] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network.

https://arxiv.org/abs/1609.05158

-- Dynamic Filter Layer
[5] Dynamic Filter Networks. 

https://arxiv.org/abs/1605.09673

-- Correlation Layer
[6] Convolutional neural network architecture for geometric matching. 

https://arxiv.org/pdf/1703.05593.pdf





