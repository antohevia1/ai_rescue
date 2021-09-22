

from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os
import sys
sys.path.append('/home/user4/develop/Keras-FasterRCNN/keras_frcnn')

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras.callbacks import TensorBoard
from keras.utils.layer_utils import get_source_inputs
from keras_frcnn import vgg as nn
import cv2

img_input = Input(shape= (None, None, 3))


shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers


C = config.Config()
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)



model_rpn = Model(img_input, rpn[:2])


# this is a model that holds both the RPN and the classifier, used to load/save weights for the models


try:
    # load_weights by name
    # some keras application model does not containing name
    # for this kinds of model, we need to re-construct model with naming
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)

except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
        https://github.com/fchollet/keras/tree/master/keras/applications')

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])


def format_img_size(img, C):
    img_min_side = float(C.im_size)
    (height, width ,_) = img.shape
    if (width <= height):
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return (img, ratio)

def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

img = cv2.imread('../datasets/2008_000026.jpg')
X, ratio = format_img(img,C)
X = np.transpose(X, (0, 2, 3, 1))
model_rpn.predict(X)
