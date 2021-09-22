import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import Input,Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras_frcnn import data_augment
from keras.utils import generic_utils
from keras.callbacks import TensorBoard
from tensorflow.keras.backend import image_data_format
import cv2


# tensorboard
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                  default="pascal_voc")
parser.add_option("-n", "--num_rois", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg, xception, inception_resnet_v2 or resnet50.", default='vgg')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--num_epochs", dest="num_epochs", help="Number of epochs.", default=2000)
parser.add_option("--config_filename", dest="config_filename",
                  help="Location to store all the metadata related to the training (to be used when testing).",
                  default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()

if not options.train_path:   # if filename is not given
    parser.error('Error: path to training data must be specified. Pass --path to command line')

if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

C.model_path = options.output_weight_path
C.num_rois = int(options.num_rois)

if options.network == 'vgg':
    C.network = 'vgg'
    from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
    from keras_frcnn import resnet as nn
    C.network = 'resnet50'
elif options.network == 'xception':
    from keras_frcnn import xception as nn
    C.network = 'xception'
elif options.network == 'inception_resnet_v2':
    from keras_frcnn import inception_resnet_v2 as nn
    C.network = 'inception_resnet_v2'
else:
    print('Not a valid model')
    raise ValueError

# check if weight path was passed via command line
if options.input_weight_path:

    C.base_net_weights = options.input_weight_path
else:
    # set the path to weights based on backend and model
    C.base_net_weights = nn.get_weight_path()



config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print(('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename)))



#########################images#################
all_imgs, classes_count, class_mapping = get_data(options.train_path)

# bg
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in list(class_mapping.items())}

print('Training images per class:')
pprint.pprint(classes_count)
print(('Num classes (including bg) = {}'.format(len(classes_count))))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print(('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename)))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)
print(num_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
test_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print(('Num train samples {}'.format(len(train_imgs))))
print(('Num test samples {}'.format(len(test_imgs))))

# groundtruth

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, image_data_format(), mode='train')
print('data_gen_train generated')
data_gen_test = data_generators.get_anchor_gt(test_imgs, classes_count, C, nn.get_img_output_length, image_data_format(), mode='test')
print('data_gen_test generated')
############################################

if image_data_format() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

# input placeholder
img_input = Input(shape=input_shape_img)
print('img_input generated')
roi_input = Input(shape=(None, 4))
print('roi_input generated')

# base network(feature extractor) (resnet, VGG, Inception, Inception Resnet V2, etc)
shared_layers = nn.nn_base(img_input, trainable=True)
print('shared_layers generated')

# define the RPN, built on the base layers
# RPN
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)
print('rpn generated')
model_rpn = Model(img_input, rpn[:2])

try:
    # load_weights by name
    # some keras application model does not containing name
    # for this kinds of model, we need to re-construct model with naming
    print(('loading weights from {}'.format(C.base_net_weights)))
    model_rpn.load_weights(C.base_net_weights, by_name=True)

except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
        https://github.com/fchollet/keras/tree/master/keras/applications')

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

img = cv2.imread('../../../datasets/2012_004310.jpg')
X, ratio = format_img(img,C)
X = np.transpose(X, (0, 2, 3, 1))

#print(data_gen_train[0])
#X, Y, img_data = data_gen_train
#print(X)

#img_data_aug, x_img = data_augment.augment(img, C, augment=True)


P_rpn = model_rpn.predict(X)
print(X.shape)
print(P_rpn[0].shape)
print(P_rpn[1].shape)
print(image_data_format())

R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, 'tf', use_regr=True, overlap_thresh=0.7, max_boxes=300)
## note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
#X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)
#
print(R[0])
print(R[1])
print(R[2])
print(R[3])
#print(R[1].shape)
