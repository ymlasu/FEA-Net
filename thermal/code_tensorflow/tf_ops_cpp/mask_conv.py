import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
mask_conv_module=tf.load_op_library(os.path.join(BASE_DIR, './build/lib_mask_conv.so'))


def mask_conv(resp, mask):

    return mask_conv_module.mask_conv(resp, mask)