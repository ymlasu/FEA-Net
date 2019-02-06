import tensorflow as tf
import _mask_conv_grad
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
#mask_conv_module=tf.load_op_library(os.path.join(BASE_DIR, '/home/hope-yao/Documents/FEA_Net/thermal/code_tensorflow/tf_ops_cpp/build/lib_mask_conv.so'))
mask_conv_module=tf.load_op_library( '/home/hope-yao/Documents/FEA_Net/thermal/code_tensorflow/tf_ops_cpp/build/lib_mask_conv.so')


def mask_conv(resp, mask, rho):

    return mask_conv_module.mask_conv(resp, mask, rho)