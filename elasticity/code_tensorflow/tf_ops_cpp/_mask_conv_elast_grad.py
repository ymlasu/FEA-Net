#!/usr/bin/env python3
"""
Gradients for inner product.

.. moduleauthor:: David Stutz
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
mask_conv_grad_module = tf.load_op_library('/home/hope-yao/Documents/FEA_Net/elasticity/code_tensorflow/tf_ops_cpp/build/lib_mask_conv_elast_grad.so')

rnd_name = 'MaskconvElast' #+ str(np.random.randint(0, 1E+8))
@ops.RegisterGradient(rnd_name)
def _maskconv_elast_grad_cc(op, grad):
    """
    The gradient for `inner_product` using the operation implemented in C++.

    :param op: `inner_product` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `inner_product` op.
    :return: gradients with respect to the input of `inner_product`.
    """

    return mask_conv_grad_module.maskconv_elast_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])

# uncomment this and comment the corresponding line above to use the Python
# implementation of the inner product gradient
#@ops.RegisterGradient(rnd_name)
def _mask_conv_grad(op, grad):

    return mask_conv_grad(op, grad)

def mask_conv_grad(op, grads):
    '''
        gradient of the mask convolution operator
        rearranged and merged the terms for better parallelization
    :param op: inputs of this block
    :param grads: gradient from above layers
    :return: gradient flow of mask and response field pass this layer
    '''

    grad = grads # only partial_resp is propagted to the next layer
    resp = op.inputs[0]  # partial derivative towards mask
    mask = op.inputs[1]  # partial derivative towards response field

    rho_1 = 16.
    rho_2 = 205.
    diag_coef_1 = rho_1 / 3.
    side_coef_1 = rho_1 / 3.
    diag_coef_2 = rho_2 / 3.
    side_coef_2 = rho_2 / 3.
    diag_coef_diff = diag_coef_1 - diag_coef_2
    side_coef_diff = side_coef_1 - side_coef_2
    partial_mask = tf.zeros((1,65+2,65+2,1))
    partial_resp = tf.zeros((1,66+2,66+2,1))

    for i in range(1, mask.shape[1], 1):
        for j in range(1, mask.shape[2], 1):
            partial_mask[0, i-1, j-1, 0] =  grad[0,i-1,j-1,0] * (resp[0,i-1,j-1,0] * diag_coef_diff + (resp[0,i-1,j,0] + resp[0,i,j-1,0])/2. * side_coef_diff)
            partial_mask[0, i-1, j, 0] = grad[0,i-1,j,0] * (resp[0,i-1,j+1,0] * diag_coef_diff + (resp[0,i,j+1,0] + resp[0,i-1,j,0])/ 2. * side_coef_diff)
            partial_mask[0, i, j-1, 0] = grad[0,i,j-1,0] * (resp[0,i+1,j-1,0] * diag_coef_diff + (resp[0,i+1,j,0] + resp[0,i,j-1,0])/ 2. * side_coef_diff)
            partial_mask[0, i, j, 0] = grad[0,i,j,0] * (resp[0,i+1,j+1,0] * diag_coef_diff + (resp[0,i+1,j,0] + resp[0,i,j+1,0])/ 2. * side_coef_diff)


    for i in range(1, resp.shape[1]-1, 1):
        for j in range(1, resp.shape[2]-1, 1):
            partial_resp[0, i-1, j-1, 0] = grad[0, i-1, j-1, 0] * (mask[0, i-1, j-1, 0] * diag_coef_diff + diag_coef_2)
            partial_resp[0, i-1, j, 0] = grad[0, i-1, j, 0] * ((mask[0, i-1, j-1, 0]+mask[0, i-1, j, 0])/2 * side_coef_diff + side_coef_2)
            partial_resp[0, i-1, j+1, 0] = grad[0, i-1, j+1, 0] * (mask[0, i-1, j, 0] * diag_coef_1 + (1-mask[0, i-1, j, 0]) * diag_coef_2)

            partial_resp[0, i, j-1, 0] = grad[0, i, j-1, 0] * ((mask[0, i-1, j-1, 0]+mask[0, i, j-1, 0])/2 * side_coef_diff + side_coef_2)
            partial_resp[0, i, j+1, 0] = grad[0, i, j+1, 0] * ((mask[0, i-1, j, 0]+mask[0, i, j, 0])/2 * side_coef_diff + side_coef_2)

            partial_resp[0, i+1, j-1, 0] = grad[0, i+1, j-1, 0] * (mask[0, i, j-1, 0] * diag_coef_diff + diag_coef_2)
            partial_resp[0, i+1, j, 0] = grad[0, i+1, j, 0] * ((mask[0, i, j-1, 0]+mask[0, i, j, 0])/2 * side_coef_diff + side_coef_2)
            partial_resp[0, i+1, j+1, 0] = grad[0, i+1, j+1, 0] * (mask[0, i, j, 0] * diag_coef_1 + (1-mask[0, i, j, 0]) * diag_coef_2 )

    # the propagated gradient with respect to the first and second argument respectively
    return partial_resp, partial_mask
