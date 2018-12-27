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
mask_conv_grad_module = tf.load_op_library('./build/lib_mask_conv_grad.so')

rnd_name = 'MaskConv' #+ str(np.random.randint(0, 1E+8))
@ops.RegisterGradient(rnd_name)
def _mask_conv_grad_cc(op, grad):
    """
    The gradient for `inner_product` using the operation implemented in C++.

    :param op: `inner_product` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `inner_product` op.
    :return: gradients with respect to the input of `inner_product`.
    """

    return mask_conv_grad_module.mask_conv_grad(grad, op.inputs[0], op.inputs[1])

# uncomment this and comment the corresponding line above to use the Python
# implementation of the inner product gradient
#@ops.RegisterGradient(rnd_name)
def _mask_conv_grad(op, grad):

    return mask_conv_grad(op, grad)
  

def mask_conv_grad_old(op, grads):
    '''
        gradient of the mask convolution operator
        rearranged and merged the terms for better parallelization
    :param op: inputs of this block
    :param grads: gradient from above layers
    :return: gradient flow of mask and response field pass this layer
    '''

    grad = grads[0] # only partial_resp is propagted to the next layer
    mask = op.inputs[0]  # partial derivative towards mask
    resp = op.inputs[1]  # partial derivative towards response field

    # \frac {\partial error} {\partial mask}
    for i in range(0, resp.shape[1]-1, 1):
        for j in range(0, resp.shape[1]-1, 1):
            # diagnal part + side part, looks correct...
            partial_mask_i_j = grad[0, i, j, 0] * (resp[0, i + 1, j + 1, 0] + resp[0, i, j + 1, 0]/2. + resp[0, i + 1, j, 0]/2.)\
                               + grad[0, i, j + 1, 0] * (resp[0, i + 1, j, 0] + resp[0, i + 1, j + 1, 0]/2. + resp[0, i, j, 0]/2.)\
                               + grad[0, i + 1, j, 0] * (resp[0, i, j + 1, 0] + resp[0, i + 1, j + 1, 0]/2. + resp[0, i, j, 0]/2.)\
                               + grad[0, i + 1, j + 1, 0] * (resp[0, i, j, 0] + resp[0, i, j + 1, 0]/2. + resp[0, i + 1, j, 0]/2.)
            partial_mask = partial_mask_i_j if i==0 and j==0 else tf.stack([partial_mask, partial_mask_i_j])
    partial_mask = tf.reshape(partial_mask, (mask.get_shape().as_list())) # with size the same as mask

    # \frac {\partial error} {\partial resp}
    padded_grad = tf.pad(grad, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    padded_mask = tf.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    for i in range(0, padded_mask.shape[1] - 1, 1):
        for j in range(0, padded_mask.shape[1] - 1, 1):
            # diagnal part + side part, looks strange for the mid nodes
            partial_resp_i_j = padded_grad[0, i, j, 0] * padded_mask[0, i, j, 0]\
                            + padded_grad[0, i, j + 1, 0]* (padded_mask[0, i, j, 0] + padded_mask[0, i, j + 1, 0])/2. \
                            + padded_grad[0, i, j + 2, 0] * padded_mask[0, i, j + 1, 0] \
                            + padded_grad[0, i + 1, j, 0] * (padded_mask[0, i, j + 1, 0] + padded_mask[0, i + 1, j + 1, 0])/2 \
                            + padded_grad[0, i + 1, j + 2, 0]*(padded_mask[0, i + 1, j, 0] + padded_mask[0, i + 1, j + 1, 0])/2.\
                            + padded_grad[0, i + 2, j, 0] * padded_mask[0, i + 1, j, 0] \
                            + padded_grad[0, i + 2, j + 1, 0]*(padded_mask[0, i, j, 0] + padded_mask[0, i + 1, j, 0])/2. \
                            + padded_grad[0, i + 2, j + 2, 0] *padded_mask[0, i + 1, j + 1, 0]
            partial_resp = tf.reshape(partial_resp_i_j,(1,1)) if i == 0 and j == 0 else tf.concat([partial_resp, tf.reshape(partial_resp_i_j, (1, 1))], axis=0)
    partial_resp = tf.reshape(partial_resp, (resp.get_shape().as_list()))

    # the propagated gradient with respect to the first and second argument respectively
    return partial_resp, partial_mask


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
