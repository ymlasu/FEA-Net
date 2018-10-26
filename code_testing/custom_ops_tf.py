'''
This part is for customized differentiable convolution based on element mask
operation in python, will be converted in to CUDA
Hope Yao @2018.05.09
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

def get_D_matrix(elem_mask, conductivity_1, conductivity_2):
    '''

    :param elem_mask:
    :param conductivity_1:
    :param conductivity_2:
    :return:
    '''
    # convolution with symmetric padding at boundary
    padded_elem = tf.pad(elem_mask, [[0, 0], [1, 1], [1, 1], [0, 0]],"SYMMETRIC")
    node_filter = tf.constant([[[[1 / 4.]]] * 2] * 2)
    # first material phase
    node_mask_1 = tf.nn.conv2d(padded_elem, node_filter, strides=[1, 1, 1, 1], padding='VALID')
    # second material phase
    node_mask_2 = tf.ones_like(node_mask_1) - node_mask_1
    d_matrix = node_mask_1 * conductivity_1 + node_mask_2 * conductivity_2
    return d_matrix

def tf_mask_conv(elem_mask, node_resp, diag_coef=1/3., side_coef=1/3.):
    padded_res = tf.pad(node_resp, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    padded_mask = tf.pad(elem_mask, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    # diagnal part
    for i in range(1, padded_res.shape[1]-1, 1):
        for j in range(1, padded_res.shape[1]-1, 1):
            y_diag_i_j = padded_res[0, i-1, j-1, 0] * padded_mask[0, i-1, j-1, 0] \
                     + padded_res[0, i-1, j+1, 0] * padded_mask[0, i-1, j, 0] \
                     + padded_res[0, i+1, j-1, 0] * padded_mask[0, i, j-1, 0] \
                     + padded_res[0, i+1, j+1, 0] * padded_mask[0, i, j, 0]
            y_diag = tf.reshape(y_diag_i_j, (1, 1)) if i == 1 and j == 1 else tf.concat([y_diag, tf.reshape(y_diag_i_j, (1, 1))], axis=0)
    y_diag = tf.reshape(y_diag, (node_resp.get_shape().as_list()))
    # side part
    for i in range(1, padded_res.shape[2]-1, 1):
        for j in range(1, padded_res.shape[1]-1, 1):
            y_side_i_j = padded_res[0, i-1, j, 0] * (padded_mask[0, i-1, j-1, 0] + padded_mask[0, i-1, j, 0]) / 2. \
                     + padded_res[0, i, j-1, 0] * (padded_mask[0, i-1, j-1, 0] + padded_mask[0, i, j-1, 0]) / 2. \
                     + padded_res[0, i, j + 1, 0] * (padded_mask[0, i-1, j, 0] + padded_mask[0, i, j, 0]) / 2. \
                     + padded_res[0, i+1, j, 0] * (padded_mask[0, i, j-1, 0] + padded_mask[0, i, j, 0]) / 2.
            y_side = tf.reshape(y_side_i_j, (1, 1)) if i == 1 and j == 1 else tf.concat([y_side, tf.reshape(y_side_i_j, (1, 1))], axis=0)
    y_side = tf.reshape(y_side, (node_resp.get_shape().as_list()))
    return diag_coef * y_diag + side_coef * y_side

def tf_fast_mask_conv(elem_mask, node_resp, coef):
    '''
    combined two tf_mask_conv together
    :param elem_mask:
    :param node_resp:
    :param coef:
    :return:
    '''
    diag_coef_1, side_coef_1 = coef['diag_coef_1'], coef['diag_coef_1']
    diag_coef_2, side_coef_2 = coef['diag_coef_2'], coef['diag_coef_2']
    padded_resp = tf.pad(node_resp, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    padded_mask = tf.pad(elem_mask, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    # diagnal part
    for i in range(1, padded_resp.shape[1]-1, 1):
        for j in range(1, padded_resp.shape[1]-1, 1):
            y_diag_i_j = padded_resp[0, i-1, j-1, 0] * padded_mask[0, i-1, j-1, 0] \
                     + padded_resp[0, i-1, j+1, 0] * padded_mask[0, i-1, j, 0] \
                     + padded_resp[0, i+1, j-1, 0] * padded_mask[0, i, j-1, 0] \
                     + padded_resp[0, i+1, j+1, 0] * padded_mask[0, i, j, 0]
            y_diag = tf.reshape(y_diag_i_j, (1, 1)) if i == 1 and j == 1 else tf.concat([y_diag, tf.reshape(y_diag_i_j, (1, 1))], axis=0)
    y_diag = tf.reshape(y_diag, (node_resp.get_shape().as_list()))
    # side part
    for i in range(1, padded_resp.shape[2]-1, 1):
        for j in range(1, padded_resp.shape[1]-1, 1):
            y_side_i_j = padded_resp[0, i-1, j, 0] * (padded_mask[0, i-1, j-1, 0] + padded_mask[0, i-1, j, 0]) / 2. \
                     + padded_resp[0, i, j-1, 0] * (padded_mask[0, i-1, j-1, 0] + padded_mask[0, i, j-1, 0]) / 2. \
                     + padded_resp[0, i, j + 1, 0] * (padded_mask[0, i-1, j, 0] + padded_mask[0, i, j, 0]) / 2. \
                     + padded_resp[0, i+1, j, 0] * (padded_mask[0, i, j-1, 0] + padded_mask[0, i, j, 0]) / 2.
            y_side = tf.reshape(y_side_i_j, (1, 1)) if i == 1 and j == 1 else tf.concat([y_side, tf.reshape(y_side_i_j, (1, 1))], axis=0)
    y_side = tf.reshape(y_side, (node_resp.get_shape().as_list()))
    # remaining
    for i in range(1, padded_resp.shape[2]-1, 1):
        for j in range(1, padded_resp.shape[1]-1, 1):
            y_remain_i_j = padded_resp[0, i-1, j, 0]  + padded_resp[0, i, j-1, 0]  \
                         + padded_resp[0, i, j + 1, 0]+ padded_resp[0, i+1, j, 0]
            y_remain = tf.reshape(y_remain_i_j, (1, 1)) if i == 1 and j == 1 else tf.concat([y_remain, tf.reshape(y_remain_i_j, (1, 1))], axis=0)
    y_remain = tf.reshape(y_remain, (node_resp.get_shape().as_list()))
    conv_result = (diag_coef_1 - diag_coef_2) * y_diag + diag_coef_2 * y_remain\
                + (side_coef_1 - side_coef_2) * y_side  + side_coef_2 * y_remain
    return conv_result

def tf_faster_mask_conv(elem_mask, node_resp, coef):
    '''
    combined side and diag part of tf_fast_mask_conv together
    :param elem_mask:
    :param node_resp:
    :param coef:
    :return:
    '''
    diag_coef_1, side_coef_1 = coef['diag_coef_1'], coef['diag_coef_1']
    diag_coef_2, side_coef_2 = coef['diag_coef_2'], coef['diag_coef_2']
    diag_coef_diff = diag_coef_1 - diag_coef_2
    side_coef_diff = side_coef_1 - side_coef_2
    padded_resp = tf.pad(node_resp, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    padded_mask = tf.pad(elem_mask, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    for i in range(1, padded_resp.shape[1]-1, 1):
        for j in range(1, padded_resp.shape[1]-1, 1):
            conv_result_i_j = \
            padded_mask[0, i - 1, j - 1, 0] * \
            (
                    padded_resp[0, i - 1, j - 1, 0] * diag_coef_diff
                    + (padded_resp[0, i - 1, j, 0] + padded_resp[0, i, j - 1, 0]) / 2. * side_coef_diff
            ) + \
            padded_mask[0, i - 1, j, 0] * \
            (
                    padded_resp[0, i - 1, j + 1, 0] * diag_coef_diff
                    + (padded_resp[0, i - 1, j, 0] + padded_resp[0, i, j + 1, 0]) / 2. * side_coef_diff
            ) + \
            padded_mask[0, i, j - 1, 0] * \
            (
                    padded_resp[0, i + 1, j - 1, 0] * diag_coef_diff
                    + (padded_resp[0, i, j - 1, 0] + padded_resp[0, i + 1, j, 0]) / 2. * side_coef_diff
            ) + \
            padded_mask[0, i, j, 0] * \
            (
                    padded_resp[0, i + 1, j + 1, 0] * diag_coef_diff
                    + (padded_resp[0, i, j + 1, 0] + padded_resp[0, i + 1, j, 0]) / 2. * side_coef_diff
            ) + \
            (diag_coef_2+diag_coef_2) * \
            (
                    padded_resp[0, i - 1, j, 0] + padded_resp[0, i, j - 1, 0]
                    + padded_resp[0, i, j + 1, 0] + padded_resp[0, i + 1, j, 0]
            )
            conv_result = tf.reshape(conv_result_i_j, (1, 1)) if i == 1 and j == 1 else tf.concat([conv_result, tf.reshape(conv_result_i_j, (1, 1))], axis=0)
    LU_u = tf.reshape(conv_result, (node_resp.get_shape().as_list())) + (diag_coef_2 + side_coef_2) * node_resp

    tmp = {
        'LU_u': LU_u
    }
    return tmp

def tf_conv_2phase(elem_mask, node_resp, coef_dict):
    '''

    :param elem_mask:
    :param node_resp:
    :param diag_coef:
    :param side_coef:
    :return:
    '''
    conductivity_1 = coef_dict['conductivity_1']
    conductivity_2 = coef_dict['conductivity_2']
    elem_mask_1 = elem_mask
    diag_coef_1 = side_coef_1 = conductivity_1/ 3.
    # first material phase on LU part
    LU_u_1 = tf_mask_conv(elem_mask_1, node_resp, diag_coef=diag_coef_1, side_coef=side_coef_1)
    elem_mask_2 = tf.ones_like(elem_mask) - elem_mask
    diag_coef_2 = side_coef_2 = conductivity_2/ 3.
    # second material phase on LU part
    LU_u_2 = tf_mask_conv(elem_mask_2, node_resp, diag_coef=diag_coef_2, side_coef=side_coef_2)
    LU_u = LU_u_1 + LU_u_2
    tmp = {
           'LU_u': LU_u,
           'LU_u_1': LU_u_1,
           'LU_u_2': LU_u_2,
           }
    return tmp

def mask_conv_grad(op, grads):
    '''
        gradient of the mask convolution operator
    :param op: inputs of this block
    :param grads: gradient from above layers
    :return: gradient flow of mask and response field pass this layer
    '''

    grad = grads[0] # only partial_resp is propagted to the next layer
    mask = op.inputs[0]  # partial derivative towards mask
    resp = op.inputs[1]  # partial derivative towards response field

    # \frac {\partial error} {\partial mask}
    # diagnal part
    for i in range(0, resp.shape[1]-1, 1):
        for j in range(0, resp.shape[1]-1, 1):
            partial_mask_i_j = grad[0, i, j, 0] * resp[0, i + 1, j + 1, 0]\
                               + grad[0, i, j + 1, 0] * resp[0, i + 1, j, 0]\
                               + grad[0, i + 1, j, 0] * resp[0, i, j + 1, 0]\
                               + grad[0, i + 1, j + 1, 0] * resp[0, i, j, 0]
            partial_mask_diag = partial_mask_i_j if i==0 and j==0 else tf.stack([partial_mask_diag, partial_mask_i_j])
    partial_mask_diag = tf.reshape(partial_mask_diag, (mask.get_shape().as_list()))
    # side part
    for i in range(0, resp.shape[1]-1, 1):
        for j in range(0, resp.shape[1]-1, 1):
            partial_mask_i_j = grad[0, i, j, 0] * (resp[0, i, j + 1, 0]+resp[0, i + 1, j, 0])/2\
                               + grad[0, i, j + 1, 0] * (resp[0, i + 1, j + 1, 0]+resp[0, i, j, 0])/2\
                               + grad[0, i + 1, j, 0] * (resp[0, i + 1, j + 1, 0]+resp[0, i, j, 0])/2\
                               + grad[0, i + 1, j + 1, 0] * (resp[0, i, j + 1, 0]+resp[0, i + 1, j, 0])/2
            partial_mask_side = partial_mask_i_j if i == 0 and j == 0 else tf.stack([partial_mask_side, partial_mask_i_j])
    partial_mask_side = tf.reshape(partial_mask_side, (mask.get_shape().as_list()))
    partial_mask = partial_mask_diag + partial_mask_side # with size the same as mask

    # \frac {\partial error} {\partial resp}
    padded_grad = tf.pad(grad, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    padded_mask = tf.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    # diagnal part
    for i in range(0, padded_mask.shape[1] - 1, 1):
        for j in range(0, padded_mask.shape[1] - 1, 1):
            partial_resp_i_j = padded_mask[0, i, j, 0] * padded_grad[0, i, j, 0] \
                               + padded_mask[0, i, j + 1, 0] * padded_grad[0, i, j + 2, 0] \
                               + padded_mask[0, i + 1, j, 0] * padded_grad[0, i + 2, j, 0] \
                               + padded_mask[0, i + 1, j + 1, 0] * padded_grad[0, i + 2, j + 2, 0]
            partial_resp_diag = tf.reshape(partial_resp_i_j,(1,1)) if i == 0 and j == 0 else tf.concat([partial_resp_diag, tf.reshape(partial_resp_i_j, (1, 1))], axis=0)
    partial_resp_diag = tf.reshape(partial_resp_diag, (resp.get_shape().as_list()))
    # side part
    for i in range(0, padded_mask.shape[1]-1, 1):
        for j in range(0, padded_mask.shape[1]-1, 1):
            partial_resp_i_j = (padded_mask[0, i, j, 0]+padded_mask[0, i, j + 1, 0])/2 * padded_grad[0, i, j + 1, 0]\
                               +(padded_mask[0, i, j + 1, 0]+padded_mask[0, i + 1, j + 1, 0])/2 * padded_grad[0, i + 1, j, 0]\
                               +(padded_mask[0, i + 1, j, 0]+padded_mask[0, i + 1, j + 1, 0])/2 * padded_grad[0, i + 1, j + 2, 0]\
                               +(padded_mask[0, i + 1, j, 0]+padded_mask[0, i, j, 0])/2 * padded_grad[0, i + 2, j + 1, 0]
            partial_resp_side = tf.reshape(partial_resp_i_j,(1,1)) if i == 0 and j == 0 else tf.concat([partial_resp_side, tf.reshape(partial_resp_i_j, (1, 1))], axis=0)
    partial_resp_side = tf.reshape(partial_resp_side, (resp.get_shape().as_list()))
    partial_resp = partial_resp_diag + partial_resp_side # with size the same as resp

    return partial_mask, partial_resp# the propagated gradient with respect to the first and second argument respectively

def fast_mask_conv_grad(op, grads):
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
    return partial_mask, partial_resp

def tf_mask_conv_full(elem_mask, node_resp, coef_dict ):
    conductivity_1 = coef_dict['conductivity_1']
    conductivity_2 = coef_dict['conductivity_2']
    # mixed material phase on LU matrix
    # LU_u = tf_conv_2phase(elem_mask, node_resp, conductivity_1, conductivity_2)
    LU_u = tf_faster_mask_conv(elem_mask, node_resp, coef_dict)
    # mixed material phase on D matrix
    center_coef_1 = conductivity_1 * (-8./3.)
    center_coef_2 = conductivity_2 * (-8./3.)
    d_matrix = get_D_matrix(elem_mask, center_coef_1, center_coef_2)
    DLU_u = LU_u['LU_u'] + d_matrix * node_resp

    tmp = {
           'LU_u': LU_u['LU_u'],
           # 'LU_u_1': LU_u['LU_u_1'],
           # 'LU_u_2': LU_u['LU_u_2'],
           'd_matrix': d_matrix,
           'DLU_u': DLU_u,
           }
    return tmp

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def mask_conv(mask, resp, diag_coef, side_coef, name=None):

    with ops.op_scope([mask, resp], name, "masked_conv") as name:
        z = py_func(np_mask_conv,
                    [mask, resp],#, diag_coef, side_coef
                    [tf.float32],
                    name=name,
                    grad=fast_mask_conv_grad)  # <-- here's the call to the gradient
        return z[0]


if __name__ == '__main__':
    n_elem_x = n_elem_y = 64
    # build graph
    conductivity_1_pl = tf.placeholder(tf.float32, shape=())
    conductivity_2_pl = tf.placeholder(tf.float32, shape=())
    elem_mask_pl = tf.placeholder(tf.float32, shape=(1, n_elem_x, n_elem_y, 1))
    node_resp_pl = tf.placeholder(tf.float32, shape=(1, n_elem_x+1, n_elem_y+1, 1))
    coef_dict = {
        'conductivity_1': conductivity_1_pl,
        'conductivity_2': conductivity_2_pl,
        'diag_coef_1': conductivity_1_pl * 1 / 3.,
        'side_coef_1': conductivity_1_pl * 1 / 3.,
        'diag_coef_2': conductivity_2_pl * 1 / 3.,
        'side_coef_2': conductivity_2_pl * 1 / 3.
    }
    if 0:
        resp = tf_conv_2phase(elem_mask_pl, node_resp_pl,coef_dict)
    else:
        resp = tf_mask_conv_full(elem_mask_pl, node_resp_pl, coef_dict)
    gr_mask, gr_resp = tf.gradients(resp['LU_u'], [elem_mask_pl, node_resp_pl])



    # initial graph
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    ## training starts ###
    from data_loader import load_data_elem
    resp_gt, load_gt, elem_mask, conductivity_1, conductivity_2 = load_data_elem(case=-1)
    feed_dict = {elem_mask_pl: elem_mask,
                 node_resp_pl: resp_gt,
                 conductivity_1_pl: conductivity_1,
                 conductivity_2_pl: conductivity_2}
    import matplotlib.pyplot as plt
    plt.imshow(np.squeeze(sess.run(resp['DLU_u'], feed_dict)))
    plt.colorbar()
    plt.show()
    result = sess.run([resp_1, resp_2, resp, gr_mask, gr_resp], feed_dict)
    print('done')

