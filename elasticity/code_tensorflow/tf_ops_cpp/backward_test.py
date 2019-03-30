import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import scipy.io as sio
import unittest
import numpy as np
import tensorflow as tf
# import _mask_conv_grad
# mask_conv_module = tf.load_op_library('build/lib_mask_conv_elast.so')


def load_data_elem_s12():
    num_node = 13
    data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/2D_elastic_xy_fixed.mat')
    rho = [230 / 1e3, 0.36, 200 / 1e3, 0.25]
    u_img = np.concatenate([data['ux'].reshape(1, 13, 13, 1), data['uy'].reshape(1, 13, 13, 1)], -1) * 1e6
    f_img = -1 * np.concatenate([data['fx'].reshape(1, 13, 13, 1), data['fy'].reshape(1, 13, 13, 1)], -1) / 1e6
    mask = data['mask'].reshape(1, num_node - 1, num_node - 1, 1)
    return num_node, mask, u_img, f_img, rho

def boundary_padding(x):
    ''' special symmetric boundary padding '''
    left = x[:, :, 1:2, :]
    right = x[:, :, -2:-1, :]
    upper = tf.concat([x[:, 1:2, 1:2, :], x[:, 1:2, :, :], x[:, 1:2, -2:-1, :]], 2)
    down = tf.concat([x[:, -2:-1, 1:2, :], x[:, -2:-1, :, :], x[:, -2:-1, -2:-1, :]], 2)
    padded_x = tf.concat([left, x, right], 2)
    padded_x = tf.concat([upper, padded_x, down], 1)
    return padded_x

def boundary_correct(x, num_node):
    mask = np.ones((1,num_node,num_node,1))
    mask[:, 0, :, :] /= 2
    mask[:, -1, :, :] /= 2
    mask[:, :, 0, :] /= 2
    mask[:, :, -1, :] /= 2
    return x*mask

if __name__ == '__main__':
    if 0:
        x_pl = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        w_pl = tf.constant([[10], [20]], dtype=tf.float32)
        masked_res_tf = tf.matmul(x_pl, w_pl)
        # reference Jacobian
        with tf.Session():
            partial_u_ref, partial_w_ref = tf.test.compute_gradient([x_pl, w_pl], [[2, 2], [2, 1]],
                                                                    masked_res_tf, [2, 1], delta=1e-3)
    # num_node, mask, u_img, f_img, rho1, rho2 = get_data()
    num_node, mask, u_img, f_img, rho = load_data_elem_s12()#get_data()

    x_pl = tf.constant(u_img, dtype=tf.float32)
    m_pl = tf.constant(mask, dtype=tf.float32)
    w_pl = tf.constant(rho, dtype=tf.float32)
    padded_input =boundary_padding(x_pl)
    padded_mask = boundary_padding(m_pl)
    from mask_elast_conv import *
    masked_res_tf = mask_conv(padded_input, padded_mask, w_pl)
    # masked_res_tf = boundary_correct(masked_res_tf, num_node)

    # reference Jacobian
    with tf.Session():
        partial_u_ref, partial_m_ref , partial_w_ref = tf.test.compute_gradient([x_pl, m_pl, w_pl],
                                                                               [[1, num_node, num_node, 2],
                                                                                [1, num_node-1, num_node-1, 1],
                                                                                [4]],
                                                                               masked_res_tf,
                                                                               [1, 13, 13, 2],
                                                                               delta=1e-2)

    numshow = 1000
    import matplotlib.pyplot as plt
    plt.subplot(2, 3, 1)
    plt.imshow(partial_u_ref[0][:numshow, :numshow])
    plt.colorbar()
    plt.subplot(2, 3, 2)
    plt.imshow(partial_u_ref[1][:numshow, :numshow])
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.imshow(partial_u_ref[0][:numshow, :numshow] - partial_u_ref[1][:numshow, :numshow])
    plt.colorbar()
    plt.subplot(2, 3, 4)
    plt.imshow(partial_m_ref[0][:numshow, :numshow])
    plt.colorbar()
    plt.subplot(2, 3, 5)
    plt.imshow(partial_m_ref[1][:numshow, :numshow])
    plt.colorbar()
    plt.subplot(2, 3, 6)
    plt.imshow(partial_m_ref[0][:numshow, :numshow] - partial_m_ref[1][:numshow, :numshow])
    plt.colorbar()
    plt.show()

    numshow = 10000
    plt.figure()
    plt.subplot(4, 3, 1)
    plt.plot(partial_w_ref[0][0, :numshow])
    plt.subplot(4, 3, 2)
    plt.plot(partial_w_ref[1][0, :numshow])
    plt.subplot(4, 3, 3)
    plt.plot(partial_w_ref[0][0, :numshow] - partial_w_ref[1][0, :numshow])
    plt.subplot(4, 3, 4)
    plt.plot(partial_w_ref[0][1, :numshow])
    plt.subplot(4, 3, 5)
    plt.plot(partial_w_ref[1][1, :numshow])
    plt.subplot(4, 3, 6)
    plt.plot(partial_w_ref[0][1, :numshow] - partial_w_ref[1][1, :numshow])
    plt.subplot(4, 3, 7)
    plt.plot(partial_w_ref[0][2, :numshow])
    plt.subplot(4, 3, 8)
    plt.plot(partial_w_ref[1][2, :numshow])
    plt.subplot(4, 3, 9)
    plt.plot(partial_w_ref[0][2, :numshow] - partial_w_ref[1][2, :numshow])
    plt.subplot(4, 3, 10)
    plt.plot(partial_w_ref[0][3, :numshow])
    plt.subplot(4, 3, 11)
    plt.plot(partial_w_ref[1][3, :numshow])
    plt.subplot(4, 3, 12)
    plt.plot(partial_w_ref[0][3, :numshow] - partial_w_ref[1][3, :numshow])
