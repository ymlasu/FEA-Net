import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import scipy.io as sio
import unittest
import numpy as np
import tensorflow as tf
import _mask_conv_grad
mask_conv_module = tf.load_op_library('build/lib_mask_conv.so')


# num_node = 66
# def get_data():
#     data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/thermal/data/bc1/3circle_center_25_25_rad_17_center_55_55_rad_7_center_55_25_rad_7.mat')
#     mask = np.ones((1, num_node-1, num_node-1, 1))
#     for i in range(num_node-1):
#         for j in range(num_node-1):
#             if (i-24)**2+(j-24)**2<17**2 or (i-24)**2+(j-54)**2<7**2 or (i-54)**2+(j-54)**2<7**2:
#                 mask[:, i, j, :] = 0
#
#     A = data['K']
#     f = data['f']
#     u = np.linalg.solve(A, f)
#     u_img = u.reshape(1, num_node,num_node, 1).transpose((0,2,1,3))
#     f_img = f.reshape(1, num_node,num_node, 1).transpose((0,2,1,3))
#     return mask, u_img, f_img


def get_data():
    num_node = 66
    data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/thermal/data/bc1/3circle_center_25_25_rad_17_center_55_55_rad_7_center_55_25_rad_7.mat')
    mask = np.ones((1, num_node-1, num_node-1, 1))
    for i in range(num_node-1):
        for j in range(num_node-1):
            if (i-24)**2+(j-24)**2<17**2 or (i-24)**2+(j-54)**2<7**2 or (i-54)**2+(j-54)**2<7**2:
                mask[:, i, j, :] = 0

    A = data['K']
    f = data['f']
    u = np.linalg.solve(A, f)
    u_img = u.reshape(1, num_node,num_node, 1).transpose((0,2,1,3))
    f_img = f.reshape(1, num_node,num_node, 1).transpose((0,2,1,3))
    rho_1, rho_2 = 16., 205.
    return num_node, mask, u_img, f_img, rho_1, rho_2

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
    num_node, mask, u_img, f_img, rho1, rho2 = get_data()
    x_pl = tf.constant(u_img, dtype=tf.float32)
    m_pl = tf.constant(mask, dtype=tf.float32)
    w_pl = tf.constant([rho1, rho2], dtype=tf.float32)
    padded_input = boundary_padding(x_pl)
    padded_mask = boundary_padding(m_pl)
    masked_res_tf = mask_conv_module.mask_conv(padded_input, padded_mask, w_pl)
    masked_res_tf = boundary_correct(masked_res_tf, num_node)

    # reference Jacobian
    with tf.Session():
        partial_u_ref, partial_m_ref , partial_w_ref = tf.test.compute_gradient([x_pl, m_pl, w_pl],
                                                                               [[1, num_node, num_node, 1],
                                                                                [1, num_node-1, num_node-1, 1],
                                                                                [2]],
                                                                               masked_res_tf,
                                                                               [1, num_node, num_node, 1],
                                                                               delta=1e-3)

    numshow = 100
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

    numshow = 10000
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.plot(partial_w_ref[0][0, :numshow])
    plt.subplot(2, 3, 2)
    plt.plot(partial_w_ref[1][0, :numshow])
    plt.subplot(2, 3, 3)
    plt.plot(partial_w_ref[0][0, :numshow] - partial_w_ref[1][0, :numshow])
    plt.subplot(2, 3, 4)
    plt.plot(partial_w_ref[0][1, :numshow])
    plt.subplot(2, 3, 5)
    plt.plot(partial_w_ref[1][1, :numshow])
    plt.subplot(2, 3, 6)
    plt.plot(partial_w_ref[0][1, :numshow] - partial_w_ref[1][1, :numshow])
