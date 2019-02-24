import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import scipy.io as sio
import unittest
import numpy as np
import tensorflow as tf
from mask_elast_conv import *
#import _mask_conv_grad
#mask_conv_module = tf.load_op_library('build/lib_mask_conv.so')


def load_data_elem_s12():
    num_node = 13

    if 1:
        # data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/2D_elastic_E200_E230.mat')
        # rho = [230/1e3,0.36, 200/1e3,0.25]
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/2D_elastic_xy_fixed.mat')
        rho = [230/1e3,0.36, 200/1e3,0.25]
        u_img = np.concatenate([data['ux'].reshape(1, 13, 13, 1), data['uy'].reshape(1, 13, 13, 1)], -1) * 1e6
        f_img = np.concatenate([data['fx'].reshape(1, 13, 13, 1), data['fy'].reshape(1, 13, 13, 1)], -1) / 1e6
        mask = data['mask'].reshape(1, num_node - 1, num_node - 1, 1)
    else:
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/crack_size13.mat')
        rho = [200/1e3,0.25, 200/1e3,0.25]
        ux = data['d_x'].reshape(1, num_node,num_node).transpose((0, 2, 1)) * 1e6 # changed magnitude for numerical stability
        uy = data['d_y'].reshape(1, num_node,num_node).transpose((0, 2, 1)) * 1e6
        u_img = np.concatenate([np.expand_dims(ux, 3), np.expand_dims(uy, 3)], 3)
        fx = data['f_x'].reshape(1, num_node,num_node).transpose((0, 2, 1))
        fy = data['f_y'].reshape(1, num_node,num_node).transpose((0, 2, 1))
        f_img = -1. * np.concatenate([np.expand_dims(fx, 3), np.expand_dims(fy, 3)], 3)
        mask = np.asarray([[1]*12]*4+[[1]*12]*8).reshape(1,12,12,1)

    return num_node, mask, u_img, f_img, rho

def boundary_padding(x):
    ''' special symmetric boundary padding '''
    left = x[:, :, 1:2, :]
    right = x[:, :, -2:-1, :]
    upper = np.concatenate([x[:, 1:2, 1:2, :], x[:, 1:2, :, :], x[:, 1:2, -2:-1, :]], 2)
    down = np.concatenate([x[:, -2:-1, 1:2, :], x[:, -2:-1, :, :], x[:, -2:-1, -2:-1, :]], 2)
    padded_x = np.concatenate([left, x, right], 2)
    padded_x = np.concatenate([upper, padded_x, down], 1)
    return padded_x

def boundary_correct(x):
    x[:,0,:,:] /=2
    x[:,-1,:,:] /=2
    x[:,:,0,:] /=2
    x[:,:,-1,:] /=2
    return x

with tf.Session(''):
    import matplotlib.pyplot as plt
    from scipy import signal

    num_node, mask, u_img, f_img, rho = load_data_elem_s12()#get_data()
    padded_input = boundary_padding(u_img)
    padded_mask = boundary_padding(mask)
    d_matrix = get_dmatrix(tf.constant(padded_mask,dtype=tf.float32), tf.constant(rho,dtype=tf.float32))

    masked_res_tf = mask_conv(padded_input, padded_mask, rho).eval()
    # result_tf = boundary_correct(masked_res_tf)
    ff = masked_res_tf + d_matrix * u_img
    plt.imshow(ff[0, :, :, 1])
    # u_t = (f_img - result_tf) / d_matrix
    print('tf conv done')
