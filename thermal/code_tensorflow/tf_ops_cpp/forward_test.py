import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import scipy.io as sio
import unittest
import numpy as np
import tensorflow as tf
from mask_conv import *
#import _mask_conv_grad
#mask_conv_module = tf.load_op_library('build/lib_mask_conv.so')

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



def load_data_elem_s12():
    data = sio.loadmat('/home/hope-yao/Documents/MG_net/data/heat_transfer/Downloads/biphase_12_12_new.mat')
    #/home/hope-yao/Documents/MG_net/data/heat_transfer/Downloads/biphase_12_12.mat
    rho_1, rho_2 =  205., 16.
    num_node = 13
    f_img = data['f_image'].reshape(1, num_node,num_node, 1)
    u_img = data['u_image'].reshape(1, num_node,num_node, 1)
    mask = data['mask'].reshape(1, num_node-1,num_node-1, 1)
    #mask = np.concatenate([mask,np.zeros((1,12,1,1))],2)[:,:,1:,:]
    return num_node, mask, u_img, f_img, rho_1, rho_2

def masked_conv_py(node_resp, elem_mask):
    diag_coef_1, side_coef_1 = 16/3., 16/3. #coef['diag_coef_1'], coef['diag_coef_1']
    diag_coef_2, side_coef_2 = 205/3., 205/3. #coef['diag_coef_2'], coef['diag_coef_2']
    # num_node = node_resp.shape[1] #= np.reshape(node_resp,[1,num_node,num_node,1])
    # elem_mask = np.reshape(elem_mask_orig,[1,num_node-1,num_node-1,1])

    x = node_resp
    # num_node = node_resp.shape[1]-2
    # y_diag_1 = np.zeros((num_node,num_node))
    # y_side_1 = np.zeros((num_node,num_node))
    # y_diag_2 = np.zeros((num_node,num_node))
    # y_side_2 = np.zeros((num_node,num_node))
    # for i in range(1, x.shape[1]-1, 1):
    #     for j in range(1, x.shape[1]-1, 1):
            # y_diag_1[i-1, j-1] = x[0, i-1, j-1, 0] * elem_mask[0, i-1, j-1, 0] *diag_coef_1 \
            #                      + x[0, i-1, j+1, 0] * elem_mask[0, i-1, j, 0] *diag_coef_1 \
            #                      + x[0, i+1, j-1, 0] * elem_mask[0, i, j-1, 0] *diag_coef_1 \
            #                      + x[0, i+1, j+1, 0] * elem_mask[0, i, j, 0] *diag_coef_1
            # y_side_1[i-1, j-1] = x[0, i-1, j, 0] * (elem_mask[0, i-1, j-1, 0] + elem_mask[0, i-1, j, 0]) / 2. *side_coef_1 \
            #                      + x[0, i, j-1, 0] * (elem_mask[0, i-1, j-1, 0] + elem_mask[0, i, j-1, 0]) / 2. *side_coef_1\
            #                      + x[0, i, j + 1, 0] * (elem_mask[0, i-1, j, 0] + elem_mask[0, i, j, 0]) / 2. *side_coef_1\
            #                      + x[0, i+1, j, 0] * (elem_mask[0, i, j-1, 0] + elem_mask[0, i, j, 0]) / 2. *side_coef_1
            # y_diag_2[i-1, j-1] = x[0, i-1, j-1, 0] * (1-elem_mask[0, i-1, j-1, 0]) *diag_coef_2 \
            #                      + x[0, i-1, j+1, 0] * (1-elem_mask[0, i-1, j, 0] )*diag_coef_2 \
            #                      + x[0, i+1, j-1, 0] * (1-elem_mask[0, i, j-1, 0]) *diag_coef_2 \
            #                      + x[0, i+1, j+1, 0] * (1-elem_mask[0, i, j, 0]) *diag_coef_2
            # y_side_2[i-1, j-1] = x[0, i-1, j, 0] * (2-elem_mask[0, i-1, j-1, 0] - elem_mask[0, i-1, j, 0]) / 2. *side_coef_2 \
            #                      + x[0, i, j-1, 0] * (2-elem_mask[0, i-1, j-1, 0] - elem_mask[0, i, j-1, 0]) / 2. *side_coef_2\
            #                      + x[0, i, j + 1, 0] * (2-elem_mask[0, i-1, j, 0] - elem_mask[0, i, j, 0]) / 2. *side_coef_2\
            #                      + x[0, i+1, j, 0] * (2-elem_mask[0, i, j-1, 0] - elem_mask[0, i, j, 0]) / 2. *side_coef_2
            # return y_diag_1 + y_side_1, y_diag_2 + y_side_2, np.expand_dims(
            #     np.expand_dims(y_diag_1 + y_side_1 + y_diag_2 + y_side_2, 0), -1)
            #

    y1 = np.zeros((num_node,num_node))
    y2 = np.zeros((num_node,num_node))
    for i in range(1, x.shape[1]-1, 1):
        for j in range(1, x.shape[1]-1, 1):
            w00 = w11 = w22 = w33 = -205./3.
            w02 = w20 = w31 = w13 = 205./3. /2.
            w12 = w32 = w03 = w23 = w10 = w30 = w01 = w21 = 205/3. /4.
            y1[i-1,j-1] = (w02 * x[0,i+1,j-1,0] + w12 * x[0,i+1,j,0] + w22 * x[0,i,j,0] + w32 * x[0,i,j-1,0]) * elem_mask[0, i, j-1, 0]\
                         + (w03 * x[0,i+1,j,0] + w13 * x[0,i+1,j+1,0] + w23 * x[0,i,j+1,0] + w33 * x[0,i,j,0]) * elem_mask[0, i, j, 0] \
                         + (w00 * x[0,i,j,0] + w10 * x[0,i,j+1,0] + w20 * x[0,i-1,j+1,0] + w30 * x[0,i-1,j,0]) * elem_mask[0, i-1, j, 0]\
                         + (w01 * x[0,i,j-1,0] + w11 * x[0,i,j,0] + w21 * x[0,i-1,j,0] + w31 * x[0,i-1,j-1,0]) * elem_mask[0, i-1, j-1, 0]
            w00 = w11 = w22 = w33 = -16./3.
            w02 = w20 = w31 = w13 = 16./3. /2.
            w12 = w32 = w03 = w23 = w10 = w30 = w01 = w21 = 16/3. /4.
            y2[i-1,j-1] = (w02 * x[0,i+1,j-1,0] + w12 * x[0,i+1,j,0] + w22 * x[0,i,j,0] + w32 * x[0,i,j-1,0]) * (1-elem_mask[0, i, j-1, 0])\
                         + (w03 * x[0,i+1,j,0] + w13 * x[0,i+1,j+1,0] + w23 * x[0,i,j+1,0] + w33 * x[0,i,j,0]) * (1-elem_mask[0, i, j, 0] )\
                         + (w00 * x[0,i,j,0] + w10 * x[0,i,j+1,0] + w20 * x[0,i-1,j+1,0] + w30 * x[0,i-1,j,0]) * (1-elem_mask[0, i-1, j, 0])\
                         + (w01 * x[0,i,j-1,0] + w11 * x[0,i,j,0] + w21 * x[0,i-1,j,0] + w31 * x[0,i-1,j-1,0]) * (1-elem_mask[0, i-1, j-1, 0])
            print('here')
    return y1, y2, np.expand_dims(np.expand_dims(y1+y2,0),-1)






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

    # x = np.pad(x[:, 1:-1, 1:-1, :], ((0, 0), (1, 1), (1, 1), (0, 0)), "constant") # for boundary consideration
    # x = np.pad(x[:, 1:-1, :, :], ((0, 0), (1, 1), (0, 0), (0, 0)), "constant") # for boundary consideration
    return x

with tf.Session(''):
    num_node, mask, u_img, f_img, rho1, rho2 = load_data_elem_s12()#get_data()

    from scipy import signal
    elem_mask = np.squeeze(mask)
    node_filter = np.asarray([[1 / 4.] * 2] * 2)
    node_mask_1 = signal.correlate2d(elem_mask, node_filter, boundary='symm')
    node_mask_2 = signal.correlate2d(np.ones_like(elem_mask) - elem_mask, node_filter, boundary='symm')
    d_matrix = (205. *  node_mask_1 + 16. *  node_mask_2) * (-8./3)
    d_matrix[0,:] /=2
    d_matrix[-1,:] /=2
    d_matrix[:,0] /=2
    d_matrix[:,-1] /=2
    d_matrix = np.expand_dims(np.expand_dims(d_matrix,-1),0)

    padded_input = boundary_padding(u_img)
    padded_mask = boundary_padding(mask)

    res1, res2, masked_res_py = masked_conv_py(padded_input, padded_mask)
    result_py = boundary_correct(masked_res_py)
    u_py = (f_img - result_py) / d_matrix
    print('py conv done')
    masked_res_tf = mask_conv(padded_input, padded_mask, np.asarray([rho1,rho2],'float32')).eval()
    result_tf = boundary_correct(masked_res_tf)
    u_t = (f_img - result_tf) / d_matrix
    print('tf conv done')

    import matplotlib.pyplot as plt
    plt.imshow(u_t[0, :, :, 0]-u_py[0, :, :, 0])
    plt.colorbar()
    plt.show()

    # np.save('result_py',result_py)
    # np.save('result_tf',result_tf)
    # np.testing.assert_array_equal(result_tf, result_py)

# class MaskConvTest(tf.test.TestCase):
#   def test(self):
#     pass
#
#   def test_grad(self):
#     with tf.device('/cpu:0'):
#         mask, u_img, f_img = get_data()
#         u_img_tf = tf.constant(u_img, dtype=tf.float32)
#         mask_tf = tf.constant(mask, dtype=tf.float32)
#         result_tf = mask_conv_module.mask_conv(u_img_tf, mask_tf)
#     print result_tf
#
#     with self.test_session():
#       print "---- Going to compute gradient error"
#       err = tf.test.compute_gradient_error([u_img_tf,mask_tf], [(1,66,66,1),(1,65,65,1)], result_tf, (1,66,66,1))
#       print err
#       self.assertLess(err, 1e-4)
#
# if __name__ == '__main__':
#     tf.test.main()
