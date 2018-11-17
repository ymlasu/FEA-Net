import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import scipy.io as sio
import unittest
import numpy as np
import tensorflow as tf

import _mask_conv_grad
mask_conv_module = tf.load_op_library('build/lib_mask_conv.so')

num_node = 66
def get_data():
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
    return mask, u_img, f_img

def masked_conv_py(node_resp, elem_mask_orig):
    diag_coef_1, side_coef_1 = 1, 1 #coef['diag_coef_1'], coef['diag_coef_1']
    diag_coef_2, side_coef_2 = 1, 1 #coef['diag_coef_2'], coef['diag_coef_2']
    x = np.reshape(node_resp,[1,num_node,num_node,1])
    elem_mask = np.reshape(elem_mask_orig,[1,num_node-1,num_node-1,1])

    y_diag_1 = np.zeros((1, num_node, num_node, 1))
    y_side_1 = np.zeros((1, num_node, num_node, 1))
    y_diag_2 = np.zeros((1, num_node, num_node, 1))
    y_side_2 = np.zeros((1, num_node, num_node, 1))
    for i in range(1, x.shape[1]-1, 1):
        for j in range(1, x.shape[1]-1, 1):
            y_diag_1[0, i-1, j-1, 0] = x[0, i-1, j-1, 0] * elem_mask[0, i-1, j-1, 0] *diag_coef_1 \
                                 + x[0, i-1, j+1, 0] * elem_mask[0, i-1, j, 0] *diag_coef_1 \
                                 + x[0, i+1, j-1, 0] * elem_mask[0, i, j-1, 0] *diag_coef_1 \
                                 + x[0, i+1, j+1, 0] * elem_mask[0, i, j, 0] *diag_coef_1
            y_side_1[0, i-1, j-1, 0] = x[0, i-1, j, 0] * (elem_mask[0, i-1, j-1, 0] + elem_mask[0, i-1, j, 0]) / 2. *side_coef_1 \
                                 + x[0, i, j-1, 0] * (elem_mask[0, i-1, j-1, 0] + elem_mask[0, i, j-1, 0]) / 2. *side_coef_1\
                                 + x[0, i, j + 1, 0] * (elem_mask[0, i-1, j, 0] + elem_mask[0, i, j, 0]) / 2. *side_coef_1\
                                 + x[0, i+1, j, 0] * (elem_mask[0, i, j-1, 0] + elem_mask[0, i, j, 0]) / 2. *side_coef_1
            y_diag_2[0, i-1, j-1, 0] = x[0, i-1, j-1, 0] * (1-elem_mask[0, i-1, j-1, 0]) *diag_coef_2 \
                                 + x[0, i-1, j+1, 0] * (1-elem_mask[0, i-1, j, 0] )*diag_coef_2 \
                                 + x[0, i+1, j-1, 0] * (1-elem_mask[0, i, j-1, 0]) *diag_coef_2 \
                                 + x[0, i+1, j+1, 0] * (1-elem_mask[0, i, j, 0]) *diag_coef_2
            y_side_2[0, i-1, j-1, 0] = x[0, i-1, j, 0] * (2-elem_mask[0, i-1, j-1, 0] - elem_mask[0, i-1, j, 0]) / 2. *side_coef_2 \
                                 + x[0, i, j-1, 0] * (2-elem_mask[0, i-1, j-1, 0] - elem_mask[0, i, j-1, 0]) / 2. *side_coef_2\
                                 + x[0, i, j + 1, 0] * (2-elem_mask[0, i-1, j, 0] - elem_mask[0, i, j, 0]) / 2. *side_coef_2\
                                 + x[0, i+1, j, 0] * (2-elem_mask[0, i, j-1, 0] - elem_mask[0, i, j, 0]) / 2. *side_coef_2
    return y_diag_1 + y_side_1 + y_diag_2 + y_side_2

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
    x[:, 0, :, :] /=2
    x[:, -1, :, :] /=2
    x[:, :, 0, :] /=2
    x[:, :, -1, :] /=2

    x = np.pad(x[:, 1:-1, :, :], ((0,0), (1, 1), (0, 0), (0, 0)), "constant") # for boundary consideration
    return x

# class InnerProductOpTest(unittest.TestCase):
#     def test_innerProductRandom(self):
#         with tf.Session(''):
#             mask, u_img, f_img = get_data()
#             #padded_input = boundary_padding(u_img)
#             #padded_mask = boundary_padding(mask)
#             #result_py = masked_conv_py(padded_input, padded_mask)
#             #result_py = boundary_correct(result_py)
#             result_py = masked_conv_py(u_img, mask)
#             print('py conv done')
#             result_tf = mask_conv_module.mask_conv(u_img, mask).eval()
#             print('tf conv done')
#             np.save('result_py',result_py)
#             np.save('result_tf',result_tf)
#             np.testing.assert_array_equal(result_tf, result_py)

class MaskConvTest(tf.test.TestCase):
  def test(self):
    pass

  def test_grad(self):
    with tf.device('/cpu:0'):
        mask, u_img, f_img = get_data()
        u_img_tf = tf.constant(u_img, dtype=tf.float32)
        mask_tf = tf.constant(mask, dtype=tf.float32)
        result_tf = mask_conv_module.mask_conv(u_img_tf, mask_tf)
    print result_tf

    with self.test_session():
      print "---- Going to compute gradient error"
      err = tf.test.compute_gradient_error([u_img_tf,mask_tf], [(1,66,66,1),(1,65,65,1)], result_tf, (1,66,66,1))
      print err
      self.assertLess(err, 1e-4)

if __name__ == '__main__':
    tf.test.main()
