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
    return partial_mask, partial_resp


def mask_conv_grad(op, grads):
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
    diag_coef_diff = diag_coef_1 - diag_coef_2
    side_coef_diff = side_coef_1 - side_coef_2
    for i in range(1, mask.shape[1], 1):
        for j in range(1, mask.shape[2], 1):
            partial_mask[0, i-1, j-1, 0] +=  grad[0,i-1,j-1,0] * (resp[0,i-1,j-1,0] * diag_coef_diff + (resp[0,i-1,j-1,0] + resp[0,i,j-1,0])/2. * side_coef_diff)
            partial_mask[0, i-1, j, 0] += grad[0,i-1,j,0] * (resp[0,i-1,j+1,0] * diag_coef_diff + (resp[0,i,j+1,0] + resp[0,i-1,j,0])/ 2. * side_coef_diff)
            partial_mask[0, i, j-1, 0] += grad[0,i,j-1,0] * (resp[0,i+1,j-1,0] * diag_coef_diff + (resp[0,i+1,j,0] + resp[0,i,j-1,0])/ 2. * side_coef_diff) 
            partial_mask[0, i, j, 0] += grad[0,i,j,0] * (resp[0,i+1,j+1,0] * diag_coef_diff + (resp[0,i+1,j,0] + resp[0,i,j+1,0])/ 2. * side_coef_diff) 


    for i in range(1, resp.shape[1]-1, 1):
        for j in range(1, resp.shape[2]-1, 1):
            partial_resp[0, i-1, j-1, 0] += grad[0, i-1, j-1, 0] * (mask[0, i-1, j-1, 0] * diag_coef_diff + diag_coef_2)
            partial_resp[0, i-1, j, 0] += grad[0, i-1, j, 0] * ((mask[0, i-1, j-1, 0]+mask[0, i-1, j, 0])/2 * side_coef_diff + side_coef_2)
            partial_resp[0, i-1, j+1, 0] += grad[0, i-1, j+1, 0] * (mask[0, i-1, j, 0] * diag_coef_1 + (1-mask[0, i-1, j, 0]) * diag_coef_2)

            partial_resp[0, i, j-1, 0] += grad[0, i, j-1, 0] * ((mask[0, i-1, j-1, 0]+mask[0, i, j-1, 0])/2 * side_coef_diff + side_coef_2)
            partial_resp[0, i, j+1, 0] += grad[0, i, j+1, 0] * ((mask[0, i-1, j, 0]+mask[0, i, j, 0])/2 * side_coef_diff + side_coef_2)

            partial_resp[0, i+1, j-1, 0] += grad[0, i+1, j-1, 0] * (mask[0, i, j-1, 0] * diag_coef_diff + diag_coef_2)
            partial_resp[0, i+1, j, 0] += grad[0, i+1, j, 0] * ((mask[0, i, j-1, 0]+mask[0, i, j, 0])/2 * side_coef_diff + side_coef_2)
            partial_resp[0, i+1, j+1, 0] += grad[0, i+1, j+1, 0] * (mask[0, i, j, 0] * diag_coef_1 + (1-mask[0, i, j, 0]) * diag_coef_2 )

    # the propagated gradient with respect to the first and second argument respectively
    return partial_mask, partial_resp


class InnerProductOpTest(unittest.TestCase):

    def test_innerProductGradientXHardCoded(self):
        mask, u_img, f_img = get_data()
        with tf.Session('') as sess:
            x = tf.placeholder(tf.float32, shape = (1, num_node, num_node, 1))
            w = tf.placeholder(tf.float32, shape = (1, num_node-1, num_node-1, 1))
            
            wx_tf = mask_conv_module.mask_conv(x, w)
            partial_wx_partial_x = tf.gradients(wx_tf, x)
            print(partial_wx_partial_x[0].get_shape())
            partial_wx_partial_w = tf.gradients(wx_tf, w)
            print(partial_wx_partial_w[0].get_shape())

            #gradient_tf = sess.run(grad_x_tf, feed_dict = {x: np.asarray([1, 2]).astype(np.float32)})
            #gradient_inner_product = sess.run(grad_x_inner_product, feed_dict = {x: np.asarray([1, 2]).astype(np.float32)})
            
            #self.assertEqual(gradient_tf[0][0], gradient_inner_product[0][0])
            #self.assertEqual(gradient_tf[0][1], gradient_inner_product[0][1])

if __name__ == '__main__':
    unittest.main()
