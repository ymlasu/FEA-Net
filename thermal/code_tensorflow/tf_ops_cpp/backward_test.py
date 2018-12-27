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


def boundary_padding(x):
    ''' special symmetric boundary padding '''
    left = x[:, :, 1:2, :]
    right = x[:, :, -2:-1, :]
    upper = tf.concat([x[:, 1:2, 1:2, :], x[:, 1:2, :, :], x[:, 1:2, -2:-1, :]], 2)
    down = tf.concat([x[:, -2:-1, 1:2, :], x[:, -2:-1, :, :], x[:, -2:-1, -2:-1, :]], 2)
    padded_x = tf.concat([left, x, right], 2)
    padded_x = tf.concat([upper, padded_x, down], 1)
    return padded_x
def boundary_correct(x):
    mask = np.ones((1,66,66,1))
    mask[:, 0, :, :] /= 2
    mask[:, -1, :, :] /= 2
    mask[:, :, 0, :] /= 2
    mask[:, :, -1, :] /= 2
    return x*mask

if 0:
    mask, u_img, f_img = get_data()
    x_pl = tf.constant(u_img, dtype=tf.float32)
    w_pl = tf.constant(mask, dtype=tf.float32)
    padded_input = boundary_padding(x_pl)
    padded_mask = boundary_padding(w_pl)
    masked_res_tf = mask_conv_module.mask_conv(padded_input, padded_mask)
    result_tf = boundary_correct(masked_res_tf)
    sess = tf.Session()
    # reference Jacobian
    partial_u_ref, partial_w_ref = tf.test.compute_gradient([x_pl, w_pl], [[1, 66, 66, 1], [1, 65, 65, 1]],
                                                            result_tf, [1, 66, 66, 1],
                                                            x_init_value = [u_img, mask])
else:
    if 0:
        x_pl = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        w_pl = tf.constant([[10], [20]], dtype=tf.float32)
        masked_res_tf = tf.matmul(x_pl, w_pl)
        # reference Jacobian
        with tf.Session():
            partial_u_ref, partial_w_ref = tf.test.compute_gradient([x_pl, w_pl], [[2, 2], [2, 1]],
                                                                    masked_res_tf, [2, 1], delta=1e-3)
    num_test_node = 66
    #u_img = np.ones((1,num_test_node,num_test_node,1))*2
    #mask = np.ones((1,num_test_node-1,num_test_node-1,1))*10# np.asarray([[[[1],[1]],[[0],[0]]]])#
    mask, u_img, f_img = get_data()
    x_pl = tf.constant(u_img, dtype=tf.float32)
    w_pl = tf.constant(mask, dtype=tf.float32)
    #masked_res_tf = mask_conv_module.mask_conv(x_pl, w_pl)
    padded_input = boundary_padding(x_pl)
    padded_mask = boundary_padding(w_pl)
    masked_res_tf = mask_conv_module.mask_conv(padded_input, padded_mask)
    masked_res_tf = boundary_correct(masked_res_tf)

    # reference Jacobian
    with tf.Session():
        partial_u_ref, partial_w_ref = tf.test.compute_gradient([x_pl, w_pl], [[1, num_test_node, num_test_node, 1], [1, num_test_node-1, num_test_node-1, 1]],
                                                                masked_res_tf, [1, num_test_node, num_test_node, 1], delta=1e-3)

if 0:
    numshow = 100
    import matplotlib.pyplot as plt

    plt.subplot(1, 3, 1)
    plt.imshow(partial_u_ref[0][:numshow, :numshow])
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(partial_u_ref[1][:numshow, :numshow])
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(partial_u_ref[0][:numshow, :numshow] - partial_u_ref[1][:numshow, :numshow])
    plt.colorbar()

    
np.squeeze(sess.run(tf.gradients(masked_res_tf, w_pl)[0])).tolist()
# computed Jacobian
partial_wx_partial_x = tf.gradients(result_tf, x_pl)
partial_wx_partial_x_np = sess.run(partial_wx_partial_x[0], {x_pl:u_img, w_pl:mask})
print(partial_wx_partial_x_np.shape)
partial_wx_partial_w = tf.gradients(result_tf, w_pl)
partial_wx_partial_w_np = sess.run(partial_wx_partial_w[0], {x_pl:u_img, w_pl:mask})
print(partial_wx_partial_w_np)
import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.imshow(np.squeeze(partial_wx_partial_x_np))
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(np.squeeze(partial_wx_partial_w_np))
plt.colorbar()
    # diff = tf.test.compute_gradient_error([u_img, mask], [[1, 66, 66, 1], [1, 65, 65, 1]], result_tf, [1, 66, 66, 1])
    # print(diff)

# class InnerProductOpTest(unittest.TestCase):
#
#     def test_innerProductGradientXHardCoded(self):
#         mask, u_img, f_img = get_data()
#         with tf.Session('') as sess:
#             x = tf.placeholder(tf.float32, shape = (1, num_node, num_node, 1))
#             w = tf.placeholder(tf.float32, shape = (1, num_node-1, num_node-1, 1))
#
#             wx_tf = mask_conv_module.mask_conv(x, w)
#             partial_wx_partial_x = tf.gradients(wx_tf, x)
#             print(partial_wx_partial_x[0].get_shape())
#             partial_wx_partial_w = tf.gradients(wx_tf, w)
#             print(partial_wx_partial_w[0].get_shape())
#             tf.test.compute_gradient_error([x, w], [[1, 66, 66, 1], [1, 65, 65, 1]], wx_tf, [1, 66, 66, 1])
#             #gradient_tf = sess.run(grad_x_tf, feed_dict = {x: np.asarray([1, 2]).astype(np.float32)})
#             #gradient_inner_product = sess.run(grad_x_inner_product, feed_dict = {x: np.asarray([1, 2]).astype(np.float32)})
#
#             #self.assertEqual(gradient_tf[0][0], gradient_inner_product[0][0])
#             #self.assertEqual(gradient_tf[0][1], gradient_inner_product[0][1])
#
# if __name__ == '__main__':
#     unittest.main()
