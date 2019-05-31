import tensorflow as tf
import _mask_conv_elast_grad
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
#mask_conv_module=tf.load_op_library(os.path.join(BASE_DIR, '/home/hope-yao/Documents/FEA_Net/thermal/code_tensorflow/tf_ops_cpp/build/lib_mask_conv.so'))
mask_conv_module=tf.load_op_library( '/home/hope-yao/Documents/FEA_Net/elasticity/code_tensorflow/tf_ops_cpp/build/lib_mask_conv_elast.so')
getdmat_module=tf.load_op_library( '/home/hope-yao/Documents/FEA_Net/elasticity/code_tensorflow/tf_ops_cpp/build/lib_getdmat_elast.so')

def mask_conv(resp, mask, rho):

    return mask_conv_module.maskconv_elast(resp, mask, rho)

def get_dmatrix(mask, rho):
    # flag is not used, just as a placeholder that is not well designed
    flag = 1
    return getdmat_module.getdmat_elast(flag, mask, rho)


# def get_dmatrix(mask, rho):
#     _, dx, dy, _ = mask.get_shape().as_list()
#     E_1, mu_1, E_2, mu_2 = tf.split(rho,4)
#     coef_1 = E_1 / (16. * (1 - mu_1 * mu_1))
#     coef_2 = E_2 / (16. * (1 - mu_2 * mu_2))
#
#     # from appendix
#     k1_xx_00 = k1_xx_11 = k1_xx_22 = k1_xx_33 = 8 - 8 / 3. * mu_1
#     k1_yy_00 = k1_yy_11 = k1_yy_22 = k1_yy_33 = 8 - 8 / 3. * mu_1
#     k1_xy_00 = k1_xy_22 = k1_yx_00 = k1_yx_22 = 2 * mu_1 + 2
#     k1_xy_11 = k1_xy_33 = k1_yx_11 = k1_yx_33 = -2 * mu_1 - 2
#
#     k2_xx_00 = k2_xx_11 = k2_xx_22 = k2_xx_33 = 8 - 8 / 3. * mu_2
#     k2_yy_00 = k2_yy_11 = k2_yy_22 = k2_yy_33 = 8 - 8 / 3. * mu_2
#     k2_xy_00 = k2_xy_22 = k2_yx_00 = k2_yx_22 = 2 * mu_2 + 2
#     k2_xy_11 = k2_xy_33 = k2_yx_11 = k2_yx_33 = -2 * mu_2 - 2
#
#     # from Eq.8, fea_expansion_void
#     # dmat = tf.zeros((1, dx-1, dy-1, 2))
#     # for i in range(1,dx,1):
#     #     for j in range(1,dy,1):
#     #         dmat[0,i-1,j-1,0] = coef_1*(k1_xx_00 * mask[0,i-1,j,0] + k1_xx_11 * mask[0,i-1,j-1,0] + k1_xx_22 * mask[0,i,j-1,0] + k1_xx_33 * mask[0,i,j,0])\
#     #                         + coef_2*(k2_xx_00 * (1-mask[0,i-1,j,0]) + k2_xx_11 * (1-mask[0,i-1,j-1,0]) + k2_xx_22 * (1-mask[0,i,j-1,0]) + k2_xx_33 * (1-mask[0,i,j,0]))\
#     #                         # + coef_1*(k1_yx_00 * mask[0,i-1,j,0] + k1_yx_11 * mask[0,i-1,j-1,0] + k1_yx_22 * mask[0,i,j-1,0] + k1_yx_33 * mask[0,i,j,0])\
#     #                         # + coef_2*(k2_yx_00 * (1-mask[0,i-1,j,0]) + k2_yx_11 * (1-mask[0,i-1,j-1,0]) + k2_yx_22 * (1-mask[0,i,j-1,0]) + k2_yx_33 * (1-mask[0,i,j,0]))
#     #
#     #         dmat[0,i-1,j-1,1] = coef_1*(k1_yy_00 * mask[0,i-1,j,0] + k1_yy_11 * mask[0,i-1,j-1,0] + k1_yy_22 * mask[0,i,j-1,0] + k1_yy_33 * mask[0,i,j,0])\
#     #                         + coef_2*(k2_yy_00 * (1-mask[0,i-1,j,0]) + k2_yy_11 * (1-mask[0,i-1,j-1,0]) + k2_yy_22 * (1-mask[0,i,j-1,0]) + k2_yy_33 * (1-mask[0,i,j,0]))\
#     #                         # + coef_1*(k1_xy_00 * mask[0,i-1,j,0] + k1_xy_11 * mask[0,i-1,j-1,0] + k1_xy_22 * mask[0,i,j-1,0] + k1_xy_33 * mask[0,i,j,0])\
#     #                         # + coef_2*(k2_xy_00 * (1-mask[0,i-1,j,0]) + k2_xy_11 * (1-mask[0,i-1,j-1,0]) + k2_xy_22 * (1-mask[0,i,j-1,0]) + k2_xy_33 * (1-mask[0,i,j,0]))\
#
#     dmat_x = []
#     dmat_y = []
#     for i in range(1,dx,1):
#         for j in range(1,dy,1):
#             dmat_x += [
#                 coef_1*(k1_xx_00 * mask[0,i-1,j,0] + k1_xx_11 * mask[0,i-1,j-1,0] + k1_xx_22 * mask[0,i,j-1,0] + k1_xx_33 * mask[0,i,j,0])\
#                 + coef_2*(k2_xx_00 * (1-mask[0,i-1,j,0]) + k2_xx_11 * (1-mask[0,i-1,j-1,0]) + k2_xx_22 * (1-mask[0,i,j-1,0]) + k2_xx_33 * (1-mask[0,i,j,0]))\
#                 # + coef_1*(k1_yx_00 * mask[0,i-1,j,0] + k1_yx_11 * mask[0,i-1,j-1,0] + k1_yx_22 * mask[0,i,j-1,0] + k1_yx_33 * mask[0,i,j,0])\
#                 # + coef_2*(k2_yx_00 * (1-mask[0,i-1,j,0]) + k2_yx_11 * (1-mask[0,i-1,j-1,0]) + k2_yx_22 * (1-mask[0,i,j-1,0]) + k2_yx_33 * (1-mask[0,i,j,0]))
#             ]
#
#             dmat_y += [
#                 coef_1*(k1_yy_00 * mask[0,i-1,j,0] + k1_yy_11 * mask[0,i-1,j-1,0] + k1_yy_22 * mask[0,i,j-1,0] + k1_yy_33 * mask[0,i,j,0])\
#                 + coef_2*(k2_yy_00 * (1-mask[0,i-1,j,0]) + k2_yy_11 * (1-mask[0,i-1,j-1,0]) + k2_yy_22 * (1-mask[0,i,j-1,0]) + k2_yy_33 * (1-mask[0,i,j,0]))\
#                 # + coef_1*(k1_xy_00 * mask[0,i-1,j,0] + k1_xy_11 * mask[0,i-1,j-1,0] + k1_xy_22 * mask[0,i,j-1,0] + k1_xy_33 * mask[0,i,j,0])\
#                 # + coef_2*(k2_xy_00 * (1-mask[0,i-1,j,0]) + k2_xy_11 * (1-mask[0,i-1,j-1,0]) + k2_xy_22 * (1-mask[0,i,j-1,0]) + k2_xy_33 * (1-mask[0,i,j,0]))\
#             ]
#
#     dmat_x = tf.reshape(tf.stack(dmat_x),(1,dx-1,dy-1,1))
#     dmat_y = tf.reshape(tf.stack(dmat_y),(1,dx-1,dy-1,1))
#     return tf.concat([dmat_x,dmat_y],-1)
#
