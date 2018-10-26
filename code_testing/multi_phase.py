import numpy as np
import tensorflow as tf
from custom_ops_tf import tf_conv_2phase as masked_conv
from custom_ops_tf import get_D_matrix
from data_loader import load_data_elem

def jacobi_itr(u_input, f_input, d_matrix, elem_mask, conductivity_1, conductivity_2):

    LU_u = masked_conv(elem_mask, u_input, conductivity_1, conductivity_2)
    u_new = (f_input - LU_u['LU_u']) / d_matrix
    u_new = tf.pad(u_new[:,:,1:,:], tf.constant([[0, 0], [0, 0], [1, 0], [0, 0]]),"CONSTANT")  # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!

    return u_new, LU_u


def main():
    n_elem_x = n_elem_y = 64
    conductivity_1 = tf.placeholder(tf.float32, shape=())
    conductivity_2 = tf.placeholder(tf.float32, shape=())
    f_input = tf.placeholder(tf.float32, shape=(1, n_elem_x+1, n_elem_y+1, 1))
    u_input = tf.placeholder(tf.float32, shape=(1, n_elem_x+1, n_elem_y+1, 1))
    elem_mask = tf.placeholder(tf.float32, shape=(1, n_elem_x, n_elem_y, 1))

    d_matrix = get_D_matrix(elem_mask, conductivity_1, conductivity_2)

    n_itr = 100
    u_hist = [u_input]
    LU_u_hist = []
    for i in range(n_itr):
        u_new, LU_u = jacobi_itr(u_hist[-1], f_input, d_matrix, elem_mask, conductivity_1, conductivity_2)
        u_hist += [u_new]
        LU_u_hist += [LU_u]

    u_gt, f_gt, elem_mask_gt, conductivity_1, conductivity_2 = load_data_elem(case=0)

    ## training starts ###
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    loss_hist = []
    feed_dict = {f_input: f_gt.reshape(1, n_elem_x+1, n_elem_y+1, 1),
                 u_input: np.zeros_like(u_gt.reshape(1, n_elem_x+1, n_elem_y+1, 1)),
                 elem_mask: elem_mask_gt}
    for ii in range(1, 2000, 100):
        loss_hist += [sess.run(tf.reduce_mean(tf.abs(u_hist[ii] - u_gt)),feed_dict)]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(sess.run(u_hist[-1][0, 1:-1, 1:-1, 0], feed_dict), cmap='jet')
    plt.colorbar()
    plt.grid('off')
    plt.figure()
    plt.imshow(u_gt[0, 1:-1, 1:-1, 0], cmap='jet')
    plt.colorbar()
    plt.grid('off')
    plt.show()
    return u_hist

if __name__ == '__main__':
    cfg = {
        'batch_size': 1,
        'imsize': 64,
        'physics_problem': 'heat_transfer',  # candidates: 3D plate elasticity, helmholtz, vibro-acoustics
        'alpha': 5000,  # iteration
    }

    u_hist = main()
    print('done')