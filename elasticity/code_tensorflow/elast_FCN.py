import tensorflow as tf
slim = tf.contrib.slim
# from elast_1phase_2D_tf import load_data_elem
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import os
batch_size = 6
num_node = 13

def load_data_elem(num_node, noise_mag=0):
    '''loading data obtained from FEA simulation'''
    # linear elasticity, all steel, Yfix
    def read_mat(fn):
        data = sio.loadmat(fn)
        ux = data['d_x'].reshape(1, num_node,num_node).transpose((0, 2, 1)) * 1e12 # changed magnitude for numerical stability
        uy = data['d_y'].reshape(1, num_node,num_node).transpose((0, 2, 1)) * 1e12
        u_img = np.concatenate([np.expand_dims(ux, 3), np.expand_dims(uy, 3)], 3)
        fx = data['f_x'].reshape(1, num_node,num_node).transpose((0, 2, 1))
        fy = data['f_y'].reshape(1, num_node,num_node).transpose((0, 2, 1))
        f_img = -1. * np.concatenate([np.expand_dims(fx, 3), np.expand_dims(fy, 3)], 3)
        return u_img, f_img

    data_dir = '/home/hope-yao/Documents/FEA_Net/elasticity/data'
    data_fn = [#'crack_size13.mat',
               'crack_size13_case1.mat',
               'crack_size13_case2.mat',
               #'crack_size13_case3.mat',
               'crack_size13_case4.mat',
               'crack_size13_case5.mat']
    u_img = []
    f_img = []
    for fn_i in data_fn:
        u_img_i, f_img_i = read_mat(os.path.join(data_dir, fn_i))
        if noise_mag:
            noise_mag_x = np.max(np.abs(u_img_i[:,:,:,0])) * noise_mag
            noise_mag_y = np.max(np.abs(u_img_i[:,:,:,1])) * noise_mag
            u_noise_x = np.random.uniform(0, 1, (1, num_node, num_node, 1)) * noise_mag_x
            u_noise_y = np.random.uniform(0, 1, (1, num_node, num_node, 1)) * noise_mag_y
            u_noise = np.concatenate([u_noise_x, u_noise_y], -1)
            u_img_i += u_noise
        u_img += [u_img_i]
        f_img += [f_img_i]
    u_img_train, f_img_train =  np.concatenate(u_img,0), np.concatenate(f_img,0)



    data_dir = '/home/hope-yao/Documents/FEA_Net/elasticity/data'
    data_fn = ['crack_size13.mat',
               # 'crack_size13_case1.mat',
               # 'crack_size13_case2.mat',
               'crack_size13_case3.mat',
               # 'crack_size13_case4.mat',
               # 'crack_size13_case5.mat'
               ]
    u_img = []
    f_img = []
    for fn_i in data_fn:
        u_img_i, f_img_i = read_mat(os.path.join(data_dir, fn_i))
        if noise_mag:
            noise_mag_x = np.max(np.abs(u_img_i[:,:,:,0])) * noise_mag
            noise_mag_y = np.max(np.abs(u_img_i[:,:,:,1])) * noise_mag
            u_noise_x = np.random.uniform(0, 1, (1, num_node, num_node, 1)) * noise_mag_x
            u_noise_y = np.random.uniform(0, 1, (1, num_node, num_node, 1)) * noise_mag_y
            u_noise = np.concatenate([u_noise_x, u_noise_y], -1)
            u_img_i += u_noise
        u_img += [u_img_i]
        f_img += [f_img_i]
    u_img_test01, f_img_test01 =  np.concatenate(u_img,0), np.concatenate(f_img,0)

    data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/block_case1.mat')
    num_node = 65
    ux = data['d_x'].reshape(num_node, num_node).transpose((1, 0))  # * 1e10
    uy = data['d_y'].reshape(num_node, num_node).transpose((1, 0))  # * 1e10
    test_u_img = np.concatenate([np.expand_dims(ux, 2), np.expand_dims(uy, 2)], 2)
    test_u_img2 = np.expand_dims(test_u_img, 0)
    fx = data['f_x'].reshape(num_node, num_node).transpose((1, 0))
    fy = data['f_y'].reshape(num_node, num_node).transpose((1, 0))
    test_f_img = -1. * np.concatenate([np.expand_dims(fx, 2), np.expand_dims(fy, 2)], 2)
    test_f_img2 = np.expand_dims(test_f_img, 0)

    data=sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/crack.mat')
    ux = data['d_x'].reshape(num_node, num_node).transpose((1, 0))  # * 1e10
    uy = data['d_y'].reshape(num_node, num_node).transpose((1, 0))  # * 1e10
    test_u_img = np.concatenate([np.expand_dims(ux, 2), np.expand_dims(uy, 2)], 2)
    test_u_img3 = np.expand_dims(test_u_img, 0)
    fx = data['f_x'].reshape(num_node, num_node).transpose((1, 0))
    fy = data['f_y'].reshape(num_node, num_node).transpose((1, 0))
    test_f_img = -1. * np.concatenate([np.expand_dims(fx, 2), np.expand_dims(fy, 2)], 2)
    test_f_img3 = np.expand_dims(test_f_img, 0)

    u_img_test23 = np.concatenate([test_u_img2,test_u_img3],0)
    f_img_test23 = np.concatenate([test_f_img2,test_f_img3],0)

    return u_img_train, f_img_train, u_img_test01, f_img_test01, u_img_test23, f_img_test23

if __name__ == '__main__':
    f = tf.placeholder(tf.float32, shape=(None, None, None, 2))
    u = tf.placeholder(tf.float32, shape=(None, None, None, 2))

    x = f
    if 0:
        with tf.variable_scope('FCN') as scope:
            for i in range(499):
                x = slim.conv2d(x, kernel_size=3, num_outputs=2, scope='conv1')
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse == True
            x = slim.conv2d(x, kernel_size=3, num_outputs=2, activation_fn=None, scope='conv1')
    else:
        x = slim.conv2d(x, kernel_size=3, num_outputs=64, scope='conv0')
        with tf.variable_scope('FCN') as scope:
            for i in range(5):
                x = slim.conv2d(x, kernel_size=3, num_outputs=64, scope='conv1')
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse == True
        x = slim.conv2d(x, kernel_size=3, num_outputs=2, activation_fn=None, scope='conv2')

    l1_loss = tf.reduce_mean(tf.abs(x - u))
    l2_loss = tf.reduce_sum((x - u) ** 2)
    weighted_l2_loss = tf.reduce_sum((x - u) ** 2 * abs(u))
    loss = l1_loss#weighted_l2_loss

    # optimizer
    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate)  #
    grads = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads)

    # initialize
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    u_img_train, f_img_train, u_img_test01, f_img_test01, u_img_test23, f_img_test23 = load_data_elem(num_node, noise_mag=0.0)

    loss_value_hist = []
    pred_error_hist = []
    pred_hist = []
    for itr in tqdm(range(10000)):
        u_input = u_img_train
        f_input = f_img_train
        feed_dict_train = {f: f_input, u: u_input}
        loss_value_i, _ = sess.run([loss, train_op], feed_dict_train)

        pred_i = sess.run(x, {f: f_input})
        pred_error_i = np.linalg.norm(pred_i - u_input) / np.linalg.norm(u_input)
        test_pred_i = sess.run(x, {f: f_img_test01})
        test_pred_error_i = np.linalg.norm(test_pred_i[0] - u_img_test01[0]) / np.linalg.norm(u_img_test01[0])

        pred_hist += [pred_i]
        loss_value_hist += [loss_value_i]
        pred_error_hist += [pred_error_i]
        print(
        "iter:{}  train_cost: {}  pred_er: {}  test_pred_er: {}".format(itr, loss_value_i, pred_error_i, test_pred_error_i))

    print('done')

    # import matplotlib.pyplot as plt
    # plt.subplot(1,2,1)
    # plt.imshow(test_pred_i[0,:,:,0], cmap='jet', interpolation='none')
    # plt.colorbar()
    # plt.subplot(1,2,2)
    # plt.imshow(test_pred_i[0,:,:,1], cmap='jet', interpolation='none')
    # plt.colorbar()
