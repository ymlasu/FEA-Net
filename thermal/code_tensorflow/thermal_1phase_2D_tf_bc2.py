import numpy as np
import scipy.io as sio
from tf_utils import *
import os

class Jacobi_block():
    def __init__(self):
        # NOTICE: right now for homogeneous anisotropic material only!!
        self.rho = tf.Variable(1., tf.float32)
        R_filter = 1/3. * np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]])
        self.R_filter = np.reshape(R_filter,(3,3,1,1)) * self.rho
        self.D_matrix  = tf.reshape(-8./3.*self.rho,(1,1,1,1))

        self.bc_mask = np.ones((batch_size, num_node, num_node, 1))
        self.bc_mask[:, 0, :, :] /= 2
        self.bc_mask[:, -1, :, :] /= 2
        self.bc_mask[:, :, 0, :] /= 2
        self.bc_mask[:, :, -1, :] /= 2

    def LU_layers(self, input_tensor):
        padded_input = boundary_padding(input_tensor)  # for boundary consideration
        # LU_u = signal.correlate2d(padded_input, heat_filter, mode='valid')  # perform convolution
        R_u = tf.nn.conv2d(input=padded_input, filter=self.R_filter, strides=[1, 1, 1, 1], padding='VALID')
        R_u_bc = R_u * self.bc_mask
        R_u_bc = tf.pad(R_u_bc[:, 1:-1, 1:-1, :], ((0,0), (1, 1), (1, 1), (0, 0)), "constant")  # for boundary consideration
        return R_u_bc


    def apply(self, f, max_itr=10):
        result = {}
        u_input = np.zeros((1, num_node, num_node, 1), 'float32')  # where u is unknown
        result['u_hist'] = [u_input]
        for itr in range(max_itr):
            R_u = self.LU_layers(result['u_hist'][-1])
            u = (f - R_u) / self.D_matrix  # jacobi formulation of linear system of equation solver
            result['u_hist'] += [u]

        result['final'] = result['u_hist'][-1]
        return result

def get_w_matrix(coef_dict):
    E, mu = coef_dict['E'], coef_dict['mu']
    cost_coef = E/16./(1-mu**2)
    wxx = cost_coef * np.asarray([
        [-4*(1 - mu / 3.),        16 * mu / 3.,             -4*(1 - mu / 3.)],
        [-8*(1 + mu / 3.),       32. * (1 - mu / 3.),       -8*(1 + mu / 3.)],
        [-4*(1 - mu / 3.),        16 * mu / 3.,             -4*(1 - mu / 3.)],
    ])

    wxy = wyx = cost_coef * np.asarray([
        [-2 * (mu + 1),        0,             2 * (mu + 1)],
        [0,                        0,                  0],
        [2 * (mu + 1),        0,             -2 * (mu + 1)],
    ])

    wyy = cost_coef * np.asarray([
        [-4 * (1 - mu / 3.),         -8 * (1 + mu / 3.),        -4 * (1 - mu / 3.)],
        [16 * mu / 3.,               32. * (1 - mu / 3.),        16 * mu / 3.],
        [-4 * (1 - mu / 3.),         -8 * (1 + mu / 3.),        -4 * (1 - mu / 3.)],
    ])
    return wxx, wxy, wyx, wyy

def np_get_D_matrix_elast(coef_dict, mode='symm'):
    # convolution with symmetric padding at boundary
    d_matrix_xx_val, d_matrix_yy_val = coef_dict['wxx'][1,1], coef_dict['wyy'][1,1]
    d_matrix_xx = d_matrix_xx_val*np.ones((num_node,num_node))
    d_matrix_yy = d_matrix_yy_val*np.ones((num_node,num_node))
    d_matrix = np.stack([d_matrix_xx, d_matrix_yy], -1)
    d_matrix[0,:] /=2
    d_matrix[-1,:] /=2
    d_matrix[:,0] /=2
    d_matrix[:,-1] /=2

    return d_matrix


def load_data_elem(num_node, noise_mag=0):
    '''loading data obtained from FEA simulation'''
    def read_mat(fn):
        data = sio.loadmat(fn)
        u_img = data['d'].reshape(1, num_node, num_node, 1).transpose((0, 2, 1, 3)) 
        f_img = data['f'].reshape(1, num_node, num_node, 1).transpose((0, 2, 1, 3)) 
        return u_img, f_img

    data_dir = '/home/hope-yao/Documents/FEA_Net/thermal/data'
    data_fn = ['bc2/case1.mat',
               'bc2/case2.mat',
               'bc2/case3.mat',
               'bc2/case4.mat',
               'bc2/case5.mat']

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
    rho = 16.
    return np.concatenate(u_img,0), np.concatenate(f_img,0), rho


if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import seaborn as sns
    from utils import creat_dir
    from tqdm import tqdm

    # build network
    batch_size = 5
    num_node = 13
    u_img, f_img, conductivity = load_data_elem(num_node=13)
    if 0:
        # plot out training data
        for i in range(5):
            plt.subplot(2, 5, i + 1)
            plt.imshow(f_img[i, :, :, 0].transpose((1, 0)))
            plt.axis('off')
        for i in range(6):
            plt.subplot(2, 5, 5 + i + 1)
            plt.imshow(u_img[i, :, :, 0].transpose((1, 0)), cmap='jet', interpolation='bilinear')
            plt.axis('off')

    f = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 1))
    u = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 1))
    jacobi = Jacobi_block()
    jacobi_result = jacobi.apply(f, max_itr=500)

    # optimizer
    jacobi_result['loss'] = loss = tf.reduce_mean(tf.abs(jacobi_result['final'] - u ))
    lr = 1.
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdamOptimizer(learning_rate)#
    grads=optimizer.compute_gradients(loss)
    train_op=optimizer.apply_gradients(grads)

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

    loss_value_hist = []
    k_value_hist = []

    for itr in tqdm(range(10000)):
        for i in range(1):
            u_input = u_img
            f_input = f_img
            feed_dict_train = {f: f_input, u: u_input}
            _, loss_value_i, k_value_i = sess.run([train_op, loss, jacobi.rho], feed_dict_train)
            print("iter:{}  train_cost: {}  k_value: {}".format(itr, np.mean(loss_value_i), k_value_i))

    print('done')
