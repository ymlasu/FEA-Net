import numpy as np
import scipy.io as sio
from tf_utils import *
import os

class Jacobi_block():
    def __init__(self):
        # NOTICE: right now for homogeneous anisotropic material only!!
        self.E = tf.Variable(1.0, tf.float32) #ref:2
        self.mu = tf.Variable(0.1, tf.float32) #ref:0.25
        # prior knowledge on the range of material property
        self.loss_E =  self.get_bound_loss(self.E, 0.1, 3)
        self.loss_mu =  self.get_bound_loss(self.mu, 0., 0.5)
        self.bound_loss = self.loss_E + self.loss_mu

        wxx, wxy, wyx, wyy, d_matrix_xx, d_matrix_yy = self.get_w_matrix()
        self.R_filter = tf.stack([tf.stack([wxx, wyx], -1), tf.stack([wyx, wyy], -1)], -1)
        D_matrix_xx = tf.ones((batch_size, num_node, num_node, 1)) * d_matrix_xx
        D_matrix_yy = tf.ones((batch_size, num_node, num_node, 1)) * d_matrix_yy
        self.D_matrix = tf.concat([D_matrix_xx, D_matrix_yy], -1)
        self.bc_mask = np.ones((batch_size, num_node, num_node, 1))
        self.bc_mask[:, 0, :, :] /= 2.
        self.bc_mask[:, -1, :, :] /= 2.
        self.bc_mask[:, :, 0, :] /= 2.
        self.bc_mask[:, :, -1, :] /= 2.
        self.omega = 2./3

    def get_bound_loss(self, var, lb, ub):
        lb_loss = tf.clip_by_value(lb-var, 0, 1000)
        ub_loss = tf.clip_by_value(var-ub, 0, 1000)
        return lb_loss+ub_loss

    def get_w_matrix(self):
        E, mu = self.E, self.mu
        cost_coef = E / 16. / (1 - mu ** 2)
        wxx = cost_coef * tf.Variable([
            [-4 * (1 - mu / 3.), 16 * mu / 3., -4 * (1 - mu / 3.)],
            [-8 * (1 + mu / 3.), 0, -8 * (1 + mu / 3.)],
            [-4 * (1 - mu / 3.), 16 * mu / 3., -4 * (1 - mu / 3.)],
        ])
        d_matrix_xx = cost_coef * 32. * (1 - mu / 3.)

        wxy = wyx = cost_coef * tf.Variable([
            [-2 * (mu + 1), 0, 2 * (mu + 1)],
            [0, 0, 0],
            [2 * (mu + 1), 0, -2 * (mu + 1)],
        ])

        wyy = cost_coef * tf.Variable([
            [-4 * (1 - mu / 3.), -8 * (1 + mu / 3.), -4 * (1 - mu / 3.)],
            [16 * mu / 3., 0, 16 * mu / 3.],
            [-4 * (1 - mu / 3.), -8 * (1 + mu / 3.), -4 * (1 - mu / 3.)],
        ])
        d_matrix_yy = cost_coef * 32. * (1 - mu / 3.)

        return wxx, wxy, wyx, wyy, d_matrix_xx, d_matrix_yy

    def LU_layers(self, input_tensor):
        padded_input = boundary_padding(input_tensor)  # for boundary consideration
        R_u = tf.nn.conv2d(input=padded_input, filter=self.R_filter, strides=[1, 1, 1, 1], padding='VALID')
        R_u_bc = R_u * self.bc_mask
        R_u_bc = tf.pad(R_u_bc[:, 1:-1, 1:-1, :], ((0,0), (1, 1), (1, 1), (0, 0)), "constant")  # for boundary consideration
        return R_u_bc


    def apply(self, f, max_itr=10):
        result = {}
        u_input = f #np.zeros((1, num_node, num_node, 2), 'float32')  # where u is unknown
        result['u_hist'] = [u_input]
        for itr in range(max_itr):
            R_u = self.LU_layers(result['u_hist'][-1])
            # u = (f - R_u) / self.D_matrix  # jacobi doesn't converge
            # jacobi formulation of linear system of equation solver
            u = self.omega * (f - R_u) / self.D_matrix + (1 - self.omega) * result['u_hist'][-1]
            result['u_hist'] += [u]

        result['final'] = result['u_hist'][-1]
        return result



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
    # linear elasticity, all steel, Yfix
    def read_mat(fn):
        data = sio.loadmat(fn)
        ux = data['d_x'].reshape(1, num_node,num_node).transpose((0, 2, 1)) * 1e11 # changed magnitude for numerical stability
        uy = data['d_y'].reshape(1, num_node,num_node).transpose((0, 2, 1)) * 1e11
        u_img = np.concatenate([np.expand_dims(ux, 3), np.expand_dims(uy, 3)], 3)
        fx = data['f_x'].reshape(1, num_node,num_node).transpose((0, 2, 1))
        fy = data['f_y'].reshape(1, num_node,num_node).transpose((0, 2, 1))
        f_img = -1. * np.concatenate([np.expand_dims(fx, 3), np.expand_dims(fy, 3)], 3)
        return u_img, f_img

    data_dir = '/home/hope-yao/Documents/MG_net/elasticity/data'
    data_fn = ['crack_size13.mat',
               'crack_size13_case1.mat',
               'crack_size13_case2.mat',
               'crack_size13_case3.mat',
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
    return np.concatenate(u_img,0), np.concatenate(f_img,0)


def visualize(loss_hist, pred_error_hist, resp_pred, resp_gt):
    import matplotlib.pyplot as plt

    BIGGER_SIZE = 16
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure()
    plt.semilogy(loss_hist, 'b-', label='loss convergence')
    # plt.semilogy([1, len(loss_hist)], [1, len(loss_hist) ** -1], 'k--', label='$O(n^{-1})$')
    # plt.semilogy([1, len(loss_hist)], [1, len(loss_hist) ** -2], 'k--', label='$O(n^{-2})$')
    plt.legend()
    plt.xlabel('network depth')
    plt.ylabel('prediction error')

    plt.figure()
    plt.semilogy(pred_error_hist, 'b-', label='error convergence')
    # plt.semilogy([1, len(loss_hist)], [1, len(loss_hist) ** -1], 'k--', label='$O(n^{-1})$')
    # plt.semilogy([1, len(loss_hist)], [1, len(loss_hist) ** -2], 'k--', label='$O(n^{-2})$')
    plt.legend()
    plt.xlabel('network depth')
    plt.ylabel('prediction error')

    BIGGER_SIZE = 12
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(resp_pred[0, :, :, 0], cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.subplot(2, 3, 4)
    plt.imshow(resp_pred[0, :, :, 1], cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.subplot(2, 3, 2)
    plt.imshow(resp_gt[0, :, :, 0], cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.subplot(2, 3, 5)
    plt.imshow(resp_gt[0, :, :, 1], cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.subplot(2, 3, 3)
    plt.imshow(resp_pred[0, :, :, 0] - resp_gt[0, :, :, 0], cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.subplot(2, 3, 6)
    plt.imshow(resp_pred[0, :, :, 1] - resp_gt[0, :, :, 1], cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.show()

if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import seaborn as sns
    from utils import creat_dir
    from tqdm import tqdm

    batch_size = 6
    num_node = 13

    u_img, f_img = load_data_elem(num_node, noise_mag=0.0)
    if 0:
        # plot out training data
        for i in range(6):
            plt.subplot(3, 6, i + 1)
            plt.imshow(f_img[i, :, :, 0].transpose((1, 0)), cmap='gray', interpolation='none')
            plt.axis('off')
        for i in range(6):
            plt.subplot(3, 6, 12 + i + 1)
            plt.imshow(u_img[i, :, :, 0].transpose((1, 0)), cmap='jet', interpolation='bilinear')
            plt.axis('off')
        for i in range(6):
            plt.subplot(3, 6, 6 + i + 1)
            plt.imshow(u_img[i, :, :, 1].transpose((1, 0)), cmap='jet', interpolation='bilinear')
            plt.axis('off')

    # build network
    f = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 2))
    u = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 2))
    jacobi = Jacobi_block()
    max_itr = 500
    jacobi_result = jacobi.apply(f, max_itr)
    l1_loss = tf.reduce_mean(tf.abs(jacobi_result['final'] - u))
    l2_loss = tf.reduce_sum((jacobi_result['final'] - u)**2)
    weighted_l2_loss = tf.reduce_sum((jacobi_result['final'] - u)**2 * abs(u))
    loss = weighted_l2_loss + jacobi.bound_loss

    # optimizer
    learning_rate = 1e-1
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
    E_pred_hist = []
    mu_pred_hist = []
    pred_error_hist = []
    pred_hist = []
    for itr in tqdm(range(10000)):
        for i in range(1):
            u_input = u_img
            f_input = f_img
            feed_dict_train = {f: f_input, u: u_input}
            loss_value_i, E_pred_i, mu_pred_i, _ = sess.run([loss, jacobi.E, jacobi.mu, train_op], feed_dict_train)
            pred_i = sess.run(jacobi_result['final'], {f: f_img})
            pred_error_i = np.linalg.norm(pred_i[0] - u_img[0]) / np.linalg.norm(u_img[0])

            pred_hist += [pred_i]
            loss_value_hist += [loss_value_i]
            E_pred_hist += [E_pred_i]
            mu_pred_hist += [mu_pred_i]
            pred_error_hist += [pred_error_i]
            print("iter:{}  train_cost: {}  pred_er: {} E_pred: {}  mu_pred: {}".format(itr, loss_value_i, pred_error_i, E_pred_i, mu_pred_i))

    print('done')

    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(pred[0, :, :, 0]), cmap='jet', interpolation='bilinear')
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.imshow(pred[0, :, :, 1]), cmap='jet', interpolation='bilinear')
    # plt.colorbar()
