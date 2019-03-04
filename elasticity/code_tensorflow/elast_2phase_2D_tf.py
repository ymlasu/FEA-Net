import numpy as np
import scipy.io as sio
import os
import tensorflow as tf


class Jacobi_block():
    def __init__(self, num_node, load, mask, rho, resp, beta):
        # NOTICE: right now for homogeneous anisotropic material only!!
        self.num_node = num_node
        self.load = load
        self.mask = mask
        self.beta = beta
        self.nicer_mask = mask#self.apply_topology_filter(mask, self.beta)
        self.rho = rho
        self.E_1, self.mu_1, self.E_2, self.mu_2 = tf.split(self.rho,4)
        self.resp = resp
        self.bc_mask = self.get_bc_mask()
        self.d_matrix = self.get_d_matrix()
        self.omega = 2./3

    def apply_topology_filter(self, mask, beta):
        r = np.sqrt(2)
        d = np.asarray([[np.sqrt(2), 1., np.sqrt(2.)], [1., 0., 1.], [np.sqrt(2), 1., np.sqrt(2)]])
        w = 1. - d / r
        x = self.boundary_padding(mask)
        x = tf.nn.conv2d(x, w.reshape(3,3,1,1), strides=[1,1,1,1], padding='VALID')
        x /= np.sum(w)
        # rho
        rho = ( tf.tanh(beta/2) + tf.tanh(beta*(x-0.5)) ) / (2*tf.tanh(beta/2))
        return rho

    def get_bc_mask(self):
        bc_mask = np.ones((batch_size, num_node, num_node, 1))
        bc_mask[:, 0, :, :] /= 2
        bc_mask[:, -1, :, :] /= 2
        bc_mask[:, :, 0, :] /= 2
        bc_mask[:, :, -1, :] /= 2
        return bc_mask

    def boundary_padding(self,x):
        ''' special symmetric boundary padding '''
        left = x[:, :, 1:2, :]
        right = x[:, :, -2:-1, :]
        upper = tf.concat([x[:, 1:2, 1:2, :], x[:, 1:2, :, :], x[:, 1:2, -2:-1, :]], 2)
        down = tf.concat([x[:, -2:-1, 1:2, :], x[:, -2:-1, :, :], x[:, -2:-1, -2:-1, :]], 2)
        padded_x = tf.concat([left, x, right], 2)
        padded_x = tf.concat([upper, padded_x, down], 1)
        return padded_x

    def get_d_matrix(self):
        from tf_ops_cpp.mask_elast_conv import get_dmatrix
        # be careful here
        padded_mask = self.boundary_padding(self.nicer_mask)
        dmat = get_dmatrix(padded_mask, self.rho)
        return dmat

    def LU_layers(self, input_tensor, mask_tensor):
        from tf_ops_cpp.mask_elast_conv import mask_conv
        padded_input = self.boundary_padding(input_tensor)  # for boundary consideration
        padded_mask = self.boundary_padding(mask_tensor)  # for boundary consideration
        R_u = mask_conv(padded_input, padded_mask, self.rho)
        R_u_bc = R_u * self.bc_mask # boundary_corrrect
        R_u_bc = tf.pad(R_u_bc[:, 1:-1, 1:-1, :], ((0,0), (1, 1),(1, 1), (0, 0)), "constant")  # for boundary consideration
        return R_u_bc

    def forward_pass(self,resp, mask):
        R_u = self.LU_layers(resp, mask)
        wx = self.R_u + self.resp * self.d_matrix
        return wx

    def apply(self, max_itr=10):
        result = {}
        u0 = np.zeros((1, num_node, num_node, 1), 'float32')  # where u is unknown
        result['u_hist'] = [u0]
        for itr in range(max_itr):
            R_u = self.LU_layers(result['u_hist'][-1], self.nicer_mask)
            # u = (self.load - R_u) / self.d_matrix  # jacobi formulation of linear system of equation solver
            u = self.omega * (self.load - R_u) / self.d_matrix + (1 - self.omega) * result['u_hist'][-1]
            result['u_hist'] += [u]

        self.u_hist = result['u_hist']
        self.prediction = result['u_hist'][-1]
        return result

    def get_loss(self):
        self.pred_err = tf.reduce_mean(tf.abs(jacobi.prediction - resp_pl))
        # k = 0.2
        # self.penalty = tf.reduce_mean(
        #     tf.abs(k*self.mask) + tf.abs(k*(1 - self.mask)) - 0.5*k - tf.abs(k*(0.5 - self.mask)) )  # a W shaped penalty
        self.mask_err = tf.reduce_mean(tf.abs(self.nicer_mask - mask_data))
        self.loss = self.pred_err# + self.penalty

    def get_optimizer(self):

        if 0:
            lr = 0.1
            # self.optimizer = tf.train.MomentumOptimizer(lr, 0.9)  #
            self.optimizer = tf.train.AdamOptimizer(lr, beta1=0.5)
            self.grads = self.optimizer.compute_gradients(self.loss, var_list=self.mask)#self.rho)
            self.train_op = self.optimizer.apply_gradients(self.grads)
        else:
            # ScipyOptimizerInterface = tf.contrib.opt.ScipyOptimizerInterface
            # ScipyOptimizerInterface(self.loss, var_list=[jacobi.rho], var_to_bounds={self.rho: ([0, 1], np.infty)},
            #                         method='fmin_cg')

            lr = 0.001
            self.optimizer = tf.train.AdamOptimizer(lr, beta1=0.5)
            self.grads = self.optimizer.compute_gradients(self.loss, var_list=self.rho)
            self.train_op = self.optimizer.apply_gradients(self.grads)

def load_data_elem_s12():
    num_node = 13
    data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/2D_elastic_xy_fixed.mat')
    rho = [230 / 1e3, 0.36, 200 / 1e3, 0.25]
    u_img = np.concatenate([data['ux'].reshape(1, 13, 13, 1), data['uy'].reshape(1, 13, 13, 1)], -1) * 1e6
    f_img = -1 * np.concatenate([data['fx'].reshape(1, 13, 13, 1), data['fy'].reshape(1, 13, 13, 1)], -1) / 1e6
    mask = data['mask'].reshape(1, num_node - 1, num_node - 1, 1)
    return num_node, mask, u_img, f_img, rho

if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    num_node, mask_data, resp_data, load_data, rho = load_data_elem_s12()#get_data()

    # placeholders
    batch_size = 1
    load_pl = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 2))
    resp_pl = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 2)) # defined on the nodes
    initial_mask =  mask_data
    # initial_mask = np.ones_like(mask_data)#*0.5
    mask_pl = tf.Variable(initial_value=initial_mask,dtype=tf.float32, name='mask_pl') #defined on the elements
    rho_pl = tf.Variable([0.1,0.1,0.1,0.1], tf.float32)#rho
    beta = tf.constant(1.0, tf.float32)#controls the topology mass hyper-parameter

    # build network
    jacobi = Jacobi_block(num_node, load_pl, mask_pl, rho_pl, resp_pl, beta)
    if 1: # forward solving
        jacobi.apply(max_itr=200)
        jacobi.get_loss()
        jacobi.get_optimizer()

    else: #inverse generative
        jacobi.prediction = jacobi.forward_pass(resp_pl,mask_pl)
        jacobi.pred_err = tf.reduce_mean(tf.abs(jacobi.prediction - load_pl))
        # k = 0.2
        # jacobi.penalty = tf.reduce_mean(
        #     tf.abs(k*mask_pl) + tf.abs(k*(1 - mask_pl)) - 0.5*k - tf.abs(k*(0.5 - mask_pl)) )  # a W shaped penalty
        jacobi.mask_err = tf.reduce_mean(tf.abs(mask_pl - mask_data))
        jacobi.loss = jacobi.pred_err #+ jacobi.penalty
        # jacobi.get_optimizer()

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

    loss_hist = []
    pred_err_hist = []
    pred_hist = []
    E1_hist = []
    E2_hist = []
    mu1_hist = []
    mu2_hist = []
    mask_err_hist = []
    mask_hist = []
    nicer_mask_hist = []
    beta_val = 0.5
    for itr in tqdm(range(1000)):
        if itr % 1000 == 0:
            beta_val = beta_val * 1.5
        feed_dict_train = {load_pl: load_data, resp_pl: resp_data, jacobi.beta:beta_val}#, mask_pl: mask_data
        loss_value_i, pred_err_i, pred_i, mask_err_i, mask_i, nicer_mask_i, E1_value_i, mu1_value_i, E2_value_i, mu2_value_i = \
                                                                                        sess.run([jacobi.loss,
                                                                                                    jacobi.pred_err,
                                                                                                    jacobi.prediction,
                                                                                                    jacobi.mask_err,
                                                                                                    jacobi.mask,
                                                                                                    jacobi.nicer_mask,
                                                                                                    jacobi.E_1,
                                                                                                    jacobi.mu_1,
                                                                                                    jacobi.E_2,
                                                                                                    jacobi.mu_2,],
                                                                                                   feed_dict_train)
        print("iter:{}  pred_err: {} mask_err: {}  E1_value: {}  E2_value: {}  mu1_value: {}  mu2_value: {}".
              format(itr, np.mean(pred_err_i),
              np.mean(mask_err_i),
              E1_value_i,
              E2_value_i,
              mu1_value_i,
              mu2_value_i))
        sess.run(jacobi.train_op, feed_dict_train)
        # jacobi.optimizer.minimize(sess,feed_dict_train)

        loss_hist += [loss_value_i]
        E1_hist += [E1_value_i]
        E2_hist += [E2_value_i]
        mu1_hist += [mu1_value_i]
        mu2_hist += [mu2_value_i]
        pred_err_hist += [pred_err_i]
        pred_hist += [pred_i]
        mask_err_hist += [mask_err_i]
        mask_hist += [mask_i]
        nicer_mask_hist += [nicer_mask_i]

    plt.subplot(1, 5, 1)
    plt.imshow(np.squeeze(resp_data[0]))
    plt.colorbar()
    plt.subplot(1, 5, 2)
    plt.imshow(np.squeeze(pred_hist[-1][0]))
    plt.colorbar()
    plt.subplot(1, 5, 3)
    plt.imshow(np.squeeze(mask_data[0]))
    plt.colorbar()
    plt.subplot(1, 5, 4)
    plt.imshow(np.squeeze(mask_hist[-1][0]))
    plt.colorbar()
    plt.subplot(1, 5, 5)
    plt.imshow(np.squeeze(nicer_mask_hist[-1][0]))
    plt.colorbar()
    print('done')