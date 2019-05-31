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
        if HAS_TO_FILTER:
            #self.nicer_mask = self.apply_topology_filter(mask, self.beta)#
            nicer_mask = tf.clip_by_value(mask, 0, 1)
            beta = 10.
            self.nicer_mask = (tf.tanh(beta / 2) + tf.tanh(beta * (nicer_mask - 0.5))) / (2 * tf.tanh(beta / 2))

        else:
            self.nicer_mask = mask  #
        self.rho = rho
        self.E_1, self.mu_1, self.E_2, self.mu_2 = tf.split(self.rho,4)
        # self.E_1 = tf.clip_by_value(self.E_1, 0, 1)
        # self.E_2 = tf.clip_by_value(self.E_2, 0, 1)
        # self.mu_1 = tf.clip_by_value(self.mu_1, 0, 0.5)
        # self.mu_2 = tf.clip_by_value(self.mu_2, 0, 0.5)

        self.resp = resp
        self.bc_mask = self.get_bc_mask()
        self.d_matrix = self.get_d_matrix()
        self.omega = 2./3


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
        dmat = get_dmatrix(1, padded_mask, self.rho)
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
        # wx = R_u + self.resp * self.d_matrix
        return R_u

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
        self.pred_err = tf.reduce_mean(tf.abs(jacobi.prediction - resp_pl))#[:,:,:,0]
        self.mask_err = tf.reduce_mean(tf.abs(self.nicer_mask - mask_data))
        self.loss = self.pred_err# + self.penalty

    def get_optimizer(self):

        # if EST_MASK:
        #     # estimate mask
        #     lr = 0.01
        #     self.optimizer = tf.train.AdamOptimizer(lr, beta1=0.5)
        #     self.grads = self.optimizer.compute_gradients(self.loss, var_list=self.mask)
        #     self.train_op = self.optimizer.apply_gradients(self.grads)
        # else:
        #     # estimate rho
        #     lr = 0.001
        #     self.optimizer = tf.train.AdamOptimizer(lr, beta1=0.5)
        #     self.grads = self.optimizer.compute_gradients(self.loss, var_list=self.rho)
        #     self.train_op = self.optimizer.apply_gradients(self.grads)

        mask_lr = 0.01
        self.optimizer = tf.train.AdamOptimizer(mask_lr, beta1=0.5)
        self.mask_grads = self.optimizer.compute_gradients(self.loss, var_list=self.mask)
        self.mask_train_op = self.optimizer.apply_gradients(self.mask_grads)
        # estimate rho
        rho_lr = 1
        # self.optimizer = tf.train.AdamOptimizer(rho_lr, beta1=0.5)
        # self.rho_grads = self.optimizer.compute_gradients(self.loss, var_list=self.rho)
        # self.rho_train_op = self.optimizer.apply_gradients(self.rho_grads)
        self.rho_grads = self.optimizer.compute_gradients(self.loss, var_list=self.rho)
        ScipyOptimizerInterface = tf.contrib.opt.ScipyOptimizerInterface
        self.rho_optimizer = ScipyOptimizerInterface(self.loss, var_list=[self.rho],
                                                     var_to_bounds={self.rho: (0, 0.5)},
                                                     method='L-BFGS-B',
                                                     options = {'approx_grad':False, 'fprime':self.rho_grads,'factr':10, 'maxls':50, 'gtol':1e-05, 'disp': True, 'maxfun':150000, 'maxiter':150000}
                                                     )

def load_data_elem_3circle():
    if PROBLEM_SIZE == 13:
        num_node = 13
    elif PROBLEM_SIZE == 25:
        num_node = 25
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/3circles/2D_elastic_24by24_xy_fixed_3circle.mat')
    elif PROBLEM_SIZE == 49:
        num_node = 49
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/3circles/2D_elastic_48by48_xy_fixed_3circle.mat')
    elif PROBLEM_SIZE == 73:
        num_node = 73
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/3circles/2D_elastic_72by72_xy_fixed_3circle.mat')

    rho = [230 / 1e3, 0.36, 200 / 1e3, 0.25]
    u_img = np.concatenate([data['ux'].reshape(1, num_node, num_node, 1), data['uy'].reshape(1, num_node, num_node, 1)],
                           -1) * 1e6
    f_img = -1 * np.concatenate(
        [data['fx'].reshape(1, num_node, num_node, 1), data['fy'].reshape(1, num_node, num_node, 1)], -1) / 1e6
    mask = data['mask'].reshape(1, num_node - 1, num_node - 1, 1)
    return num_node, mask, u_img, f_img, rho

def load_data_elem_1circle():
    if PROBLEM_SIZE == 13:
        num_node = 13
        data = sio.loadmat('../data/biphase/1circle/2D_elastic_xy_fixed.mat')
    elif PROBLEM_SIZE == 25:
        num_node = 25
        data = sio.loadmat('../data/biphase/1circle/2D_elastic_24by24_xy_fixed.mat')
    elif PROBLEM_SIZE == 49:
        num_node = 49
        data = sio.loadmat('../data/biphase/1circle/2D_elastic_48by48_xy_fixed.mat')
    elif PROBLEM_SIZE == 65:
        num_node = 65
        data = sio.loadmat('../data/biphase/1circle/2D_elastic_64by64_xy_fixed.mat')

    rho = [230 / 1e3, 0.36, 200 / 1e3, 0.25]
    u_img = np.concatenate([data['ux'].reshape(1, num_node, num_node, 1), data['uy'].reshape(1, num_node, num_node, 1)], -1) * 1e6
    f_img = -1 * np.concatenate([data['fx'].reshape(1, num_node, num_node, 1), data['fy'].reshape(1, num_node, num_node, 1)], -1) / 1e6
    mask = data['mask'].reshape(1, num_node - 1, num_node - 1, 1)
    return num_node, mask, u_img, f_img, rho

def load_data_elem_2circles():
    if PROBLEM_SIZE == 13:
        pass
    elif PROBLEM_SIZE == 49:
        num_node = 49
        data = sio.loadmat('../data/biphase/2circles/2D_elastic_48by48_xy_fixed_2circle.mat')

    rho = [230 / 1e3, 0.36, 200 / 1e3, 0.25]
    u_img = np.concatenate([data['ux'].reshape(1, num_node, num_node, 1), data['uy'].reshape(1, num_node, num_node, 1)], -1) * 1e6
    f_img = -1 * np.concatenate([data['fx'].reshape(1, num_node, num_node, 1), data['fy'].reshape(1, num_node, num_node, 1)], -1) / 1e6
    mask = data['mask'].reshape(1, num_node - 1, num_node - 1, 1)
    return num_node, mask, u_img, f_img, rho

def load_data_elem_micro_noise():
    num_node = 55
    # data = np.load('../data/biphase/real_micro/add_noise/realmicro_snr_60.npy').item()
    # data = np.load('../data/biphase/real_micro/add_noise/realmicro_wo_noise.npy').item()

    data = np.load('../data/biphase/real_micro/add_noise_loadallnodes/realmicro_snr_60.npy').item()
    u_img = data['response_w_noise'] * 1e6#
    f_img = data['loading'] / 1e6
    rho = [212 / 1e3, 0.288, 230/ 1e3, 0.275]
    mask = data['mask'].reshape(1, num_node - 1, num_node - 1, 1)
    return num_node, mask[:1], u_img[:1], -f_img, rho

def load_data_elem_micro():
    if PROBLEM_SIZE == 73:
        num_node = 73
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/2D_elastic_72by72_xy_fixed_micro_pic3.mat')
        # data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/2D_elastic_72by72_xy_fixed_micro_pic2.mat')
        # data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/2D_elastic_72by72_xy_fixed_micro_pic3.mat')
        # data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/2D_elastic_72by72_xy_fixed_micro_pic4.mat')
        # data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/2D_elastic_72by72_xy_fixed_micro_pic5.mat')
        # data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/2D_elastic_72by72_xy_fixed_micro_pic6.mat')
        rho = [230 / 1e3, 0.36, 200 / 1e3, 0.25]
    elif PROBLEM_SIZE == 151:
        num_node = 151
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/res150/biphase_150by150_micro4.mat')
        rho = [212 / 1e3, 0.288, 230/ 1e3, 0.275]
    u_img = np.concatenate([data['ux'].reshape(1, num_node, num_node, 1), data['uy'].reshape(1, num_node, num_node, 1)], -1) * 1e6
    f_img = -1 * np.concatenate([data['fx'].reshape(1, num_node, num_node, 1), data['fy'].reshape(1, num_node, num_node, 1)], -1) / 1e6
    mask = data['mask'].reshape(1, num_node - 1, num_node - 1, 1)
    return num_node, mask, u_img, f_img, rho

if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    EST_MASK = 0
    PROBLEM_SIZE = 13
    FORWARD_SOLVING = 0
    HAS_TO_FILTER = 0

    num_node, mask_data, resp_data, load_data, rho = load_data_elem_1circle()#load_data_elem_1circle()#load_data_elem_micro_noise()#load_data_elem_micro()#
    # placeholders
    batch_size = 1
    load_pl = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 2))
    resp_pl = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 2)) # defined on the nodes

    # initial_mask = np.random.randint(0,2,mask_data.shape)#np.ones_like(mask_data)*0.5#mask_data#
    initial_mask = mask_data
    mask_pl = tf.Variable(initial_value=initial_mask,dtype=tf.float32, name='mask_pl') #defined on the elements
    rho_pl = tf.Variable([0.23,0.36,0.2,0.2], tf.float32)#rho#[0.1,0.1,0.1,0.1]#[0.23,0.36,0.2,0.25]

    # build network
    beta = tf.constant(1.0, tf.float32)#controls the topology mass hyper-parameter
    jacobi = Jacobi_block(num_node, load_pl, mask_pl, rho_pl, resp_pl, beta)

    #inverse generative
    jacobi.prediction = jacobi.forward_pass(resp_pl,mask_pl)
    jacobi.pred_err = tf.reduce_mean(tf.abs(jacobi.prediction - load_pl))
    jacobi.loss = jacobi.pred_err #+ jacobi.penalty
    jacobi.mask_err = tf.reduce_mean(tf.abs(jacobi.nicer_mask - mask_data))
    jacobi.get_optimizer()

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
    beta_val = 2
    for itr in tqdm(range(10001)):
        if itr % 100 == 0:
            beta_val = beta_val * 2
        idx = itr%resp_data.shape[0]
        feed_dict_train = {load_pl: load_data[idx:idx+1], resp_pl: resp_data[idx:idx+1], jacobi.beta:beta_val}#, mask_pl: mask_data
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

        num_itr = 1000
        if not itr%num_itr and itr%(num_itr*2):
            flag = 0
            if itr!=0:
                jacobi.mask = tf.clip_by_value(jacobi.mask, 0 ,1)
                jacobi.mask = tf.cast(tf.cast(jacobi.mask+0.5,tf.int32), tf.float32)
                jacobi.nicer_mask = jacobi.mask
        elif not itr%(num_itr*2):
            flag = 1
        print(itr, flag)

        if flag:
            # sess.run(jacobi.mask_train_op, feed_dict_train)
            jacobi.rho_optimizer.minimize(sess,feed_dict_train)
        else:
            # sess.run(jacobi.rho_train_op, feed_dict_train)
            jacobi.rho_optimizer.minimize(sess,feed_dict_train)

        # sess.run(jacobi.rho_train_op, feed_dict_train)

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
    #
    #     if itr%1000==0:
    #         np.save('micro4_res{}'.format(PROBLEM_SIZE),
    #                 {'loss_hist': loss_hist, 'E1_hist': E1_hist, 'E2_hist': E2_hist, 'mu1_hist': mu1_hist, 'mu2_hist': mu2_hist,
    #                  'pred_err_hist': pred_err_hist, 'pred_hist': pred_hist, 'mask_err_hist': mask_err_hist,
    #                  'mask_hist': mask_hist, 'nicer_mask_hist': nicer_mask_hist})
    #
    # plt.subplot(1, 5, 1)
    # plt.imshow(np.squeeze(resp_data[0]))
    # plt.colorbar()
    # plt.subplot(1, 5, 2)
    # plt.imshow(np.squeeze(pred_hist[-1][0]))
    # plt.colorbar()
    # plt.subplot(1, 5, 3)
    # plt.imshow(np.squeeze(mask_data[0]))
    # plt.colorbar()
    # plt.subplot(1, 5, 4)
    # plt.imshow(np.squeeze(mask_hist[-1][0]))
    # plt.colorbar()
    # plt.subplot(1, 5, 5)
    # plt.imshow(np.squeeze(nicer_mask_hist[-1][0]))
    # plt.colorbar()
    # print('done')
    #
    #
