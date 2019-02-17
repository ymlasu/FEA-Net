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
        self.nicer_mask = self.apply_topology_filter(mask, self.beta)
        self.rho = rho
        self.rho_1, self.rho_2 = tf.split(self.rho,2)
        self.resp = resp
        self.bc_mask = self.get_bc_mask()
        self.d_matrix = self.get_d_matrix()

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
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        self.elem_mask = tf.pad(self.nicer_mask, paddings, "SYMMETRIC")
        node_filter = np.asarray([[1 / 4.] * 2] * 2)
        node_filter = np.expand_dims(np.expand_dims(node_filter,-1),-1)

        self.node_mask_1 = tf.nn.conv2d(self.elem_mask, node_filter, strides=[1,1,1,1], padding='VALID')
        self.node_mask_2 = tf.nn.conv2d(tf.ones_like(self.elem_mask) - self.elem_mask, node_filter, strides=[1,1,1,1], padding='VALID')
        d_matrix = (self.rho_1 * self.node_mask_1 + self.rho_2 * self.node_mask_2) * (-8. / 3)
        d_matrix *= self.bc_mask
        return d_matrix

    def LU_layers(self, input_tensor, mask_tensor):
        from tf_ops_cpp.mask_conv import mask_conv
        padded_input = self.boundary_padding(input_tensor)  # for boundary consideration
        padded_mask = self.boundary_padding(mask_tensor)  # for boundary consideration
        R_u = mask_conv(padded_input, padded_mask, self.rho)
        R_u_bc = R_u * self.bc_mask # boundary_corrrect
        R_u_bc = tf.pad(R_u_bc[:, :, 1:-1, :], ((0,0), (0, 0),(1, 1), (0, 0)), "constant")  # for boundary consideration
        return R_u_bc

    def forward_pass(self,resp, mask):
        R_u = self.LU_layers(resp, mask)
        wx = R_u + self.resp * self.d_matrix
        return wx

    def apply(self, max_itr=10):
        result = {}
        u0 = np.zeros((1, num_node, num_node, 1), 'float32')  # where u is unknown
        result['u_hist'] = [u0]
        for itr in range(max_itr):
            R_u = self.LU_layers(result['u_hist'][-1], self.nicer_mask)
            u = (self.load - R_u) / self.d_matrix  # jacobi formulation of linear system of equation solver
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

        if 1:
            lr = 0.1
            # self.optimizer = tf.train.MomentumOptimizer(lr, 0.9)  #
            self.optimizer = tf.train.AdamOptimizer(lr, beta1=0.5)
            self.grads = self.optimizer.compute_gradients(self.loss, var_list=self.mask)#self.rho)
            self.train_op = self.optimizer.apply_gradients(self.grads)
        else:
            ScipyOptimizerInterface = tf.contrib.opt.ScipyOptimizerInterface
            ScipyOptimizerInterface(self.loss, var_list=[jacobi.rho], var_to_bounds={self.rho: ([1, 2], np.infty)},
                                    method='fmin_cg')
            # self.train_op = optimizer.apply_gradients(self.grads)

def load_data_elem_s24():
    if 0:
        f_img = sio.loadmat('/home/hope-yao/Documents/MG_net/data/heat_transfer/Downloads/f_image.mat')['f_image']
        u_img = sio.loadmat('/home/hope-yao/Documents/MG_net/data/heat_transfer/Downloads/u_image.mat')['u_image']
        mask = sio.loadmat('/home/hope-yao/Documents/MG_net/data/heat_transfer/Downloads/ind2.mat')['ind2']
        rho_1, rho_2 =  205., 16.
        num_node = 25
        f_img = f_img.reshape(1,num_node,num_node,1).transpose(0,2,1,3)
        u_img = u_img.reshape(1,num_node,num_node,1).transpose(0,2,1,3)
        mask = mask.reshape(1,num_node-1,num_node-1,1).transpose(0,2,1,3)

    data = sio.loadmat('/home/hope-yao/Documents/MG_net/data/heat_transfer/Downloads/biphase_24_24.mat')
    rho_1, rho_2 = 205., 16.
    num_node = 25
    f_img = data['f_image'].reshape(1, num_node,num_node, 1)
    u_img = data['u_image'].reshape(1, num_node,num_node, 1)
    mask = data['mask'].reshape(1, num_node-1,num_node-1, 1)
    return num_node, u_img, f_img, mask, rho_1, rho_2

def load_data_elem_s12_more():
    rho_1, rho_2 =  32., 16.
    num_node = 13
    f_img = np.zeros((10,num_node,num_node,1))
    u_img = np.zeros((10,num_node,num_node,1))
    mask = np.zeros((10,num_node-1,num_node-1,1))
    for i in range(1,11,1):
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/thermal/data/bi_phase_data/res12/biphase_12_12_ro16_ro_32_case{}.mat'.format(i))
        f_img[i-1] = data['f_image'].reshape(1, num_node, num_node, 1)
        u_img[i-1] = data['u_image'].reshape(1, num_node, num_node, 1)
        mask[i-1] = data['mask'].reshape(1, num_node - 1, num_node - 1, 1)

def load_data_elem_s12():
    if 1:
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/thermal/data/bi_phase_data/res12/biphase_12_12_ro16_ro_32.mat')
        rho_1, rho_2 =  32., 16.
    else:
        data = sio.loadmat('/home/hope-yao/Documents/MG_net/data/heat_transfer/Downloads/biphase_12_12_new.mat')
        rho_1, rho_2 = 205., 16.
    num_node = 13
    f_img = data['f_image'].reshape(1, num_node,num_node, 1)
    u_img = data['u_image'].reshape(1, num_node,num_node, 1)
    mask = data['mask'].reshape(1, num_node-1,num_node-1, 1)
    return num_node, u_img, f_img, mask, rho_1, rho_2

def load_data_elem():
    '''loading data obtained from FEA simulation'''
    # data = sio.loadmat('./data/heat_transfer_1phase/matrix.mat')
    # f = data['matrix'][0][0][1]
    # A = data['matrix'][0][0][0]
    num_node = 66
    # NEW MULTI CIRCLE CASE
    data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/thermal/data/bc1/3circle_center_25_25_rad_17_center_55_55_rad_7_center_55_25_rad_7.mat')
    mask = np.ones((1, num_node-1, num_node-1, 1))
    for i in range(num_node-1):
        for j in range(num_node-1):
            if (i-24)**2+(j-24)**2<17**2 or (i-24)**2+(j-54)**2<7**2 or (i-54)**2+(j-54)**2<7**2:
                mask[:, i, j, :] = 0

    A = data['K']
    f = data['f']
    u = np.linalg.solve(A, f)
    u_img = u.reshape(1, num_node,num_node, 1)
    f_img = f.reshape(1, num_node,num_node, 1)
    mask = mask.transpose(0,2,1,3)
    rho_1, rho_2 = 205., 16
    if 0:
        plt.subplot(1, 2, 1)
        plt.imshow(f_img[0, :, :, 0])
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(u_img[0, :, :, 0], cmap='jet', interpolation='bilinear')
        plt.axis('off')
    return num_node, np.asarray(u_img,'float32'), np.asarray(f_img,'float32'), np.asarray(mask, 'float32'), rho_1, rho_2


if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    num_node, resp_data, load_data, mask_data, rho_1, rho_2 = load_data_elem_s12()

    # placeholders
    batch_size = 1
    load_pl = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 1))
    resp_pl = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 1)) # defined on the nodes
    # initial_mask =  mask_data
    initial_mask = np.ones_like(mask_data)#*0.5
    mask_pl = tf.Variable(initial_value=initial_mask,dtype=tf.float32, name='mask_pl') #defined on the elements
    rho_pl = tf.Variable([rho_1,rho_2], tf.float32)#tf.placeholder(tf.float32,shape=(2))
    beta = tf.constant(1.0, tf.float32)#controls the topology mass hyper-parameter

    # build network
    jacobi = Jacobi_block(num_node, load_pl, mask_pl, rho_pl, resp_pl, beta)
    if 0: # forward solving
        jacobi.apply(max_itr=300)
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
    k1_hist = []
    k2_hist = []
    mask_err_hist = []
    mask_hist = []
    nicer_mask_hist = []
    beta_val = 0.5
    for itr in tqdm(range(10000)):
        if itr % 1000 == 0:
            beta_val = beta_val * 1.5
        feed_dict_train = {load_pl: load_data, resp_pl: resp_data, jacobi.beta:beta_val}#, mask_pl: mask_data
        sess.run(jacobi.train_op, feed_dict_train)
        # jacobi.optimizer.minimize(sess,feed_dict_train)
        loss_value_i, pred_err_i, pred_i, mask_err_i, mask_i, nicer_mask_i, k1_value_i, k2_value_i = sess.run([jacobi.loss,
                                                                                                    jacobi.pred_err,
                                                                                                    jacobi.prediction,
                                                                                                    jacobi.mask_err,
                                                                                                    jacobi.mask,
                                                                                                    jacobi.nicer_mask,
                                                                                                    jacobi.rho_1,
                                                                                                    jacobi.rho_2],
                                                                                                   feed_dict_train)
        print("iter:{}  pred_err: {} mask_err: {}  k1_value: {}  k2_value: {}".format(itr, np.mean(pred_err_i), np.mean(mask_err_i), k1_value_i, k2_value_i))
        loss_hist += [loss_value_i]
        k1_hist += [k1_value_i]
        k2_hist += [k2_value_i]
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