import numpy as np
import scipy.io as sio
import os
import tensorflow as tf

from tf_ops_cpp.mask_elast_conv import get_dmatrix
from tf_ops_cpp.mask_elast_conv import mask_conv


class Jacobi_block():
    def __init__(self, num_node, load, mask, rho, resp, beta):
        # NOTICE: right now for homogeneous anisotropic material only!!
        self.u_in = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 2))
        self.num_node = num_node
        self.load = load
        self.mask = mask
        self.beta = beta
        if HAS_TO_FILTER:
            self.nicer_mask = self.apply_topology_filter(mask, self.beta)#
        else:
            self.nicer_mask = mask  #
        self.rho = rho
        self.E_1, self.mu_1, self.E_2, self.mu_2 = tf.split(self.rho,4)
        self.E_1 = tf.clip_by_value(self.E_1, 0, 1)
        self.E_2 = tf.clip_by_value(self.E_2, 0, 1)
        self.mu_1 = tf.clip_by_value(self.mu_1, 0, 0.5)
        self.mu_2 = tf.clip_by_value(self.mu_2, 0, 0.5)

        self.resp = resp
        self.bc_mask = self.get_bc_mask()
        self.d_matrix = self.get_d_matrix()
        self.omega = 2

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
        bc_mask = np.ones((batch_size, num_node, num_node, 2))
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
        # be careful here
        padded_mask = self.boundary_padding(self.nicer_mask)
        dmat = get_dmatrix(padded_mask, self.rho)
        return dmat

    def FEA_conv(self, input_tensor, mask_tensor):
        padded_input = self.boundary_padding(input_tensor)  # for boundary consideration
        padded_mask = self.boundary_padding(mask_tensor)  # for boundary consideration
        R_u = mask_conv(padded_input, padded_mask, self.rho)
        R_u_bc = R_u * self.bc_mask # boundary_corrrect
        R_u_bc = tf.pad(R_u_bc[:, 1:-1, 1:-1, :], ((0,0), (1, 1),(1, 1), (0, 0)), "constant")  # for boundary consideration
        return R_u_bc

    def forward_pass(self,resp, mask):
        return self.FEA_conv(resp, mask)

    def apply(self):
        wx = self.FEA_conv(self.u_in, self.nicer_mask)
        self.u_out = self.omega * (self.load - wx) * self.d_matrix +  self.u_in

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
        # data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/3circles/2D_elastic_72by72_xy_fixed_3circle.mat')
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/3circles/2D_elastic_72by72_xy_fixed_3circle_xyforce2.mat')

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
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/1circle/2D_elastic_xy_fixed.mat')
    elif PROBLEM_SIZE == 25:
        num_node = 25
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/1circle/2D_elastic_24by24_xy_fixed.mat')
    elif PROBLEM_SIZE == 49:
        num_node = 49
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/1circle/2D_elastic_48by48_xy_fixed.mat')
    elif PROBLEM_SIZE == 65:
        num_node = 65
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/1circle/2D_elastic_64by64_xy_fixed.mat')

    # rho = [230 / 1e3, 0.36, 200 / 1e3, 0.25]
    rho = [0.22999817230, 0.35994583, 0.20000985, 0.24995165]
    u_img = np.concatenate([data['ux'].reshape(1, num_node, num_node, 1), data['uy'].reshape(1, num_node, num_node, 1)], -1) * 1e6
    f_img = -1 * np.concatenate([data['fx'].reshape(1, num_node, num_node, 1), data['fy'].reshape(1, num_node, num_node, 1)], -1) / 1e6
    mask = data['mask'].reshape(1, num_node - 1, num_node - 1, 1)
    return num_node, mask, u_img, f_img, rho

def load_data_elem_micro():
    if PROBLEM_SIZE == 73:
        num_node = 73
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/2D_elastic_72by72_xy_fixed_micro_pic1_xyforce2.mat')
        # data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/2D_elastic_72by72_xy_fixed_micro_pic1.mat')
        # data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/2D_elastic_72by72_xy_fixed_micro_pic2.mat')
        # data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/2D_elastic_72by72_xy_fixed_micro_pic3.mat')
        # data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/2D_elastic_72by72_xy_fixed_micro_pic4.mat')
        # data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/2D_elastic_72by72_xy_fixed_micro_pic5.mat')
        # data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/2D_elastic_72by72_xy_fixed_micro_pic6.mat')
    elif PROBLEM_SIZE == 141:
        num_node = 141
        data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/biphase/real_micro/2D_elastic_140by140_xy_fixed_micro.mat')

    rho = [0.22999817230, 0.35994583, 0.20000985, 0.24995165]
    # rho = [230 / 1e3, 0.36, 200 / 1e3, 0.25]
    u_img = np.concatenate([data['ux'].reshape(1, num_node, num_node, 1), data['uy'].reshape(1, num_node, num_node, 1)], -1) * 1e6
    f_img = -1 * np.concatenate([data['fx'].reshape(1, num_node, num_node, 1), data['fy'].reshape(1, num_node, num_node, 1)], -1) / 1e6
    mask = data['mask'].reshape(1, num_node - 1, num_node - 1, 1)
    return num_node, mask, u_img, f_img, rho

def load_data_elem_2circle():
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

def load_data_forward(inc):
    if inc == 1:
        data = sio.loadmat('../data/biphase/forward/2D_elastic_25by25_xy_fixed_1circle.mat')
    if inc == 2:
        data = sio.loadmat('../data/biphase/forward/2D_elastic_25by25_xy_fixed_2circle.mat')
    if inc == 3:
        data = sio.loadmat('../data/biphase/forward/2D_elastic_25by25_xy_fixed_3circle.mat')
    num_node = 25
    u_img = np.concatenate([data['ux'][:num_node,:num_node].transpose().reshape(1, num_node, num_node, 1), data['uy'][:num_node,:num_node].transpose().reshape(1, num_node, num_node, 1)], -1) * 1e6
    f_img = -1 * np.concatenate([data['fx'][:num_node,:num_node].transpose().reshape(1, num_node, num_node, 1), data['fy'][:num_node,:num_node].transpose().reshape(1, num_node, num_node, 1)], -1) / 1e6
    mask = data['mask'].reshape(1, num_node - 1, num_node - 1, 1)
    rho = [230 / 1e3, 0.36, 200 / 1e3, 0.25]
    return num_node, mask, u_img, f_img, rho

def load_data_elem_sp():
    num_node = PROBLEM_SIZE
    data = sio.loadmat('/home/hope-yao/Documents/FEA_Net/elasticity/data/singlephase/single_phase_xyfixed_case4.mat')
    u_img = np.concatenate([data['ux'][:num_node,:num_node].transpose().reshape(1, num_node, num_node, 1), data['uy'][:num_node,:num_node].transpose().reshape(1, num_node, num_node, 1)], -1) * 1e6
    f_img = -1 * np.concatenate([data['fx'][:num_node,:num_node].transpose().reshape(1, num_node, num_node, 1), data['fy'][:num_node,:num_node].transpose().reshape(1, num_node, num_node, 1)], -1) / 1e6
    rho = [200 / 1e3, 0.25, 200 / 1e3, 0.25]
    mask = np.ones((1, num_node - 1, num_node - 1, 1))
    return num_node, mask, u_img, f_img, rho


if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    EST_MASK = 0
    PROBLEM_SIZE = 25#61#
    FORWARD_SOLVING = 1
    HAS_TO_FILTER = 0

    num_node, mask_data, resp_data, load_data, rho = load_data_elem_3circle()#load_data_forward(1)
    # num_node, mask_data, resp_data, load_data, rho = load_data_elem_sp()

    # placeholders
    batch_size = 1
    load_pl = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 2))
    resp_pl = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 2)) # defined on the nodes
    initial_mask = mask_data
    mask_pl = tf.Variable(initial_value=initial_mask, dtype=tf.float32, name='mask_pl')  # defined on the elements
    rho_pl = tf.Variable(rho, tf.float32)#rho

    # build network
    beta = tf.constant(1.0, tf.float32)#controls the topology mass hyper-parameter
    jacobi = Jacobi_block(num_node, load_pl, mask_pl, rho_pl, resp_pl, beta)

    max_itr = 20000
    result=jacobi.apply()

    # initialize
    # FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    loss_hist = []
    pred_hist = []
    itr_hist = []
    pred_i = np.zeros((1,num_node,num_node,2))
    for itr in tqdm(range(0, max_itr, 10)):
        feed_dict_train = {jacobi.u_in: pred_i, load_pl: load_data.tolist(), mask_pl: mask_data.tolist()}
        pred_i = sess.run(jacobi.u_out, feed_dict_train)
        pred_err_i = np.mean(np.abs(pred_i-resp_data))
        print("iter:{}  pred_err: {}". format(itr, np.mean(pred_err_i)))
        pred_hist += [pred_i]
        loss_hist += [np.mean(pred_err_i)]
        itr_hist += [itr]

        np.save('bp_inf_conv_inc3',{'iter': itr_hist, 'mask_hist': pred_hist, 'loss_hist': loss_hist})
