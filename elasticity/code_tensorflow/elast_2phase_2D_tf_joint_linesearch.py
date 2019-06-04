import numpy as np
import scipy.io as sio
import os
import tensorflow as tf
from scipy import ndimage


class Jacobi_block():
    def __init__(self, num_node, load, resp, train_var_pl, beta):
        # NOTICE: right now for homogeneous anisotropic material only!!
        self.num_node = num_node
        self.load = load
        self.resp = resp
        self.mask = train_var_pl[:-4]
        self.rho = train_var_pl[-4:]
        self.bc_mask = self.get_bc_mask()

        self.beta = beta
        if HAS_TO_FILTER:
            #self.nicer_mask = self.apply_topology_filter(mask, self.beta)#
            self.nicer_mask = tf.clip_by_value(self.mask, 0, 1)
            # nicer_mask = mask
            beta = 10.
            self.nicer_mask = (tf.tanh(beta / 2) + tf.tanh(beta * (self.nicer_mask - 0.5))) / (2 * tf.tanh(beta / 2))

        else:
            self.nicer_mask = self.mask  #

        #inverse generative
        self.prediction = self.forward_pass(self.resp,self.nicer_mask)
        self.pred_err = tf.reduce_mean((self.prediction - self.load)**2)
        self.loss = self.pred_err #+ jacobi.penalty
        self.mask_err = tf.reduce_mean(tf.abs(self.nicer_mask - mask_data))

        self.trainable_var_pl = train_var_pl
        self.grads = tf.gradients(self.loss, self.trainable_var_pl)
        # self.hessian = tf.hessians(self.loss, self.trainable_var_pl)

        # reference: https://stackoverflow.com/questions/35266370/tensorflow-compute-hessian-matrix-and-higher-order-derivatives/37666032
        # def cons(x):
        #     return tf.constant(x, dtype=tf.float32)
        #
        # def compute_hessian(f, vars):
        #     mat = []
        #     N = (num_node-1)**2 + 4
        #     for i in range(N):
        #         v1 = vars[i]
        #         temp = []
        #         for j in range(N):
        #             v2 = vars[j]
        #             # computing derivative twice, first w.r.t v2 and then w.r.t v1
        #             temp.append(tf.gradients(tf.gradients(f, v2)[0], v1)[0])
        #         temp = [cons(0) if t == None else t for t in
        #                 temp]  # tensorflow returns None when there is no gradient, so we replace None with 0
        #         temp = tf.pack(temp)
        #         mat.append(temp)
        #     mat = tf.pack(mat)
        #     return mat
        #
        # self.rho_hessian = compute_hessian(self.loss, self.trainable_var_pl)

        # self.rho = rho
        # self.E_1, self.mu_1, self.E_2, self.mu_2 = tf.split(self.rho,4)
        # self.E_1 = tf.clip_by_value(self.E_1, 0.1, 0.5)
        # self.E_2 = tf.clip_by_value(self.E_2, 0.1, 0.5)
        # self.mu_1 = tf.clip_by_value(self.mu_1, 0.1, 0.5)
        # self.mu_2 = tf.clip_by_value(self.mu_2, 0.1, 0.5)

        self.init()

    def init(self):
        # initialize
        FLAGS = tf.app.flags.FLAGS
        tfconfig = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        init = tf.global_variables_initializer()
        self.sess.run(init)

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
        bc_mask = np.ones((1, num_node, num_node, 1))
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

    def LU_layers(self, input_tensor, mask_tensor):
        from tf_ops_cpp.mask_elast_conv import mask_conv
        padded_input = self.boundary_padding(input_tensor)  # for boundary consideration
        padded_mask = self.boundary_padding(tf.reshape(mask_tensor,(1,num_node-1,num_node-1,1)))  # for boundary consideration
        R_u = mask_conv(padded_input, padded_mask, self.rho)
        R_u_bc = R_u * self.bc_mask # boundary_corrrect
        R_u_bc = tf.pad(R_u_bc[:, 1:-1, 1:-1, :], ((0,0), (1, 1),(1, 1), (0, 0)), "constant")  # for boundary consideration
        return R_u_bc

    def forward_pass(self,resp, mask):
        R_u = self.LU_layers(resp, mask)
        return R_u

    def get_loss(self,v):
        feed_dict = {self.trainable_var_pl: v}
        loss_np = self.sess.run(self.loss, feed_dict)
        return loss_np

    def get_grad(self, v):
        feed_dict = {self.trainable_var_pl: v}
        grads_np = self.sess.run(self.grads, feed_dict)[0]
        phase_grads = grads_np[:-4].reshape(num_node-1,num_node-1)
        blurred_phase_grads = ndimage.gaussian_filter(phase_grads, sigma=0.01)
        blurred_grads_np = np.concatenate([blurred_phase_grads.flatten(), grads_np[-4:]],-1)
        return blurred_grads_np

    def get_hessian(self, v):
        feed_dict = {self.trainable_var_pl: v}
        hessian_np = self.sess.run(self.hessian, feed_dict)
        return hessian_np


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
    PROBLEM_SIZE = 49
    FORWARD_SOLVING = 0
    HAS_TO_FILTER = 0

    num_node, mask_data, resp_data, load_data, rho = load_data_elem_3circle()#load_data_elem_1circle()#load_data_elem_micro_noise()#load_data_elem_micro()#
    load_pl= load_data.astype('float32')
    resp_pl= resp_data.astype('float32')
    # mask_data= mask_data[:, 23:25,23:25,:].astype('float32')
    # load_pl= load_data[:, 23:26,23:26,:].astype('float32')
    # resp_pl= resp_data[:, 23:26,23:26,:].astype('float32')
    # num_node= 26-23

    tt_num_elems = (num_node-1)**2
    initial_mask = np.random.rand(tt_num_elems).astype('float32')
    initial_rho = np.random.rand(4).astype('float32')#np.asarray([0.1,0.1,0.1,0.1])
    train_var_np = np.concatenate([initial_mask, initial_rho],-1)

    initial_mask = mask_data.flatten()
    initial_rho = np.asarray([0.23,0.36,0.2,0.25])#
    train_var_ref = np.concatenate([initial_mask, initial_rho],-1)

    train_var_pl = tf.Variable(initial_value=train_var_np,dtype=tf.float32, name='mask_pl') #defined on the elements

    # build network
    beta = tf.constant(1.0, tf.float32)#controls the topology mass hyper-parameter
    jacobi = Jacobi_block(num_node, load_pl, resp_pl, train_var_pl, beta)

    from scipy.optimize import minimize, fmin_tnc
    # result = minimize(jacobi.loss, jacobi.trainable_var_pl, method='Newton-CG',
    #                        jac=jacobi.grads, hess=jacobi.hessian,
    #                        options={'xtol': 1e-8, 'disp': True})
    result = fmin_tnc(jacobi.get_loss,
                      train_var_np,
                      fprime=jacobi.get_grad,
                      bounds=zip([0]*(tt_num_elems+4), [1]*(tt_num_elems+4)),
                      stepmx=100,
                      pgtol=1e-9,
                      ftol=1e-15,
                      maxfun=20000,
                      disp='True')

    print(jacobi.get_loss(train_var_np))
    print(jacobi.get_loss(result[0]))
    print(jacobi.get_loss(train_var_ref))

    print(np.mean(np.abs(jacobi.get_grad(train_var_np))))
    print(np.mean(np.abs(jacobi.get_grad(result[0]))))
    print(np.mean(np.abs(jacobi.get_grad(train_var_ref))))

    plt.subplot(1,2,1)
    plt.imshow(result[0][:-4].reshape(num_node-1,num_node-1))
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(np.round(result[0][:-4].reshape(num_node-1,num_node-1)))
    plt.show()
    print(result[0][-4:])
    pass