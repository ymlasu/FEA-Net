import numpy as np
import scipy.io as sio
from tf_utils import *
import os
import tensorflow as tf


class Jacobi_block():
    def __init__(self, num_node, load, mask):
        # NOTICE: right now for homogeneous anisotropic material only!!
        self.num_node = num_node
        self.load = load
        self.mask = mask
        self.rho_1 = tf.constant(16.) #tf.Variable(1., tf.float32)
        self.rho_2 = tf.constant(205.) #tf.Variable(1., tf.float32)
        self.bc_mask = self.get_bc_mask()
        self.d_matrix = self.get_d_matrix()

    def get_bc_mask(self):
        bc_mask = np.ones((batch_size, num_node, num_node, 1))
        bc_mask[:, 0, :, :] /= 2
        bc_mask[:, -1, :, :] /= 2
        bc_mask[:, :, 0, :] /= 2
        bc_mask[:, :, -1, :] /= 2
        return bc_mask

    def get_d_matrix(self):
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        self.elem_mask = tf.pad(self.mask, paddings, "SYMMETRIC")
        node_filter = np.asarray([[1 / 4.] * 2] * 2)
        node_filter = np.expand_dims(np.expand_dims(node_filter,-1),-1)

        self.node_mask_1 = tf.nn.conv2d(self.elem_mask, node_filter, strides=[1,1,1,1], padding='VALID')
        self.node_mask_2 = tf.nn.conv2d(tf.ones_like(self.elem_mask) - self.elem_mask, node_filter, strides=[1,1,1,1], padding='VALID')
        d_matrix = (self.rho_1 * self.node_mask_1 + self.rho_2 * self.node_mask_2) * (-8. / 3)
        d_matrix *= self.bc_mask
        return d_matrix

    def LU_layers(self, input_tensor):
        from tf_ops_cpp.mask_conv import mask_conv
        mask_tensor = self.mask

        padded_input = boundary_padding(input_tensor)  # for boundary consideration
        padded_mask = boundary_padding(mask_tensor)  # for boundary consideration
        #R_u = tf.nn.conv2d(input=padded_input, filter=self.R_filter, strides=[1, 1, 1, 1], padding='VALID')
        R_u = mask_conv(padded_input, padded_mask, [self.rho_1, self.rho_2])
        R_u_bc = R_u * self.bc_mask # boundary_corrrect
        return R_u_bc


    def apply(self, max_itr=10):
        result = {}
        u0 = np.zeros((1, num_node, num_node, 1), 'float32')  # where u is unknown
        result['u_hist'] = [u0]
        for itr in range(max_itr):
            R_u = self.LU_layers(result['u_hist'][-1])
            u = (self.load - R_u) / self.d_matrix  # jacobi formulation of linear system of equation solver
            result['u_hist'] += [u]

        self.prediction = result['u_hist'][-1]
        return result

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
    u_img = u.reshape(1, num_node,num_node, 1).transpose((0,2,1,3))
    f_img = f.reshape(1, num_node,num_node, 1).transpose((0,2,1,3))
    rho_1, rho_2 = 16., 205.
    return num_node, np.asarray(u_img,'float32'), np.asarray(f_img,'float32'), np.asarray(mask, 'float32'), rho_1, rho_2


if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # build network
    batch_size = 1
    num_node, resp_data, load_data, mask_data, rho_1, rho_2 = load_data_elem()
    if 0:
        # plot out training data
        for i in range(5):
            plt.subplot(2, 5, i + 1)
            plt.imshow(load_data[i, :, :, 0].transpose((1, 0)))
            plt.axis('off')
        for i in range(6):
            plt.subplot(2, 5, 5 + i + 1)
            plt.imshow(resp_data[i, :, :, 0].transpose((1, 0)), cmap='jet', interpolation='bilinear')
            plt.axis('off')

    load_pl = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 1))
    resp_pl = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 1)) # defined on the nodes
    #initial_mask = np.ones_like(mask_data)
    initial_mask =  mask_data
    mask_pl = tf.Variable(initial_value=initial_mask,dtype=tf.float32, name='mask_pl') #defined on the elements
    jacobi = Jacobi_block(num_node, load_pl, mask_pl)
    jacobi.apply(max_itr=100)

    # optimizer
    pred_err = tf.reduce_mean(tf.abs(jacobi.prediction - resp_pl ))
    penalty = tf.reduce_mean( tf.abs(mask_pl) + tf.abs(1-mask_pl) + 0.5-tf.abs(0.5-mask_pl)) # a W shaped penalty
    mask_err = tf.reduce_mean(tf.abs(mask_pl-mask_data))
    loss = pred_err + penalty
    lr = 0.1
    optimizer=tf.train.MomentumOptimizer(lr,0.99)#
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

    loss_hist = []
    pred_err_hist = []
    pred_hist = []
    k1_hist = []
    k2_hist = []
    mask_err_hist = []
    mask_hist = []
    for itr in tqdm(range(10000)):
        feed_dict_train = {load_pl: load_data, resp_pl: resp_data}#, mask_pl: mask_data
        _, loss_value_i, pred_err_i, pred_i, mask_err_i, mask_i, k1_value_i, k2_value_i = sess.run([train_op,
                                                                                            loss,
                                                                                            pred_err,
                                                                                            jacobi.prediction,
                                                                                            mask_err,
                                                                                            jacobi.mask,
                                                                                            jacobi.rho_1,
                                                                                            jacobi.rho_2], feed_dict_train)
        print("iter:{}  pred_err: {} mask_err: {}  k1_value: {}  k2_value: {}".format(itr, np.mean(pred_err_i), np.mean(mask_err_i), k1_value_i, k2_value_i))
        loss_hist += [loss_value_i]
        k1_hist += [k1_value_i]
        k2_hist += [k2_value_i]
        pred_err_hist += [pred_err_i]
        pred_hist += [pred_i]
        mask_err_hist += [mask_err_i]
        mask_hist += [mask_i]

    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(resp_data))
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(pred_hist[0]))
    plt.colorbar()
    print('done')
