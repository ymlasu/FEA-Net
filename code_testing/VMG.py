import numpy as np
import tensorflow as tf
from Jacobi_block import Jacobi_block
import scipy.io as sio


def smoothing_restriction(img):
    '''
    http://www.maths.lth.se/na/courses/FMNN15/media/material/Chapter9.09c.pdf
    '''
    lp_filter = np.asarray([[1 / 16., 1 / 8., 1 / 16.], [1 / 8., 1 / 4., 1 / 8.], [1 / 16., 1 / 8., 1 / 16.]])
    lp_filter = tf.constant(lp_filter.reshape(3, 3, 1, 1), dtype=tf.float32)
    # use some image to double check on the filtering effect here
    smoothed_img = tf.nn.conv2d(input=img, filter=lp_filter, strides=[1, 1, 1, 1], padding='SAME')
    return smoothed_img



class VMG_algebraic():
    def __init__(self, cfg, jacobi):
        self.alpha_1 = 5
        self.alpha_2 = 50
        self.jacobi = jacobi
        self.cfg = cfg
        self.max_depth = self.cfg['max_depth']

    def Ax_net(self, input_tensor, A_weights):
        LU_filter, LU_bias, D_mat = A_weights['LU_filter'], A_weights['LU_bias'], A_weights['D_matrix']
        LU_u = self.jacobi.LU_layers(input_tensor, LU_filter, LU_bias)
        return D_mat * input_tensor + LU_u

    def apply_MG_block(self,f,u):
        result = {}
        u_h_hist = self.jacobi.apply(f, u, max_itr=self.alpha_1)
        result['u_h'] = u_h_hist['u_hist'][-1]
        res = self.Ax_net(result['u_h'], self.jacobi.A_weights)
        result['res_pool'] = tf.nn.avg_pool(res, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        result['res_smooth'] = smoothing_restriction(res)
        return result

    def apply(self, f, u):
        if self.max_depth == 0:
            u_pred = {}
            sol = self.jacobi.apply(f, u, max_itr=self.alpha_2)
            u_pred['final'] = sol['u_hist'][-1]
            return u_pred

        u_h = {}
        r_h = {}
        # fine to coarse: 0,1,2,3
        for layer_i in range(self.max_depth):
            mg_result = self.apply_MG_block(f, u)
            f = mg_result['res_smooth'] # residual as input rhs
            u = tf.constant(np.zeros((self.cfg['batch_size'], self.cfg['imsize'], self.cfg['imsize'], 1), dtype='float32')) # all zero initial guess
            u_h['layer_{}'.format(layer_i)] = mg_result['u_h']
            r_h['layer_{}'.format(layer_i)] = mg_result['res_smooth']

        # bottom level, lowest frequency part
        e_bottom = self.jacobi.apply(mg_result['res_smooth'], u, max_itr=self.alpha_2)

        # coarse to fine: 3,2,1,0
        u_pred = {}
        for layer_i in range(self.max_depth - 1, -1, -1):
            if layer_i == self.max_depth - 1:
                u_pred['layer_{}'.format(layer_i)] = e_bottom['u_hist'][-1] + u_h['layer_{}'.format(layer_i)]
            else:
                u_pred['layer_{}'.format(layer_i)] = u_pred['layer_{}'.format(layer_i + 1)] + u_h[
                    'layer_{}'.format(layer_i)]
        u_pred['final'] = u_pred['layer_{}'.format(layer_i)]
        return u_pred

class VMG_geometric():
    def __init__(self, cfg, jacobi):
        self.alpha_1 = cfg['alpha1']
        self.alpha_2 = cfg['alpha2']
        self.jacobi = jacobi
        self.cfg = cfg
        self.max_depth = self.cfg['max_depth']
        self.imsize = cfg['imsize']

    def Ax_net(self, input_tensor, jacobi):
        D_mat = -8./3. * self.jacobi.k
        u_input = tf.pad(input_tensor, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]), "CONSTANT")  # add zeros to Dirc BC
        LU_u = jacobi.LU_layers(input_tensor)
        return D_mat * input_tensor + LU_u

    def apply_MG_block(self, f, u, max_itr):
        result = {}
        u_hist = self.jacobi.apply(f, u, max_itr=max_itr)
        result['u'] = u_hist['final']
        ax = self.Ax_net(result['u'], self.jacobi)
        result['res'] = f[:,:,1:-1,:]-ax
        return result

    def apply(self, f, u):
        f_level = {}
        f_i = f
        f_level['1h'] = f
        for layer_i in range(1, self.max_depth, 1):
            # times 4 because surface force needs integration to become nodal force!
            f_i_center = 4.*tf.nn.avg_pool(f_i[:,:,1:-1:], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!
            f_i = tf.pad(f_i_center, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]), "CONSTANT") # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!
            f_level['{}h'.format(2 ** layer_i)] = f_i

        u_h = {}
        r_h = {}
        cur_u = u # inital guess
        cur_f = f
        # fine to coarse h, 2h, 4h, 8h, ...
        for layer_i in range(1,self.max_depth,1):
            mg_result = self.apply_MG_block(cur_f, cur_u, self.alpha_1)
            u_h['{}h'.format(2**(layer_i-1))] = mg_result['u']
            r_h['{}h'.format(2**(layer_i-1))] = mg_result['res']
            # downsample residual to next level input
            cur_f = 4.*tf.nn.avg_pool(mg_result['res'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            cur_f = tf.pad(cur_f, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]), "CONSTANT") # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!
            cur_u = tf.zeros_like(cur_f) # inital 0 for residual

        # bottom level, lowest frequency part
        e_bottom = self.jacobi.apply(cur_f, cur_u, max_itr=self.alpha_2)
        u_h['{}h'.format(2 ** (self.max_depth - 1))] = e_bottom['final']

        u_correct = {}
        layer_i = 1
        # coarse to fine: ..., 8h, 4h, 2h, h
        u_correct['{}h'.format(2**(self.max_depth-1))] = cur_level_sol = u_h['{}h'.format(2**(self.max_depth-1))]
        while layer_i<self.max_depth: # 4h, 2h, h
            upper_level_sol = u_h['{}h'.format(2 ** (self.max_depth-layer_i-1))]
            upper_level_sol_dim = upper_level_sol.get_shape().as_list()[1:3]
            upsampled_cur_level_sol = 0.25* tf.image.resize_images(cur_level_sol, size=upper_level_sol_dim,method=tf.image.ResizeMethod.BILINEAR)#NEAREST_NEIGHBOR
            # divided by 4 because surface force needs integration to become nodal force!
            cur_level_sol = upper_level_sol + upsampled_cur_level_sol
            cur_level_sol = tf.pad(cur_level_sol, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]), "CONSTANT") # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!
            cur_level_f = f_level['{}h'.format(2 ** (self.max_depth - layer_i - 1))]
            cur_level_sol_correct = self.jacobi.apply(cur_level_f, cur_level_sol, self.alpha_1)
            u_correct['{}h'.format(2**(self.max_depth-layer_i-1))] = cur_level_sol_correct['final']
            layer_i += 1
        if self.max_depth>1:
            u_pred = u_correct['1h']
        else:
            u_pred = u_h['1h']

        return u_pred, u_h, u_correct

if __name__ == '__main__':
    from tqdm import tqdm

    cfg = {
        'batch_size': 1,
        'imsize': 64,
        'physics_problem': 'heat_transfer',  # candidates: 3D plate elasticity, helmholtz, vibro-acoustics
        'max_depth': 5, # depth of V cycle, degrade to Jacobi if set to 0
        'alpha1': 3, # iteration at high frequency
        'alpha2': 3, # iteration at low freqeuncy
        'max_v_cycle': 20,
    }

    f = tf.placeholder(tf.float32, shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize']+2, 1))
    u = tf.placeholder(tf.float32, shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize']+2, 1))
    jacobi = Jacobi_block(cfg)
    vmg = VMG_geometric(cfg, jacobi)
    # jacobi_result = jacobi.apply(f, u_initial, max_itr=cfg['alpha2'])

    vmg_result_itr = {}
    vmg_u_h_itr = {}
    vmg_u_correct_itr = {}
    u_initial = tf.zeros_like(f)
    vmg_result, vmg_u_h_itr['v_0'], vmg_u_correct_itr['v_0'] = vmg.apply(f, u_initial) # f is inital guess for u
    vmg_result_itr['v_0'] = tf.pad(vmg_result, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]), "CONSTANT") # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!
    for v_idx in range(1, cfg['max_v_cycle'], 1):
        vmg_result, vmg_u_h, vmg_u_correct = vmg.apply(f, vmg_result_itr['v_{}'.format(v_idx-1)]) # previous V cycle is inital guess for u
        vmg_result_itr['v_{}'.format(v_idx)] = tf.pad(vmg_result, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]), "CONSTANT") # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!
        vmg_u_h_itr['v_{}'.format(v_idx)] = vmg_u_h
        vmg_u_correct_itr['v_{}'.format(v_idx)] =  vmg_u_correct
    vmg_result = vmg_result_itr['v_{}'.format(cfg['max_v_cycle']-1)]

    # optimizer
    loss = tf.reduce_mean(tf.abs(vmg_result - u ))
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

    u1 = sio.loadmat('/home/hope-yao/Downloads/Solution_200_6466.mat')['U1']
    f1 = sio.loadmat('/home/hope-yao/Downloads/Input_200_q.mat')['F1']
    idx = 41
    loss_hist = []
    for ii in range(1, 20, 1):
        loss_hist += [
            sess.run(tf.reduce_mean(tf.abs(vmg_result_itr['v_{}'.format(ii)][:, :, 1:-1, :] - u[:, :, 1:-1, :])),
                     {f: f1[idx, :, :].reshape(1, 64, 66, 1), u: u1[idx, :, :].reshape(1, 64, 66, 1)})]

    data = sio.loadmat('/home/hope-yao/Downloads/matrix.mat')
    f1 = data['matrix'][0][0][1]
    A1 = data['matrix'][0][0][0]
    u1 = np.linalg.solve(A1, f1)
    u_gt = u1.reshape(1, 66, 68, 1)[:,1:-1, 2:-2,:]
    f1 = f1.reshape(1, 66, 68, 1)[:,1:-1, 2:-2,:]

    # u1 = sio.loadmat('/home/hope-yao/Downloads/Solution_6664.mat')['U1'][:, 1:-1, :]
    # f1 = sio.loadmat('/home/hope-yao/Downloads/Input_q.mat')['F1'][:, 1:-1, :]
    #
    # u_gt = u1.reshape(10, cfg['imsize'], cfg['imsize'], 1)
    # f1 = f1.reshape(10, cfg['imsize'], cfg['imsize'], 1)

    u_input = np.tile(u_gt, (16, 1, 1, 1))
    f_input = np.tile(f1, (16, 1, 1, 1))
    feed_dict_train = {f: f_input, u: u_input}
    loss_value, u_value = sess.run([loss, vmg_result], feed_dict_train)

    import matplotlib.pyplot as plt

    plt.imshow(u_value[0, :, :, 0], cmap='hot')
    plt.grid('off')
    plt.colorbar()
    plt.show()
    print('done')