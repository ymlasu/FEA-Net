import numpy as np
import tensorflow as tf
from Jacobi_block import Jacobi_block
from VMG import VMG_geometric
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


class FMG():
    def __init__(self,cfg):
        self.alpha_1 = 5
        self.alpha_2 = 50
        self.jacobi = Jacobi_block(cfg)
        self.max_depth = cfg['max_depth']

        self.vmg_stack = {}
        for depth_i in range(1, self.max_depth+1, 1):
            cfg['max_depth'] = depth_i
            self.vmg_stack['depth_{}'.format(depth_i)] = VMG_geometric(cfg, self.jacobi)

    def apply(self, f, u):
        result = {}
        f_level = {}
        u_level = {}
        f_level['1h'] = f
        u_level['1h'] = u
        for depth_i in range(1, self.max_depth, 1):
            f_level['{}h'.format(2**depth_i)] = f = tf.nn.avg_pool(f, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            u_level['{}h'.format(2**depth_i)] = u =tf.nn.avg_pool(u, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # run several VMG of different depth
        u_level_i = u_level['{}h'.format(2**(self.max_depth-1))]
        for depth_i in range(1, self.max_depth+1, 1):
            vmg_level_i = self.vmg_stack['depth_{}'.format(depth_i)]
            f_level_i = f_level['{}h'.format(2**(self.max_depth-depth_i))]
            result_level_i, _, _ = vmg_level_i.apply(f_level_i, u_level_i)
            result['depth_{}'.format(depth_i)] = u_level_i = result_level_i
            if depth_i<self.max_depth:
                upper_level_sol_dim = u_level['{}h'.format(2**(self.max_depth-depth_i-1))].get_shape().as_list()[1:3]
                u_level_i = tf.image.resize_images(u_level_i, size=upper_level_sol_dim,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        result['final'] = result['depth_{}'.format(self.max_depth)]
        return result

if __name__ == '__main__':
    cfg = {
        'batch_size': 16,
        'imsize': 64,
        'physics_problem': 'heat_transfer',  # candidates: 3D plate elasticity, helmholtz, vibro-acoustics
        'max_depth': 4, # depth of V cycle
        'alpha1': 3,  # iteration at high frequency
        'alpha2': 3,  # iteration at low freqeuncy
    }

    f = tf.placeholder(tf.float32, shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize'], 1))
    u = tf.placeholder(tf.float32, shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize'], 1))

    jacobi = Jacobi_block(cfg)
    fmg = FMG(cfg)
    fmg_result = fmg.apply(f, f)

    # optimizer
    u_hat = fmg_result['final']
    loss = tf.reduce_mean(tf.abs(u_hat - u ))
    lr = 0.01
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdadeltaOptimizer(learning_rate)
    grads=optimizer.compute_gradients(loss)
    train_op=optimizer.apply_gradients(grads)
