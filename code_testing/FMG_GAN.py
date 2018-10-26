import numpy as np
import tensorflow as tf
from Jacobi_block import Jacobi_block
from VMG import VMG
from FMG import FMG

class FMG_GAN():
    def __init__(self,cfg):

        self.alpha_1 = 5
        self.alpha_2 = 50
        self.jacobi = Jacobi_block(cfg)
        self.max_depth = cfg['max_depth']
        self.fmg = FMG(cfg)

    def build_model(self):
        self.fmg_result = self.fmg.apply(f, u)

        self.loss = {}
        for depth_i in range(cfg['max_depth']):
            u_hat = self.fmg_result['depth_{}'.format(depth_i)]
            self.loss['depth_{}'.format(depth_i)] = tf.reduce_mean(tf.abs(u_hat - u))

    def train_model(self):
        '''training FMG_GAN in a progressive way'''



    def train_model(self):
        # optimizer
        lr = 0.01
        learning_rate = tf.Variable(lr)  # learning rate for optimizer
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        grads = optimizer.compute_gradients(self.loss)
        train_op = optimizer.apply_gradients(grads)


if __name__ == '__main__':
    cfg = {
        'batch_size': 16,
        'imsize': 64,
        'physics_problem': 'heat_transfer',  # candidates: 3D plate elasticity, helmholtz, vibro-acoustics
        'max_depth': 4,  # depth of V cycle
    }

    f = tf.placeholder(tf.float32, shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize'], 1))
    u = tf.placeholder(tf.float32, shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize'], 1))

    fmg_gan = FMG_GAN(cfg)
    fmg_gan.build_model()


