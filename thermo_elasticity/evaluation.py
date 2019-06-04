import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from therm_elast_train_2filters import *
from data_loader import *

class Evaluator(object):
    def __init__(self, model, data):
        self.model = model

        self.data = data
        self.init_w = np.zeros((3,3,1,1))

        self.loss_value = None
        self.grads_value = None

        self.loss_tf = self.model.get_loss()
        self.hessian_tf = self.model.get_hessian()
        self.grad_tf = self.model.get_grad()
        self.initial_graph()

    def initial_graph(self):
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

    def loss(self, w):
        self.feed_dict = {self.model.load_pl: data['train_load'],
                          self.model.resp_pl: data['train_resp'],
                          self.model.trainable_var_pl: w}
        self.loss_value = self.sess.run(self.loss_tf, self.feed_dict).astype('float64')
        return self.loss_value

    def grads(self, w):
        self.feed_dict = {self.model.load_pl: data['train_load'],
                          self.model.resp_pl: data['train_resp'],
                          self.model.trainable_var_pl: w}
        self.grads_value = self.sess.run(self.grad_tf, self.feed_dict)[0].flatten().astype('float64')
        return self.grads_value

    def hessian(self, w):
        self.feed_dict = {self.model.load_pl: data['train_load'],
                          self.model.resp_pl: data['train_resp'],
                          self.model.trainable_var_pl: w}
        self.hessian_value = self.sess.run(self.hessian_tf, self.feed_dict)[0].astype('float64')
        return self.hessian_value

    def pred(self,w):
        feed_dict = {self.model.load_pl: data['train_load'],
                      self.model.resp_pl: data['train_resp'],
                      self.model.trainable_var_pl: w.astype('float32')}
        pred_value = self.sess.run(self.model.load_pred, feed_dict)
        return pred_value

    def run_BFGS(self):
        from scipy.optimize import fmin_l_bfgs_b
        x, min_val, info = fmin_l_bfgs_b(self.loss, self.init_w.flatten(),
                                         fprime=self.grads, maxiter=200, maxfun=200,
                                         disp= True)
        print('    loss: {}'.format(min_val))
        pass

    def run_newton(self):
        from scipy.optimize import minimize
        self.result = minimize(self.loss, self.model.trainable_var_np, method='Newton-CG',
                          jac=self.grads, hess=self.hessian,
                          options={'xtol': 1e-8, 'disp': True})
        return self.result

    def visualize(self, w):
        pred_value = self.pred(w)
        plt.figure(figsize=(6, 6))
        idx = 0  # which data to visualize
        for i in range(3):
            plt.subplot(4, 3, i + 1)
            plt.imshow(self.data['test_resp'][idx, 1:-1, 1:-1, i])
            plt.colorbar()
            plt.subplot(4, 3, 3 + i + 1)
            plt.imshow(self.data['test_load'][idx, 1:-1, 1:-1, i])
            plt.colorbar()
            plt.subplot(4, 3, 6 + i + 1)
            plt.imshow(pred_value[idx, 1:-1, 1:-1, i])
            plt.colorbar()
            plt.subplot(4, 3, 9 + i + 1)
            plt.imshow(self.data['test_load'][idx, 1:-1, 1:-1, i] - pred_value[idx, 1:-1, 1:-1, i])
            plt.colorbar()
        plt.show()

    def run_forward(self, filter, data):
        max_itr = 100
        omega = 1.
        load = data['test_load']
        resp = data['test_resp']
        loss_hist = []
        pred_hist = []
        itr_hist = []

        self.model.init_solve(load, omega)
        pred_i = resp#np.zeros_like(load)
        for itr in tqdm(range(0, max_itr, 10)):
            feed_dict = {self.model.u_in: pred_i, self.model.trainable_var_pl:filter}
            pred_i = self.sess.run(self.model.u_out, feed_dict)
            pred_err_i = np.mean(np.abs(pred_i - resp))
            print("iter:{}  pred_err: {}".format(itr, np.mean(pred_err_i)))
            pred_hist += [pred_i]
            loss_hist += [np.mean(pred_err_i)]
            itr_hist += [itr]


if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    cfg = {'lr': 0.001,
           'epoch': 1,
           }

    # load data
    data = load_data()#snr=100

    # build the network
    model = FEA_Net_h(data,cfg)

    # train the network
    evaluator = Evaluator(model, data)
    result = evaluator.run_newton()

    # visualize training result
    evaluator.visualize(result.x)
    for i in range(2):
        mat = result.x[9*i:9*(i+1)]
        print(mat.reshape(3,3))
        print(np.sum(mat))

    # test the model
    evaluator.run_forward(result.x, data)
    # evaluator.run_forward(model.trainable_var_np, data)