import tensorflow as tf
import numpy as np

conductivity_1 = 16
conductivity_2 = 56
heat_filter_1 = conductivity_1 / 3. * np.asarray([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]]).reshape(3,3,1,1)
heat_filter_2 = conductivity_2 / 3. * np.asarray([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]]).reshape(3,3,1,1)

################
import scipy.io as sio
u1 = sio.loadmat('/home/hope-yao/Downloads/solution_6666.mat')['U1'][0][1:-1,1:-1]
f1 = sio.loadmat('/home/hope-yao/Downloads/q_6666.mat')['F1'][0][1:-1,1:-1]

f_input = tf.placeholder(tf.float32, shape=(1, 64, 64, 1))
u_opt = tf.Variable(tf.zeros((1,64,64,1)))
padded_input = tf.pad(u_opt, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
f_pred_opt = tf.nn.conv2d(input=padded_input, filter=heat_filter_1, strides=[1, 1, 1, 1], padding='VALID')
loss = tf.reduce_mean(tf.abs(f_pred_opt - f_input))
opt = tf.train.AdamOptimizer(0.1)
grads_g = opt.compute_gradients(loss)
apply_gradient_op = opt.apply_gradients(grads_g)
init_op = tf.initialize_variables(tf.all_variables())

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

loss_val_hist =[]
x_output_val_hist = []
for t in np.linspace(0., 1., 10):
    x_output_val, loss_val, _ = sess.run([u_opt, loss, apply_gradient_op], {f_input:f1.reshape(1,64,64,1)})
    loss_val_hist += [loss_val]
    x_output_val_hist += [x_output_val]

import matplotlib.pyplot as plt
plt.figure()
plt.plot(loss_val_hist)
plt.figure()
plt.imshow(x_output_val_hist[-1][0,:,:,0],cmap='hot')
plt.colorbar()
plt.show()

print('done')

import matplotlib.pyplot as plt
plt.imshow(f1,cmap='hot')
plt.colorbar()
plt.show()
