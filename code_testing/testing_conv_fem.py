import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np


def Jacobi_solver_ax(A1, f1, u):
    # data = sio.loadmat('/home/hope-yao/Downloads/matrix.mat')
    # A1 = data['matrix'][0][0][0]
    # f1 = data['matrix'][0][0][1]
    # u1 = np.linalg.solve(A1, f1)

    D = A1.diagonal()
    D_inv = np.diag(1. / np.asarray(D))
    LU = A1 - np.diag(D)
    # a, b = np.linalg.eig(np.matmul(D_inv, LU))

    # u = np.zeros_like(f1)
    u_hist = [u]
    er_hist = [np.mean(np.abs(u_hist[-1] - u))]
    for i in range(200):
        u = np.matmul(D_inv, (f1 - np.matmul(LU, u)))
        u_hist += [u]
        er_hist += [np.matmul(A1,u_hist[-1]) - f1]

    # plt.plot(er_hist)
    # plt.show()
    return u_hist[-1], er_hist[-1]


def VMG_solver_ax():
    data = sio.loadmat('/home/hope-yao/Downloads/matrix.mat')
    A1 = data['matrix'][0][0][0]
    f1 = data['matrix'][0][0][1]
    max_depth = 2
    f_level = {}
    f_i = f1.reshape(66,68)
    f_level['1h'] = f_i
    for layer_i in range(1, max_depth, 1):
        f_i = np.asarray([np.mean(f_i[2*i:2*(i+1),2*j:2*(j+1)]) for i in range(33) for j in range(34)]).reshape(33,34)
        f_level['{}h'.format(2 ** layer_i)] = f_i.flatten()

    # fine to coarse h, 2h, 4h, 8h, ...
    f_cur = f1
    u_level = []
    for layer_i in range(1, max_depth+1, 1):
        u_cur = f_cur
        u_h, r_h = Jacobi_solver_ax(A1, f_cur.flatten(), u_cur.flatten())
        r_h = r_h.reshape(66, 68)
        u_level += [u_h]
        # downsample residual to next level input
        r_h_down = np.asarray([np.mean(r_h[2*i:2*(i+1),2*j:2*(j+1)]) for i in range(33) for j in range(34)]).reshape(33,34)
        f_cur = r_h_down

    # bottom level, lowest frequency part
    e_bottom = Jacobi_solver_ax(A1, f_cur.flatten(), f_cur.flatten())

    u_pred = []
    layer_i = 1
    # coarse to fine: ..., 8h, 4h, 2h, h
    cur_level_sol = e_bottom
    u_pred += [cur_level_sol]
    while layer_i < max_depth:  # 4h, 2h, h
        upper_level_sol = u_level[-layer_i]
        import scipy.ndimage as ndimage
        upsampled_cur_level_sol = ndimage.zoom(e_bottom.reshape(33, 34), 2, order=0)
        cur_level_sol = upper_level_sol + upsampled_cur_level_sol
        cur_level_sol_correct = Jacobi_solver_ax(A1, cur_level_sol.flatten(), cur_level_sol.flatten())
        u_level += [cur_level_sol_correct + u_level[0]]
        layer_i += 1
    u_pred['final'] = u_pred['1h']

    return u_pred


def ax_vs_wx():
    data = sio.loadmat('./data/heat_transfer/Heat_Transfer.mat')
    ftest = np.zeros((66,68),dtype='float32')
    ftest[1:65,2:66] = data['u']


    # axpy
    data = sio.loadmat('/home/hope-yao/Downloads/matrix.mat')
    A1 = data['matrix'][0][0][0]
    f1 = data['matrix'][0][0][1]
    u1 = np.linalg.solve(A1, f1)
    b=np.matmul(A1,ftest.reshape(66*68,1))#np.ones_like(f1))
    bb = b.reshape(66, 68)
    plt.figure()
    plt.imshow(bb)
    plt.grid('off')
    plt.colorbar()

    # conv
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    # imsize = 64
    A_weights = {}
    A_weights['k'] = 16.
    w_filter = np.asarray([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]],'float32') * A_weights['k'] / 3.
    w_filter = tf.constant(w_filter.reshape((3,3,1,1)))
    u = np.ones((1,68,68,1),'float32')
    u[0,1:-1,:,0] = ftest.reshape(66,68)
    u[0,0,:,0] = 0
    u[0,-1,:,0] = 0
    u = tf.constant(u)
    output = tf.nn.conv2d(input=u, filter=w_filter, strides=[1,1,1,1], padding='SAME')
    img = sess.run(output)
    plt.figure()
    plt.imshow(img[0,1:-1, :,  0])
    plt.colorbar()
    plt.grid('off')
    plt.show()
    print('error: {}'.format(np.mean(np.abs(bb-img[0,1:-1, :,  0]))))


def Jacobi_solver_wx():

    A_weights = {}
    A_weights['k'] = 16.
    lu_filter = np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]])
    A_weights['LU_filter'] = np.reshape(lu_filter, (3, 3, 1, 1)) * A_weights['k']/ 3.
    A_weights['D_matrix'] = np.tile(np.reshape(-8. * A_weights['k']/3, (1, 1, 1, 1)), (1, 66, 68, 1))
    A_weights['D_matrix'] = tf.constant(A_weights['D_matrix'],dtype='float32')

    data = sio.loadmat('/home/hope-yao/Downloads/matrix.mat')
    f1 = data['matrix'][0][0][1]
    A1 = data['matrix'][0][0][0]
    u1 = np.linalg.solve(A1, f1)
    u_gt = u1.reshape(1,66,68,1)
    f1 = f1.reshape(1,66,68,1)

    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    result = {}
    u_input = np.zeros((1, 68, 68, 1), 'float32')  # where u is unknown
    for itr in range(1000):
        padded_input = tf.pad(u_input, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
        LU_u = tf.nn.conv2d(input=padded_input, filter=A_weights['LU_filter'], strides=[1, 1, 1, 1], padding='VALID')
        u = (f1 - LU_u[:, 1:-1, :]) / A_weights['D_matrix']
        u_input = tf.pad(u, tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]]), "CONSTANT")
    result['final'] = sess.run(u)[0,:,:,0]

    plt.figure()
    plt.imshow(u_gt[0,:,:,0],cmap='hot')
    plt.grid('off')
    plt.colorbar()
    plt.figure()
    plt.imshow(result['final'],cmap='hot')
    plt.grid('off')
    plt.colorbar()
    plt.show()
    return result


# Jacobi_solver_ax()
# ax_vs_wx()
# Jacobi_solver_wx()
# VMG_solver_ax()
ax_vs_wx_2phase()
