import numpy as np
import scipy.io as sio
from scipy import signal

num_node = 65
omega = 2./3

def reference_jacobi_solver(A, f):

    D = np.diag(np.diag(A))
    inv_D = np.expand_dims(1. / np.diag(A), 1)
    LU = A-D
    u = np.copy(f)
    u_hist = [u]
    u_img_hist = [u.reshape(num_node,num_node,2)]
    R_u_img_hist = []
    n_iter = 100000
    for itr_i in range(n_iter):
        R_u = np.matmul(LU, u_hist[-1])
        u_new = omega * inv_D * (f-R_u) + (1-omega)*u_hist[-1]

        R_u_x = np.asarray([R_u[2 * i] for i in range(num_node ** 2)]).reshape(num_node, num_node).transpose((1, 0))
        R_u_y = np.asarray([R_u[2 * i + 1] for i in range(num_node ** 2)]).reshape(num_node, num_node).transpose((1, 0))
        R_u_img = np.stack([R_u_x, R_u_y], -1)
        u_new_x = np.asarray([u_new[2 * i] for i in range(num_node ** 2)]).reshape(num_node, num_node).transpose((1, 0))
        u_new_y = np.asarray([u_new[2 * i + 1] for i in range(num_node ** 2)]).reshape(num_node, num_node).transpose((1, 0))
        u_new_img = np.stack([u_new_x, u_new_y], -1)
        # import matplotlib.pyplot as plt
        # plt.imshow(LU_u_x_hist)
        # plt.show()

        u_hist += [u_new]
        u_img_hist += [u_new_img]
        R_u_img_hist += [R_u_img]

    res = {
        'u_hist' : np.asarray(u_img_hist),
        'R_u_hist' : np.asarray(R_u_img_hist)
    }
    return res

def load_data_elem():
    '''loading data obtained from FEA simulation'''
    # linear elasticity, all steel, Yfix
    data = sio.loadmat('/home/hope-yao/Documents/MG_net/elasticity/data/block_case1.mat')
    ux = data['d_x'].reshape(num_node,num_node).transpose((1,0)) #* 1e10
    uy = data['d_y'].reshape(num_node,num_node).transpose((1,0)) #* 1e10
    u_img = np.concatenate([np.expand_dims(ux, 2), np.expand_dims(uy, 2)], 2)
    fx = data['f_x'].reshape(num_node,num_node).transpose((1,0))
    fy = data['f_y'].reshape(num_node,num_node).transpose((1,0))
    f_img = -1. * np.concatenate([np.expand_dims(fx, 2), np.expand_dims(fy, 2)], 2)

    A = data['K']
    # f = np.asarray(zip(data['f_x'], data['f_y'])).flatten()
    f = np.asarray([var for tup in zip(data['f_x'], data['f_y']) for var in tup])

    if 0:
        # check FEA data is correct
        u = np.asarray([var for tup in zip(data['d_x'], data['d_y']) for var in tup])
        f_pred = np.matmul(A, u)
        import matplotlib.pyplot as plt
        f_pred_img = np.asarray([f_pred[2 * i] for i in range(num_node ** 2)]).reshape(num_node, num_node).transpose((1, 0))
        plt.imshow(f_pred_img)
        plt.show()
    # get reference jacobi data
    #ref_res = reference_jacobi_solver(A, f)

    coef = {
        'E': 1.941e11,  # 200e9,
        'mu': 0.2512,
    }

    coef['wxx'], coef['wxy'], coef['wyx'], coef['wyy'] = get_w_matrix(coef)
    coef['d_matrix'] = np_get_D_matrix_elast(coef, mode='symm')

    return u_img, f_img, coef

def get_w_matrix(coef_dict):
    E, mu = coef_dict['E'], coef_dict['mu']
    cost_coef = E/16./(1-mu**2)
    wxx = cost_coef * np.asarray([
        [-4*(1 - mu / 3.),        16 * mu / 3.,             -4*(1 - mu / 3.)],
        [-8*(1 + mu / 3.),       32. * (1 - mu / 3.),       -8*(1 + mu / 3.)],
        [-4*(1 - mu / 3.),        16 * mu / 3.,             -4*(1 - mu / 3.)],
    ])

    wxy = wyx = cost_coef * np.asarray([
        [-2 * (mu + 1),        0,             2 * (mu + 1)],
        [0,                        0,                  0],
        [2 * (mu + 1),        0,             -2 * (mu + 1)],
    ])

    wyy = cost_coef * np.asarray([
        [-4 * (1 - mu / 3.),         -8 * (1 + mu / 3.),        -4 * (1 - mu / 3.)],
        [16 * mu / 3.,               32. * (1 - mu / 3.),        16 * mu / 3.],
        [-4 * (1 - mu / 3.),         -8 * (1 + mu / 3.),        -4 * (1 - mu / 3.)],
    ])
    return wxx, wxy, wyx, wyy

def np_get_D_matrix_elast(coef_dict, mode='symm'):
    # convolution with symmetric padding at boundary
    d_matrix_xx_val, d_matrix_yy_val = coef_dict['wxx'][1,1], coef_dict['wyy'][1,1]
    d_matrix_xx = d_matrix_xx_val*np.ones((num_node,num_node))
    d_matrix_yy = d_matrix_yy_val*np.ones((num_node,num_node))
    d_matrix = np.stack([d_matrix_xx, d_matrix_yy], -1)
    d_matrix[0,:] /=2
    d_matrix[-1,:] /=2
    d_matrix[:,0] /=2
    d_matrix[:,-1] /=2

    return d_matrix


def boundary_padding(x):
    ''' special symmetric boundary padding '''
    x = np.expand_dims(x, 0)
    left = x[:, :, 1:2, :]
    right = x[:, :, -2:-1, :]
    upper = np.concatenate([x[:, 1:2, 1:2, :], x[:, 1:2, :, :], x[:, 1:2, -2:-1, :]], 2)
    down = np.concatenate([x[:, -2:-1, 1:2, :], x[:, -2:-1, :, :], x[:, -2:-1, -2:-1, :]], 2)
    padded_x = np.concatenate([left, x, right], 2)
    padded_x = np.concatenate([upper, padded_x, down], 1)
    padded_x = padded_x[0,:,:,:]
    return padded_x

def boundary_correct(x):
    x[0,:] /=2
    x[-1,:] /=2
    x[:,0] /=2
    x[:,-1] /=2

    x = np.pad(x[1:-1,1:-1], ((1, 1), (1, 1), (0, 0)), "constant") # for boundary consideration
    return x


def np_conv_elast(node_resp, coef):

    wxx, wxy, wyx, wyy = coef['wxx'], coef['wxy'], coef['wyx'], coef['wyy']
    wxx[1,1] = 0
    wyy[1,1] = 0
    node_resp_x, node_resp_y = node_resp[:,:,0], node_resp[:,:,1]

    res = {}
    res['Ru_part1'] = Ru_part1  = signal.correlate2d(node_resp_x, wxx, mode='valid') # perform convolution
    res['Ru_part2'] = Ru_part2 = signal.correlate2d(node_resp_y, wxy, mode='valid') # perform convolution
    res['Ru_part3'] = Ru_part3 = signal.correlate2d(node_resp_y, wyy, mode='valid') # perform convolution
    res['Ru_part4'] = Ru_part4 = signal.correlate2d(node_resp_x, wyx, mode='valid') # perform convolution

    Ru_x = Ru_part1 + Ru_part2
    Ru_y = Ru_part3 + Ru_part4

    res['Ru'] = np.stack([Ru_x, Ru_y], -1)


    return res


def apply(u_input, f, coef):
    '''jacobi iteration'''
    padded_input = boundary_padding(u_input) # for boundary consideration
    res = np_conv_elast(padded_input, coef)# perform convolution
    Ru_bc = boundary_correct(res['Ru'])
    u = omega * (f - Ru_bc) / coef['d_matrix'] + (1-omega)*u_input# jacobi formulation of linear system of equation solver
    return u

def visualize(loss_hist, resp_pred, resp_gt):
    import matplotlib.pyplot as plt

    BIGGER_SIZE = 16
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure()
    plt.semilogy(loss_hist, 'b-', label='convergence')
    plt.semilogy([10, len(loss_hist)], [10**-1, len(loss_hist) ** -1], 'k--', label='$O(n^{-1})$')
    plt.semilogy([10, len(loss_hist)], [10**-2, len(loss_hist) ** -2], 'k--', label='$O(n^{-2})$')
    plt.legend()
    plt.xlabel('network depth')
    plt.ylabel('prediction error')

    BIGGER_SIZE = 12
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(resp_pred[:, :, 0], cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.subplot(2, 3, 4)
    plt.imshow(resp_pred[:, :, 1], cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.subplot(2, 3, 2)
    plt.imshow(resp_gt[:, :, 0], cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.subplot(2, 3, 5)
    plt.imshow(resp_gt[:, :, 1], cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.subplot(2, 3, 3)
    plt.imshow(resp_pred[:, :, 0] - resp_gt[:, :, 0], cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.subplot(2, 3, 6)
    plt.imshow(resp_pred[:, :, 1] - resp_gt[:, :, 1], cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.show()

def main():
    resp_gt, load_gt, coef = load_data_elem()

    u_hist = [np.zeros((num_node,num_node,2))]
    loss_hist = []
    error_hist = []
    for i in range(3000):
        u_new = apply(u_hist[-1], load_gt, coef)
        u_hist += [u_new]
        loss_i = np.linalg.norm(u_hist[i] - resp_gt)/np.linalg.norm(resp_gt)

        padded_input = boundary_padding(u_hist[i])  # for boundary consideration
        res = np_conv_elast(padded_input, coef)  # perform convolution
        Ru_bc = boundary_correct(res['Ru'])
        load_pred =  Ru_bc + coef['d_matrix'] * u_hist[i]  # jacobi formulation of linear system of equation solver

        error_i = np.linalg.norm(load_pred - load_gt)/np.linalg.norm(load_gt)
        loss_hist += [loss_i]
        error_hist += [error_i]
        print('n_itr: {}, loss: {}, error: {}'.format(i,loss_i,error_i))

    visualize(loss_hist, u_hist[-1], resp_gt)
    return u_hist

if __name__ == '__main__':
    u_hist = main()
