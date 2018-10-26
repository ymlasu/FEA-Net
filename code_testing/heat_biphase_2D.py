import numpy as np
import scipy.io as sio


def load_data_elem(case):
    if case in[-2, -1, 0, 1]:
        # heat transfer
        if case == -2:
            # all steel
            # u = np.zeros((66, 66), 'float32')
            u = sio.loadmat('/home/hope-yao/Downloads/steel_U.mat')['U1'][0][1:-1, 1:-1]
            f = sio.loadmat('/home/hope-yao/Downloads/steel_q.mat')['F1'][0][1:-1,1:-1]
            mask_1 = np.asarray([[1., ] * 43 + [1.] * 20] * 63, dtype='float32')
            conductivity_1 = np.float32(16.)
            conductivity_2 = np.float32(16.)
        elif case == -1:
            # toy case
            mask_1 = np.asarray([[1, 1, 1, 1, 1, 1, 1, 1, 1, ],
                                 [1, 1, 1, 1, 0, 0, 0, 0, 0, ],
                                 [1, 1, 1, 1, 0, 0, 0, 0, 0, ],
                                 [1, 1, 1, 1, 0, 0, 0, 0, 0, ],
                                 [1, 1, 1, 1, 0, 0, 0, 0, 0, ],
                                 [1, 1, 1, 1, 1, 0, 0, 0, 0, ],
                                 [1, 1, 1, 1, 1, 0, 0, 0, 0, ],
                                 [1, 1, 1, 1, 1, 0, 0, 0, 0, ],
                                 [1, 1, 1, 1, 1, 0, 0, 0, 0, ]])
            mask_1 = np.asarray(mask_1, dtype='float32')
            f = u = np.ones((1, 10, 10, 1), dtype='float32')
            conductivity_1 = np.float32(10.)
            conductivity_2 = np.float32(100.)
        elif case == 0:
            mask_1 = sio.loadmat('./data/heat_transfer_2phase/mask.mat')['ind2']
            f = sio.loadmat('./data/heat_transfer_2phase/input_heatFlux.mat')['aa']
            u = sio.loadmat('./data/heat_transfer_2phase/steel_Aluminum_solution.mat')['u1']
            conductivity_1 = np.float32(16.)
            conductivity_2 = np.float32(205.)
        elif case == 1:
            mask_1 = sio.loadmat('./data/heat_transfer_2phase/mask.mat')['ind2']
            f = sio.loadmat('./data/heat_transfer_2phase/input_heatFlux.mat')['aa']
            u = sio.loadmat('./data/heat_transfer_2phase/steel_Air_solution.mat')['u1']
            conductivity_1 = np.float32(16.)
            conductivity_2 = np.float32(0.0262)
        coef_dict={
            'conductivity_1': conductivity_1,
            'conductivity_2': conductivity_2
        }
    return u, f, mask_1, coef_dict

def get_D_matrix(elem_mask, coef_dict):
    '''

    :param elem_mask:
    :param conductivity_1:
    :param conductivity_2:
    :return:
    '''
    conductivity_1, conductivity_2 = coef_dict['conductivity_1'], coef_dict['conductivity_2']
    # convolution with symmetric padding at boundary
    from scipy import signal
    elem_mask = np.squeeze(elem_mask)
    node_filter = np.asarray([[1 / 4.] * 2] * 2)
    # first material phase
    node_mask_1 = signal.correlate2d(elem_mask, node_filter, boundary='symm')
    # second material phase
    node_mask_2 = signal.correlate2d(np.ones_like(elem_mask) - elem_mask, node_filter, boundary='symm')
    d_matrix = node_mask_1 * conductivity_1*(-8./3.) + node_mask_2 * conductivity_2*(-8./3.)
    return np.expand_dims(np.expand_dims(d_matrix,0),3)

def masked_conv(elem_mask_orig, node_resp, coef):
    diag_coef_1, side_coef_1 = coef['diag_coef_1'], coef['diag_coef_1']
    diag_coef_2, side_coef_2 = coef['diag_coef_2'], coef['diag_coef_2']
    x  = np.pad(node_resp, ((0, 0), (1, 1), (1, 1), (0, 0)), "symmetric")
    elem_mask = np.pad(elem_mask_orig, ((0, 0), (1, 1), (1, 1), (0, 0)), "symmetric")
    y_diag_1 = np.zeros_like(node_resp)
    y_side_1 = np.zeros_like(node_resp)
    y_diag_2 = np.zeros_like(node_resp)
    y_side_2 = np.zeros_like(node_resp)
    for i in range(1, x.shape[1]-1, 1):
        for j in range(1, x.shape[1]-1, 1):
            y_diag_1[0, i-1, j-1, 0] = x[0, i-1, j-1, 0] * elem_mask[0, i-1, j-1, 0] *diag_coef_1 \
                                 + x[0, i-1, j+1, 0] * elem_mask[0, i-1, j, 0] *diag_coef_1 \
                                 + x[0, i+1, j-1, 0] * elem_mask[0, i, j-1, 0] *diag_coef_1 \
                                 + x[0, i+1, j+1, 0] * elem_mask[0, i, j, 0] *diag_coef_1
            y_side_1[0, i-1, j-1, 0] = x[0, i-1, j, 0] * (elem_mask[0, i-1, j-1, 0] + elem_mask[0, i-1, j, 0]) / 2. *side_coef_1 \
                                 + x[0, i, j-1, 0] * (elem_mask[0, i-1, j-1, 0] + elem_mask[0, i, j-1, 0]) / 2. *side_coef_1\
                                 + x[0, i, j + 1, 0] * (elem_mask[0, i-1, j, 0] + elem_mask[0, i, j, 0]) / 2. *side_coef_1\
                                 + x[0, i+1, j, 0] * (elem_mask[0, i, j-1, 0] + elem_mask[0, i, j, 0]) / 2. *side_coef_1
            y_diag_2[0, i-1, j-1, 0] = x[0, i-1, j-1, 0] * (1-elem_mask[0, i-1, j-1, 0]) *diag_coef_2 \
                                 + x[0, i-1, j+1, 0] * (1-elem_mask[0, i-1, j, 0] )*diag_coef_2 \
                                 + x[0, i+1, j-1, 0] * (1-elem_mask[0, i, j-1, 0]) *diag_coef_2 \
                                 + x[0, i+1, j+1, 0] * (1-elem_mask[0, i, j, 0]) *diag_coef_2
            y_side_2[0, i-1, j-1, 0] = x[0, i-1, j, 0] * (2-elem_mask[0, i-1, j-1, 0] - elem_mask[0, i-1, j, 0]) / 2. *side_coef_2 \
                                 + x[0, i, j-1, 0] * (2-elem_mask[0, i-1, j-1, 0] - elem_mask[0, i, j-1, 0]) / 2. *side_coef_2\
                                 + x[0, i, j + 1, 0] * (2-elem_mask[0, i-1, j, 0] - elem_mask[0, i, j, 0]) / 2. *side_coef_2\
                                 + x[0, i+1, j, 0] * (2-elem_mask[0, i, j-1, 0] - elem_mask[0, i, j, 0]) / 2. *side_coef_2

    tmp = {
        'LU_u_1': y_diag_1 + y_side_1,
        'LU_u_2': y_diag_2 + y_side_2,
        'LU_u': y_diag_1 + y_side_1 + y_diag_2 + y_side_2
    }
    return tmp


def jacobi_itr(u_input, f_input, d_matrix, elem_mask, coef):

    LU_u = masked_conv(elem_mask, u_input, coef)
    u_new = (f_input - LU_u['LU_u']) / d_matrix
    u_new = np.pad(u_new[:,:,1:,:], ((0, 0), (0, 0), (1, 0), (0, 0)),"constant")  # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!

    return u_new, LU_u


def main():
    resp_gt, load_gt, elem_mask, coef_dict_data = load_data_elem(case=-2)
    load_gt = np.expand_dims(np.expand_dims(load_gt,0),3)
    resp_gt = np.expand_dims(np.expand_dims(resp_gt,0),3)
    elem_mask = np.expand_dims(np.expand_dims(elem_mask,0),3)
    conductivity_1, conductivity_2 = coef_dict_data['conductivity_1'], coef_dict_data['conductivity_2']
    coef_dict = {
        'conductivity_1': conductivity_1,
        'conductivity_2': conductivity_2,
        'diag_coef_1': conductivity_1 * 1 / 3.,
        'side_coef_1': conductivity_1 * 1 / 3.,
        'diag_coef_2': conductivity_2 * 1 / 3.,
        'side_coef_2': conductivity_2 * 1 / 3.
    }
    n_elem_x = n_elem_y = 64

    d_matrix = get_D_matrix(elem_mask, coef_dict)

    n_itr = 30000
    u_hist = [np.zeros_like(resp_gt)]
    LU_u_hist = []
    loss_hist = []
    for i in range(n_itr):
        u_new, LU_u = jacobi_itr(u_hist[-1], load_gt, d_matrix, elem_mask, coef_dict)
        u_hist += [u_new]
        LU_u_hist += [LU_u]
        loss_i = np.mean(np.abs(u_hist[i] - resp_gt))
        loss_hist += [loss_i]
        print('n_itr: {}, loss: {}'.format(i,loss_i))


    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_hist)
    plt.figure()
    plt.imshow(u_hist[-1][0, 1:-1, 1:-1, 0], cmap='jet')
    plt.colorbar()
    plt.grid('off')
    plt.figure()
    plt.imshow(resp_gt[0, 1:-1, 1:-1, 0], cmap='jet')
    plt.colorbar()
    plt.grid('off')
    plt.show()
    return u_hist

if __name__ == '__main__':
    cfg = {
        'batch_size': 1,
        'imsize': 64,
        'physics_problem': 'heat_transfer',  # candidates: 3D plate elasticity, helmholtz, vibro-acoustics
        'alpha': 5000,  # iteration
    }

    u_hist = main()
    print('done')