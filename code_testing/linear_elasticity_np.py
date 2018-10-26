import numpy as np
def np_get_D_matrix_elast(elem_mask, coef_dict, mode='symm'):
    '''

    :param elem_mask:
    :param conductivity_1:
    :param conductivity_2:
    :return:
    '''
    # convolution with symmetric padding at boundary
    from scipy import signal
    elem_mask = np.squeeze(elem_mask)
    node_filter = np.asarray([[1 / 4.] * 2] * 2)
    # first material phase
    node_mask_1 = signal.correlate2d(elem_mask, node_filter, boundary=mode)#symm, fill
    # second material phase
    node_mask_2 = signal.correlate2d(np.ones_like(elem_mask) - elem_mask, node_filter, boundary=mode)

    wxx_1_center, wyy_1_center = coef_dict['wxx_1'][1,1], coef_dict['wyy_1'][1,1]
    wxx_2_center, wyy_2_center = coef_dict['wxx_2'][1,1], coef_dict['wyy_2'][1,1]

    d_matrix_xx = node_mask_1 * wxx_1_center + node_mask_2 * wxx_2_center
    d_matrix_yy = node_mask_1 * wyy_1_center + node_mask_2 * wyy_2_center
    d_matrix = np.concatenate([np.expand_dims(np.expand_dims(d_matrix_xx, 0), 3),
                               np.expand_dims(np.expand_dims(d_matrix_yy, 0), 3)
                               ],3)

    return d_matrix

def sym_padding(x):
    # x[:, 0, 0, :] *= 4
    # x[:, 0, -1, :] *= 4
    # x[:, -1, 0, :] *= 4
    # x[:, -1, -1, :] *= 4
    left = x[:, :, 1:2, :]
    right = x[:, :, -2:-1, :]
    upper = np.concatenate([x[:, 1:2, 1:2, :], x[:, 1:2, :, :], x[:, 1:2, -2:-1, :]], 2)
    down = np.concatenate([x[:, -2:-1, 1:2, :], x[:, -2:-1, :, :], x[:, -2:-1, -2:-1, :]], 2)
    padded_x = np.concatenate([left, x, right], 2)
    padded_x = np.concatenate([upper, padded_x, down], 1)
    return padded_x

def np_faster_mask_conv_elast(elem_mask, node_resp, coef):
    wxx_1, wxy_1, wyx_1, wyy_1 = coef['wxx_1'], coef['wxy_1'], coef['wyx_1'], coef['wyy_1'],
    wxx_2, wxy_2, wyx_2, wyy_2 = coef['wxx_2'], coef['wxy_2'], coef['wyx_2'], coef['wyy_2'],

    diag_coef_diff = wxx_1[0,0] - wxx_2[0,0]
    fist_side_coef_diff = wxx_1[1,0] - wxx_2[1,0]
    second_side_coef_diff = wxx_1[0,1] - wxx_2[0,1]
    diag_coef_2 = wxx_2[0,0]
    first_side_coef_2 = wxx_2[1,0]
    second_side_coef_2 = wxx_2[0,1]
    coupling_coef_diff = wxy_1[0,0] - wxy_2[0,0]
    coupling_coef_2 = wxy_2[0,0]

    node_resp_x = node_resp[:,:,:,:1]
    node_resp_y = node_resp[:,:,:,1:]
    zero_padded_resp_x = np.pad(node_resp_x, ((0, 0), (1, 1), (1, 1), (0, 0)), "constant")#constant
    zero_padded_resp_y = np.pad(node_resp_y, ((0, 0), (1, 1), (1, 1), (0, 0)), "constant")
    padded_resp_x = sym_padding(node_resp_x)
    padded_resp_y = sym_padding(node_resp_y)
    padded_mask = np.pad(elem_mask, ((0, 0), (1, 1), (1, 1), (0, 0)), "symmetric")
    for i in range(1, padded_resp_x.shape[1]-1, 1):
        for j in range(1, padded_resp_x.shape[2]-1, 1):
            conv_result_x_i_j = \
            padded_mask[0, i - 1, j - 1, 0] * \
            (
                    padded_resp_x[0, i - 1, j - 1, 0] * diag_coef_diff
                    + padded_resp_x[0, i, j - 1, 0] * fist_side_coef_diff/ 2.
                    + padded_resp_x[0, i - 1, j, 0] * second_side_coef_diff/ 2.
            ) + \
            padded_mask[0, i - 1, j, 0] * \
            (
                    padded_resp_x[0, i - 1, j + 1, 0] * diag_coef_diff
                    + padded_resp_x[0, i, j + 1, 0]   * fist_side_coef_diff/ 2.
                    + padded_resp_x[0, i - 1, j, 0]   * second_side_coef_diff/ 2.
            ) + \
            padded_mask[0, i, j - 1, 0] * \
            (
                    padded_resp_x[0, i + 1, j - 1, 0] * diag_coef_diff
                    + padded_resp_x[0, i, j - 1, 0]    * fist_side_coef_diff/ 2.
                    + padded_resp_x[0, i + 1, j, 0] * second_side_coef_diff/ 2.
            ) + \
            padded_mask[0, i, j, 0] * \
            (
                    padded_resp_x[0, i + 1, j + 1, 0] * diag_coef_diff
                    + padded_resp_x[0, i, j + 1, 0]   * fist_side_coef_diff/ 2.
                    + padded_resp_x[0, i + 1, j, 0] * second_side_coef_diff/ 2.
            ) + \
            coupling_coef_diff * \
            (
                    zero_padded_resp_y[0, i - 1, j - 1, 0] - zero_padded_resp_y[0, i - 1, j + 1, 0]
                    - zero_padded_resp_y[0, i + 1, j - 1, 0] + zero_padded_resp_y[0, i + 1, j + 1, 0]
            ) + \
            diag_coef_2 * \
            (
                    padded_resp_x[0, i + 1, j - 1, 0] + padded_resp_x[0, i + 1, j + 1, 0]
                    + padded_resp_x[0, i - 1, j - 1, 0] + padded_resp_x[0, i - 1, j + 1, 0]

            ) + \
            first_side_coef_2 * \
            (
                    padded_resp_x[0, i, j + 1, 0] + padded_resp_x[0, i, j - 1, 0]
            ) + \
            second_side_coef_2 * \
            (
                    padded_resp_x[0, i - 1, j, 0] + padded_resp_x[0, i + 1, j, 0]
            ) + \
            coupling_coef_2 * \
            (
                    zero_padded_resp_y[0, i - 1, j - 1, 0] - zero_padded_resp_y[0, i - 1, j + 1, 0]
                    - zero_padded_resp_y[0, i + 1, j - 1, 0] + zero_padded_resp_y[0, i + 1, j + 1, 0]
            )
            # response in x direction

            conv_result_x = np.reshape(conv_result_x_i_j, (1, 1)) if i == 1 and j == 1 \
                else np.concatenate([conv_result_x, np.reshape(conv_result_x_i_j, (1, 1))], axis=0)

            # response in y direction
            conv_result_y_i_j = \
                padded_mask[0, i - 1, j - 1, 0] * \
                (
                        padded_resp_y[0, i - 1, j - 1, 0] * diag_coef_diff
                        + padded_resp_y[0, i, j - 1, 0] * second_side_coef_diff / 2.
                        + padded_resp_y[0, i - 1, j, 0] * fist_side_coef_diff / 2.
                ) + \
                padded_mask[0, i - 1, j, 0] * \
                (
                        padded_resp_y[0, i - 1, j + 1, 0] * diag_coef_diff
                        + padded_resp_y[0, i, j + 1, 0] * second_side_coef_diff / 2.
                        + padded_resp_y[0, i - 1, j, 0] * fist_side_coef_diff / 2.
                ) + \
                padded_mask[0, i, j - 1, 0] * \
                (
                        padded_resp_y[0, i + 1, j - 1, 0] * diag_coef_diff
                        + padded_resp_y[0, i, j - 1, 0] * second_side_coef_diff / 2.
                        + padded_resp_y[0, i + 1, j, 0] * fist_side_coef_diff / 2.
                ) + \
                padded_mask[0, i, j, 0] * \
                (
                        padded_resp_y[0, i + 1, j + 1, 0] * diag_coef_diff
                        + padded_resp_y[0, i, j + 1, 0] * second_side_coef_diff / 2.
                        + padded_resp_y[0, i + 1, j, 0] * fist_side_coef_diff / 2.
                ) + \
                coupling_coef_diff * \
                (
                        zero_padded_resp_x[0, i - 1, j - 1, 0] - zero_padded_resp_x[0, i - 1, j + 1, 0]
                        - zero_padded_resp_x[0, i + 1, j - 1, 0] + zero_padded_resp_x[0, i + 1, j + 1, 0]
                ) + \
                diag_coef_2 * \
                (
                        padded_resp_y[0, i + 1, j - 1, 0] + padded_resp_y[0, i + 1, j + 1, 0]
                        + padded_resp_y[0, i - 1, j - 1, 0] + padded_resp_y[0, i - 1, j + 1, 0]
                ) + \
                first_side_coef_2 * \
                (
                        padded_resp_y[0, i - 1, j, 0] + padded_resp_y[0, i + 1, j, 0]
                ) + \
                second_side_coef_2 * \
                (
                        padded_resp_y[0, i, j + 1, 0] + padded_resp_y[0, i, j - 1, 0]
                ) + \
                coupling_coef_2 * \
                (
                        zero_padded_resp_x[0, i - 1, j - 1, 0] - zero_padded_resp_x[0, i - 1, j + 1, 0]
                        - zero_padded_resp_x[0, i + 1, j - 1, 0] + zero_padded_resp_x[0, i + 1, j + 1, 0]
                )
            conv_result_y = np.reshape(conv_result_y_i_j, (1, 1)) if i == 1 and j == 1 \
                else np.concatenate([conv_result_y, np.reshape(conv_result_y_i_j, (1, 1))], axis=0)

    LU_u_x = np.reshape(conv_result_x, (node_resp_x.shape))
    LU_u_y = np.reshape(conv_result_y, (node_resp_y.shape))
    LU_u = np.concatenate([LU_u_x,LU_u_y],3)
    weight = np.ones_like(LU_u)
    # weight[:, 0, :, :] /= 2
    # weight[:, -1, :, :] /= 2
    # weight[:, :, 0, :] /= 2
    # weight[:, :, -1, :] /= 2
    tmp = {
        'LU_u': LU_u*weight,
    }
    return tmp


import numpy as np

def jacobi_itr(u_input, f_input, d_matrix, elem_mask, coef):

    LU_u = np_faster_mask_conv_elast(elem_mask, u_input, coef)
    u_new = (f_input - LU_u['LU_u']) / d_matrix
    u_new = np.concatenate([
        np.pad(u_new[:,:,1:,:1], ((0, 0), (0, 0), (1, 0), (0, 0)),"constant")
        ,u_new[:,:,:,1:]],3)  # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!
    return u_new, LU_u

def get_w_matrix(E, mu):
    cost_coef = E/16./(1-mu**2)
    wxx = cost_coef * np.asarray([
        [-4*(1 - mu / 3.),        16 * mu / 3.,             -4*(1 - mu / 3.)],
        [-8*(1 + mu / 3.),       32. * (1 - mu / 3.),       -8*(1 + mu / 3.)],
        [-4*(1 - mu / 3.),        16 * mu / 3.,             -4*(1 - mu / 3.)],
    ])

    wxy = wyx = cost_coef*4 * np.asarray([
        [2 * (mu + 1),        0,             -2 * (mu + 1)],
        [0,                        0,                  0],
        [-2 * (mu + 1),        0,             2 * (mu + 1)],
    ])

    wyy = cost_coef * np.asarray([
        [-4 * (1 - mu / 3.),         -8 * (1 + mu / 3.),        -4 * (1 - mu / 3.)],
        [16 * mu / 3.,               32. * (1 - mu / 3.),        16 * mu / 3.],
        [-4 * (1 - mu / 3.),         -8 * (1 + mu / 3.),        -4 * (1 - mu / 3.)],
    ])
    return wxx, wxy, wyx, wyy

def main():
    from data_loader import load_data_elem
    resp_gt, load_gt, elem_mask, coef_dict_data = load_data_elem(case=12)
    load_gt = np.expand_dims(load_gt,0)
    resp_gt = np.expand_dims(resp_gt,0)
    elem_mask = np.expand_dims(np.expand_dims(elem_mask,0),3)
    mu_1, E_1 = coef_dict_data['mu_1'], coef_dict_data['E_1']
    mu_2, E_2 = coef_dict_data['mu_2'], coef_dict_data['E_2']
    wxx_1, wxy_1, wyx_1, wyy_1 = get_w_matrix(E_1, mu_1)
    wxx_2, wxy_2, wyx_2, wyy_2 = get_w_matrix(E_2, mu_2)
    coef_dict = {
        # material 1
        'mu_1': mu_1,
        'E_1': E_1,
        'wxx_1': wxx_1 ,
        'wxy_1': wxy_1,
        'wyx_1':wyx_1,
        'wyy_1': wyy_1,
        # material 2
        'mu_2': mu_2,
        'E_2': E_2,
        'wxx_2': wxx_2,
        'wxy_2': wxy_2,
        'wyx_2': wyx_2,
        'wyy_2': wyy_2,
    }
    n_elem_x = n_elem_y = 64
    d_matrix = np_get_D_matrix_elast(elem_mask, coef_dict)

    if 1:
        LU_u = np_faster_mask_conv_elast(elem_mask, resp_gt, coef_dict)
        wx = d_matrix * resp_gt + LU_u['LU_u']
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(wx[0, :, :, 0],interpolation='None')
        plt.colorbar()
        plt.figure()
        plt.imshow(wx[0, :, :, 1],interpolation='None')
        plt.colorbar()
        plt.show()


    n_itr = 10000
    u_hist = [np.zeros_like(resp_gt)]#[resp_gt]#
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