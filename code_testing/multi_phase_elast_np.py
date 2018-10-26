import numpy as np
from linear_elasticity_np import np_faster_mask_conv_elast as masked_conv

def jacobi_itr_elast(u_input, f_input, d_matrix, elem_mask, coef):
    u_input_copy = np.copy(u_input)

    weight = np.ones_like(f_input)
    weight[:, 0, :, :] /= 2
    weight[:, -1, :, :] /= 2
    weight[:, :, 0, :] /= 2
    weight[:, :, -1, :] /= 2

    LU_u = masked_conv(elem_mask, u_input, coef)
    u_new = (0.1*u_input - 0.9*LU_u['LU_u']/ d_matrix) + 0.9*f_input/d_matrix
    u_new = 0.9 * (f_input-LU_u['LU_u']*weight) / (d_matrix*weight) + 0.1 * u_input_copy
    # u_new = np.concatenate([np.pad(u_new[:,:,1:,0:1], ((0, 0), (0, 0), (1, 0), (0, 0)),"constant"),
    #                         u_new[:, :, :, 1:2]],3)  # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!
    u_new = np.pad(u_new[:,:,1:,:], ((0, 0), (0, 0), (1, 0), (0, 0)),"constant")  # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!
    # import scipy.io as sio
    # aa = sio.loadmat('/home/hope-yao/Documents/MG_net/data/linear_elast_1phase_Yfixed/fea_jac_itr_sol_UX_fixed.mat')
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(aa['UX'][:, :, 2], interpolation='None')
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(u_new[0, :, :, 0], interpolation='None')
    # plt.colorbar()
    # plt.show()
    return u_new, LU_u

def weighted_jacobi_itr_elast(u_input, f_input, d_matrix, elem_mask, coef):

    LU_u = masked_conv(elem_mask, u_input, coef)
    w = 0.666
    u_new = w * ( f_input - LU_u['LU_u'])/ d_matrix +  (1-w) * u_input
    # u_new = np.concatenate([np.pad(u_new[:,:,1:,0:1], ((0, 0), (0, 0), (1, 0), (0, 0)),"constant"),
    #                         u_new[:, :, :, 1:2]],3)  # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!
    u_new = np.pad(u_new[:,:,1:,:], ((0, 0), (0, 0), (1, 0), (0, 0)),"constant")  # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!

    return u_new, LU_u


def main():
    from data_loader import load_data_elem
    from linear_elasticity_np import np_get_D_matrix_elast, get_w_matrix
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
    d_matrix = np_get_D_matrix_elast(elem_mask, coef_dict)

    n_itr = 10000
    u_hist = [np.zeros_like(load_gt)]
    LU_u_hist = []
    loss_hist = []
    for i in range(n_itr):
        u_new, LU_u = jacobi_itr_elast(u_hist[-1], load_gt, d_matrix, elem_mask, coef_dict)
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