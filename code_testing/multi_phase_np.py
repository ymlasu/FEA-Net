import numpy as np
from custom_ops_np import np_faster_mask_conv_correct as masked_conv
from custom_ops_np import np_get_D_matrix as get_D_matrix
from data_loader import load_data_elem

def jacobi_itr(u_input, f_input, d_matrix, elem_mask, coef):

    LU_u = masked_conv(elem_mask, u_input, coef)
    u_new = (f_input - LU_u['LU_u']) / d_matrix
    u_new = np.pad(u_new[:,:,1:,:], ((0, 0), (0, 0), (1, 0), (0, 0)),"constant")  # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!

    return u_new, LU_u


def main():
    resp_gt, load_gt, elem_mask, coef_dict_data = load_data_elem(case=0)
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