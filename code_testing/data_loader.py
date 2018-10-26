import numpy as np
import scipy.io as sio


def load_data_elem(case):
    # case = 1
    if  case == -10:
        fx = np.asarray(([0,0,1],[0,0,1],[0,0,1]))
        fy = np.zeros((3,3))
        f = np.concatenate([np.expand_dims(fx, 2), np.expand_dims(fy, 2)], 2) / 1e10
        u = np.copy(f)
        mask_1 = np.ones((2,2))
        coef_dict = {
            'E_1': 20,  # 200e9,
            'mu_1': 0.25,
            'E_2': 20,  # 200e9,
            'mu_2': 0.25,
        }
    if case == 10:
        # linear elasticity, all steel, distributed load
        ux = sio.loadmat('./data/linear_elast_1phase_distributed/displacement_x_Steel_Steel_DisLoad.mat')['u1']
        uy = sio.loadmat('./data/linear_elast_1phase_distributed/displacement_y_Steel_Steel_DisLoad.mat')['u2']
        u = np.concatenate([np.expand_dims(ux, 2), np.expand_dims(uy, 2)], 2)
        fx = sio.loadmat('./data/linear_elast_1phase_distributed/f_x_Steel_Steel_DisLoad.mat')['f1']
        fy = sio.loadmat('./data/linear_elast_1phase_distributed/f_y_Steel_Steel_DisLoad.mat')['f2']
        f = -1. * np.concatenate([np.expand_dims(fx, 2), np.expand_dims(fy, 2)], 2) / 1e10
        # f = np.zeros_like(u)
        # f[0, :, -1, 0] = 1.
        mask_1 = sio.loadmat('./data/linear_elast_1phase_distributed/mask.mat')['ind2']
        coef_dict = {
            'E_1': 20,#200e9,
            'mu_1': 0.25,
            'E_2': 20,#200e9,
            'mu_2': 0.25,
        }
    if case == 11:
        # linear elasticity, all steel
        ux = sio.loadmat('./data/linear_elast_1phase/displacement_x_Steel_Steel.mat')['u1']
        uy = sio.loadmat('./data/linear_elast_1phase/displacement_y_Steel_Steel.mat')['u2']
        u = np.concatenate([np.expand_dims(ux, 2), np.expand_dims(uy, 2)], 2)
        fx = sio.loadmat('./data/linear_elast_1phase/f_x_Steel_Steel.mat')['f1']
        fy = sio.loadmat('./data/linear_elast_1phase/f_y_Steel_Steel.mat')['f2']
        f = -1. * np.concatenate([np.expand_dims(fx, 2), np.expand_dims(fy, 2)], 2) / 1e10
        # f = np.zeros_like(u)
        # f[0, :, -1, 0] = 1.
        mask_1 = sio.loadmat('./data/linear_elast_1phase/mask.mat')['ind2']
        coef_dict = {
            'E_1': 20,#200e9,
            'mu_1': 0.25,
            'E_2': 20,#200e9,
            'mu_2': 0.25,
        }

    if case == 12:
        # linear elasticity, all steel, Yfix
        ux = sio.loadmat('./data/linear_elast_1phase_Yfixed/displacement_x_Steel_Steel_Yfixd.mat')['u1']
        uy = sio.loadmat('./data/linear_elast_1phase_Yfixed/displacement_y_Steel_Steel_Yfixd.mat')['u2']
        u = np.concatenate([np.expand_dims(ux, 2), np.expand_dims(uy, 2)], 2)
        fx = sio.loadmat('./data/linear_elast_1phase_Yfixed/f_x_Steel_Steel_Yfixd.mat')['f1']
        fy = sio.loadmat('./data/linear_elast_1phase_Yfixed/f_y_Steel_Steel_Yfixd.mat')['f2']
        f = -1. * np.concatenate([np.expand_dims(fx, 2), np.expand_dims(fy, 2)], 2) / 1e10
        # f = np.zeros_like(u)
        # f[0, :, -1, 0] = 1.
        mask_1 = sio.loadmat('./data/linear_elast_1phase/mask.mat')['ind2']
        coef_dict = {
            'E_1': 20,#200e9,
            'mu_1': 0.25,
            'E_2': 20,#200e9,
            'mu_2': 0.25,
        }

    if case == 13:
        # linear elasticity, steel + al
        ux = sio.loadmat('./data/linear_elast_2phase/displacement_x_Steel_Al.mat')['u1']
        uy = sio.loadmat('./data/linear_elast_2phase/displacement_y_Steel_Al.mat')['u2']
        u = np.concatenate([np.expand_dims(ux, 2), np.expand_dims(uy, 2)], 2)
        # fx = sio.loadmat('./data/linear_elast_2phase/f_x_Steel_Al.mat')['f1']
        # fy = sio.loadmat('./data/linear_elast_2phase/f_y_Steel_Al.mat')['f2']
        # f = np.concatenate([np.expand_dims(fx, 2), np.expand_dims(fy, 2)], 2)
        f = np.zeros_like(u)
        f[0, :, -1, 0] = 1.
        mask_1 = sio.loadmat('./data/linear_elast_2phase/mask.mat')['ind2']
        coef_dict = {
            'E_1': 20,#200e9,
            'mu_1': 0.25,
            'E_2': 6.8,#68e9,
            'mu_2': 0.36,
        }

    if case in[-2, -1, 0, 1]:
        # heat transfer
        if case == -2:
            # all steel
            # u = np.zeros((66, 66), 'float32')
            u = sio.loadmat('/home/hope-yao/Downloads/steel_U.mat')['U1'][0][1:-1, 1:-1]
            f = sio.loadmat('/home/hope-yao/Downloads/steel_q.mat')['F1'][0][1:-1,1:-1]
            mask_1 = np.asarray([[1., ] * 43 + [0.] * 20] * 63, dtype='float32')
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


def load_data(case):
    # case = 1
    if case == -1:
        # toy case
        mask_1 =    np.asarray([[1,   1,   1,   1,   1,   1,   1,   1,   1,   1,],
                                [1,   1,   1,   1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,],
                                [1,   1,   1,   1, 0.5,   0,   0,   0,   0,   0,],
                                [1,   1,   1,   1, 0.5,   0,   0,   0,   0,   0,],
                                [1,   1,   1,   1, 0.5,   0,   0,   0,   0,   0,],
                                [1,   1,   1,   1, 0.5, 0.5, 0.5,   0,   0,   0,],
                                [1,   1,   1,   1,   1,   1, 0.5,   0,   0,   0,],
                                [1,   1,   1,   1,   1,   1, 0.5,   0,   0,   0,],
                                [1,   1,   1,   1,   1,   1, 0.5,   0,   0,   0,],
                                [1,   1,   1,   1,   1,   1, 0.5,   0,   0,   0,]])
        mask_1 = np.asarray(mask_1, dtype='float32').reshape(1, 10, 10, 1)
        mask_2 = np.ones_like(mask_1, dtype='float32') - mask_1
        mask_2 = np.asarray(mask_2, dtype='float32').reshape(1, 10, 10, 1)
        f = u = np.ones_like(mask_1, dtype='float32')
        conductivity_1 = 10.
        conductivity_2 = 100.

    elif case == 0:
        # all steel
        u = np.zeros((1, 66, 66, 1), 'float32')
        u[0, 1:-1, 1:-1, 0] = sio.loadmat('/home/hope-yao/Downloads/steel_U.mat')['U1'][0][1:-1, 1:-1]
        f = sio.loadmat('/home/hope-yao/Downloads/steel_q.mat')['F1'][0]
        mask_1 = np.asarray([[1., ] * 33 + [0.5] * 1 + [0.] * 32] * 66, dtype='float32').reshape(1, 66, 66, 1)
        mask_2 = np.asarray([[0., ] * 33 + [0.5] * 1 + [1.] * 32] * 66, dtype='float32').reshape(1, 66, 66, 1)
        conductivity_1 = 16.
        conductivity_2 = 16.
    elif case==1:
        # left: steel, right: aluminum
        u = np.zeros((1, 66, 66, 1), 'float32')
        u[0, 1:-1, 1:-1, 0] = sio.loadmat('/home/hope-yao/Downloads/solution_6666.mat')['U1'][0][1:-1, 1:-1]
        f = sio.loadmat('/home/hope-yao/Downloads/q_6666.mat')['F1'][0]
        mask_1 = np.asarray([[1., ] * 33 + [0.5] * 1 + [0.] * 32] * 66, dtype='float32').reshape(1, 66, 66, 1)
        mask_2 = np.asarray([[0., ] * 33 + [0.5] * 1 + [1.] * 32] * 66, dtype='float32').reshape(1, 66, 66, 1)
        conductivity_1 = 16.
        conductivity_2 = 205.
    elif case==2:
        # center: aluminum, outer: steel
        u = np.zeros((1, 66, 66, 1), 'float32')
        u[0, 1:-1, 1:-1, 0] = sio.loadmat('/home/hope-yao/Downloads/solution_square.mat')['U1'][0][1:-1, 1:-1]
        f = np.zeros((1, 66, 66, 1), 'float32')
        f[0, 1:-1, 1:-1, 0] = sio.loadmat('/home/hope-yao/Downloads/q_square.mat')['F1'][0][1:-1, 1:-1]
        mask0 = sio.loadmat('/home/hope-yao/Downloads/index_square.mat')['B']
        mask = np.asarray(mask0, 'int')
        mask_1 = np.zeros((66, 66), dtype='float32')
        mask_2 = np.zeros((66, 66), dtype='float32')
        for i in range(66):
            for j in range(66):
                if mask[i, j] == 1:
                    mask_1[i, j] = 1
                if mask[i, j] == 0:
                    mask_1[i, j] = 0.5
                    mask_2[i, j] = 0.5
                if mask[i, j] == -1:
                    mask_2[i, j] = 1
        mask_1 = np.asarray(mask_1, dtype='float32').reshape(1, 66, 66, 1)
        mask_2 = np.asarray(mask_2, dtype='float32').reshape(1, 66, 66, 1)
        conductivity_1 = 16.
        conductivity_2 = 205.
    return u, f, mask_1, mask_2, conductivity_1, conductivity_2

