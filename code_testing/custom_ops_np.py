'''
This part is for customized differentiable convolution based on element mask
operation in python, will be converted in to CUDA
Hope Yao @2018.05.09
'''

import numpy as np

def np_get_D_matrix(elem_mask, coef_dict):
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

def np_mask_conv(elem_mask, node_resp, diag_coef, side_coef):
    '''
    use numpy to test the mask convolution
    :param elem_mask: mask on the element
    :param x: response field
    :param diag_coef: coefficient on the diagnal part of the element stiffness matrix
    :param side_coef: 2x coefficient on the side part of the element stiffness matrix
    :return:
    '''
    x = np.pad(node_resp,((0,0),(1,1),(1,1),(0,0)), "symmetric")
    elem_mask = np.pad(elem_mask,((0,0),(1,1),(1,1),(0,0)), "symmetric")
    # diagnal part
    y_diag = np.zeros_like(node_resp)
    for i in range(1, x.shape[1]-1, 1):
        for j in range(1, x.shape[1]-1, 1):
            y_diag[0, i-1, j-1, 0] = x[0, i-1, j-1, 0] * elem_mask[0, i-1, j-1, 0] \
                                 + x[0, i-1, j+1, 0] * elem_mask[0, i-1, j, 0] \
                                 + x[0, i+1, j-1, 0] * elem_mask[0, i, j-1, 0] \
                                 + x[0, i+1, j+1, 0] * elem_mask[0, i, j, 0]
    # side part
    y_side = np.zeros_like(node_resp)
    for i in range(1, x.shape[1]-1, 1):
        for j in range(1, x.shape[1]-1, 1):
            y_side[0, i-1, j-1, 0] = x[0, i-1, j, 0] * (elem_mask[0, i-1, j-1, 0] + elem_mask[0, i-1, j, 0]) / 2. \
                                 + x[0, i, j-1, 0] * (elem_mask[0, i-1, j-1, 0] + elem_mask[0, i, j-1, 0]) / 2. \
                                 + x[0, i, j + 1, 0] * (elem_mask[0, i-1, j, 0] + elem_mask[0, i, j, 0]) / 2. \
                                 + x[0, i+1, j, 0] * (elem_mask[0, i, j-1, 0] + elem_mask[0, i, j, 0]) / 2.
    return diag_coef * y_diag + side_coef * y_side

def np_fast_mask_conv(elem_mask, node_resp, coef):
    '''
    combined two tf_mask_conv together
    :param elem_mask:
    :param node_resp:
    :param coef:
    :return:
    '''
    diag_coef_1, side_coef_1 = coef['diag_coef_1'], coef['diag_coef_1']
    diag_coef_2, side_coef_2 = coef['diag_coef_2'], coef['diag_coef_2']
    # node_resp = node_resp[0,:,:,0]
    # elem_mask = elem_mask[0,:,:,0]
    padded_resp = np.pad(node_resp, ((0,0), (1, 1), (1, 1), (0, 0)), "symmetric")
    padded_mask = np.pad(elem_mask, ((0,0), (1, 1), (1, 1), (0, 0)), "symmetric")
    # diagnal part
    for i in range(1, padded_resp.shape[1]-1, 1):
        for j in range(1, padded_resp.shape[1]-1, 1):
            y_diag_i_j = padded_resp[0, i-1, j-1, 0] * padded_mask[0, i-1, j-1, 0] \
                     + padded_resp[0, i-1, j+1, 0] * padded_mask[0, i-1, j, 0] \
                     + padded_resp[0, i+1, j-1, 0] * padded_mask[0, i, j-1, 0] \
                     + padded_resp[0, i+1, j+1, 0] * padded_mask[0, i, j, 0]
            y_diag = np.reshape(y_diag_i_j, (1, 1)) if i == 1 and j == 1 else np.concatenate([y_diag, np.reshape(y_diag_i_j, (1, 1))], axis=0)
    y_diag = np.reshape(y_diag, (node_resp.shape))
    # side part
    for i in range(1, padded_resp.shape[2]-1, 1):
        for j in range(1, padded_resp.shape[1]-1, 1):
            y_side_i_j = padded_resp[0, i-1, j, 0] * (padded_mask[0, i-1, j-1, 0] + padded_mask[0, i-1, j, 0]) / 2. \
                     + padded_resp[0, i, j-1, 0] * (padded_mask[0, i-1, j-1, 0] + padded_mask[0, i, j-1, 0]) / 2. \
                     + padded_resp[0, i, j + 1, 0] * (padded_mask[0, i-1, j, 0] + padded_mask[0, i, j, 0]) / 2. \
                     + padded_resp[0, i+1, j, 0] * (padded_mask[0, i, j-1, 0] + padded_mask[0, i, j, 0]) / 2.
            y_side = np.reshape(y_side_i_j, (1, 1)) if i == 1 and j == 1 else np.concatenate([y_side, np.reshape(y_side_i_j, (1, 1))], axis=0)
    y_side = np.reshape(y_side, (node_resp.shape))
    # remaining
    for i in range(1, padded_resp.shape[2]-1, 1):
        for j in range(1, padded_resp.shape[1]-1, 1):
            y_remain_i_j = padded_resp[0, i-1, j, 0]  + padded_resp[0, i, j-1, 0]  \
                         + padded_resp[0, i, j + 1, 0]+ padded_resp[0, i+1, j, 0]
            y_remain = np.reshape(y_remain_i_j, (1, 1)) if i == 1 and j == 1 else np.concatenate([y_remain, np.reshape(y_remain_i_j, (1, 1))], axis=0)
    y_remain = np.reshape(y_remain, (node_resp.shape))
    LU_u = (diag_coef_1 - diag_coef_2) * y_diag + diag_coef_2 * y_remain\
            + (side_coef_1 - side_coef_2) * y_side  + side_coef_2 * y_remain
    tmp = {
        'LU_u': LU_u
    }
    return tmp

def np_faster_mask_conv_correct(elem_mask_orig, node_resp, coef):
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

def sym_padding(x):
    x[:, 0, 0, :] *= 4
    x[:, 0, -1, :] *= 4
    x[:, -1, 0, :] *= 4
    x[:, -1, -1, :] *= 4
    left = x[:, :, 1:2, :]
    right = x[:, :, -2:-1, :]
    upper = np.concatenate([x[:, 1:2, 1:2, :], x[:, 1:2, :, :], x[:, 1:2, -2:-1, :]], 2)
    down = np.concatenate([x[:, -2:-1, 1:2, :], x[:, -2:-1, :, :], x[:, -2:-1, -2:-1, :]], 2)
    padded_x = np.concatenate([left, x, right], 2)
    padded_x = np.concatenate([upper, padded_x, down], 1)
    return padded_x

def np_faster_mask_conv(elem_mask, node_resp, coef):
    diag_coef_1, side_coef_1 = coef['diag_coef_1'], coef['diag_coef_1']
    diag_coef_2, side_coef_2 = coef['diag_coef_2'], coef['diag_coef_2']
    diag_coef_diff = diag_coef_1 - diag_coef_2
    side_coef_diff = side_coef_1 - side_coef_2
    padded_resp = sym_padding(node_resp)
    # padded_resp = np.pad(node_resp, ((0, 0), (1, 1), (1, 1), (0, 0)), "symmetric")
    padded_mask = np.pad(elem_mask, ((0, 0), (1, 1), (1, 1), (0, 0)), "symmetric")
    for i in range(1, padded_resp.shape[1]-1, 1):
        for j in range(1, padded_resp.shape[1]-1, 1):
            conv_result_i_j = \
            padded_mask[0, i - 1, j - 1, 0] * \
            (
                    padded_resp[0, i - 1, j - 1, 0] * diag_coef_diff
                    + (padded_resp[0, i - 1, j, 0] + padded_resp[0, i, j - 1, 0]) / 2. * side_coef_diff
            ) + \
            padded_mask[0, i - 1, j, 0] * \
            (
                    padded_resp[0, i - 1, j + 1, 0] * diag_coef_diff
                    + (padded_resp[0, i - 1, j, 0] + padded_resp[0, i, j + 1, 0]) / 2. * side_coef_diff
            ) + \
            padded_mask[0, i, j - 1, 0] * \
            (
                    padded_resp[0, i + 1, j - 1, 0] * diag_coef_diff
                    + (padded_resp[0, i, j - 1, 0] + padded_resp[0, i + 1, j, 0]) / 2. * side_coef_diff
            ) + \
            padded_mask[0, i, j, 0] * \
            (
                    padded_resp[0, i + 1, j + 1, 0] * diag_coef_diff
                    + (padded_resp[0, i, j + 1, 0] + padded_resp[0, i + 1, j, 0]) / 2. * side_coef_diff
            ) + \
            diag_coef_2 * \
            (
                    padded_resp[0, i - 1, j, 0] + padded_resp[0, i, j - 1, 0]
                    + padded_resp[0, i, j + 1, 0] + padded_resp[0, i + 1, j, 0]
            ) + \
            side_coef_2 * \
            (
                    padded_resp[0, i - 1, j - 1, 0] + padded_resp[0, i - 1, j + 1, 0]
                    + padded_resp[0, i + 1, j - 1, 0] + padded_resp[0, i + 1, j + 1, 0]
            )
            conv_result = np.reshape(conv_result_i_j, (1, 1)) if i == 1 and j == 1 else np.concatenate([conv_result, np.reshape(conv_result_i_j, (1, 1))], axis=0)
    conv_result = np.reshape(conv_result, (node_resp.shape))
    weight = np.ones_like(conv_result)
    weight[0,:] /= 2
    weight[-1,:] /= 2
    weight[:,0] /= 2
    weight[:,-1] /= 2
    LU_u = np.reshape(conv_result* weight, (node_resp.shape))
    tmp = {
        'LU_u': LU_u
    }
    return tmp

if __name__ == '__main__':
    from data_loader import load_data_elem
    resp_gt, load_gt, elem_mask, coef_dict = load_data_elem(case=0)
    conductivity_1, conductivity_2 = coef_dict['conductivity_1'], coef_dict['conductivity_2']
    resp_gt = np.expand_dims(np.expand_dims(resp_gt,0),3)
    elem_mask = np.expand_dims(np.expand_dims(elem_mask,0),3)
    coef_dict = {
        'conductivity_1': conductivity_1,
        'conductivity_2': conductivity_2,
        'diag_coef_1': conductivity_1 * 1 / 3.,
        'side_coef_1': conductivity_1 * 1 / 3.,
        'diag_coef_2': conductivity_2 * 1 / 3.,
        'side_coef_2': conductivity_2 * 1 / 3.
    }

    method = 'faster'
    if method == 'slow':
        load_pred_11 = np_mask_conv(elem_mask, resp_gt, coef_dict['diag_coef_1'], coef_dict['side_coef_1'])
        load_pred_12 = np_mask_conv(np.ones_like(elem_mask)-elem_mask, resp_gt, coef_dict['diag_coef_2'], coef_dict['side_coef_2'])
        load_pred_1 = {
         'LU_u': load_pred_11 + load_pred_12
        }
    if method == 'fast':
        load_pred_1 = np_fast_mask_conv(elem_mask, resp_gt, coef_dict)
    elif method == 'faster':
        load_pred_1 = np_faster_mask_conv(elem_mask, resp_gt, coef_dict)

    d_matrix = np_get_D_matrix(elem_mask, coef_dict)
    load_pred_2 = d_matrix * resp_gt
    load_pred = load_pred_1['LU_u'] + load_pred_2
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(np.squeeze(elem_mask), interpolation='None')
    plt.colorbar()
    plt.figure()
    plt.imshow(np.squeeze(load_pred)[:,1:-1], interpolation='None')
    plt.colorbar()
    plt.figure()
    plt.imshow(np.squeeze(resp_gt)[:,1:-1], interpolation='None')
    plt.colorbar()
    plt.show()
    print('done')
