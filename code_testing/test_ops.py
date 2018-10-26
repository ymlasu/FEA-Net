import tensorflow as tf
import numpy as np
from data_loader import load_data

def new_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def new_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def test_sigmoid2():
    '''
    superposition of two sigmoid function, return value at
    1/4, 1/2, 3/4,
    :return:
    '''
    plt.figure()
    x = np.linspace(0, 1, 128)
    for w in [1, 4, 8, 16, 32, 64, 128]:
        y = sess.run(1 / 4. + 1 / 4. * tf.sigmoid(w * (x - 1.25 / 4.)) + 1 / 4. * tf.sigmoid(w * (x - 2.75 / 4.)))
        plt.plot(x, y)
    plt.grid('on')
    p = plt.axvspan(1.5 / 4., 2.5 / 4., facecolor='#2ca02c', alpha=0.5)

def masked_conv_v3(u, filter_1, filter_2, mask_1, mask_2, conductivity_1, conductivity_2, filter_banks):
    '''
    problematic in d_matrix corner and conv_corner
    :param u:
    :param filter_1:
    :param filter_2:
    :param mask_1:
    :param mask_2:
    :param conductivity_1:
    :param conductivity_2:
    :param filter_banks:
    :return:
    '''
    boundary_mask = tf.round(mask_1 + 0.1) + tf.round(mask_2 + 0.1) - 1
    boundary_d_matrix = boundary_mask * (-8/3. * (conductivity_1+conductivity_2)/2.)
    mat1_d_matrix = tf.round(mask_1-0.1) * (-8/3. * conductivity_1)
    mat2_d_matrix = tf.round(mask_2-0.1)* (-8/3. * conductivity_2)
    d_matrix = boundary_d_matrix + mat1_d_matrix + mat2_d_matrix
    d_u = d_matrix * u

    # padded_input = tf.pad(u * mask_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
    # output_1 = tf.nn.conv2d(input=padded_input, filter=filter_banks['filter_1_side'], strides=[1, 1, 1, 1], padding='VALID')
    # padded_input = tf.pad(u * mask_2, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
    # output_2 = tf.nn.conv2d(input=padded_input, filter=filter_banks['filter_2_side'], strides=[1, 1, 1, 1], padding='VALID')
    # side_conv_res = output_1 + output_2
    padded_input = tf.pad(u * tf.round(mask_1 + 0.1), [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
    output_1_side = tf.nn.conv2d(input=padded_input, filter=filter_banks['filter_1_side'], strides=[1, 1, 1, 1], padding='VALID')
    padded_input = tf.pad(u * tf.round(mask_2 + 0.1), [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
    output_2_side = tf.nn.conv2d(input=padded_input, filter=filter_banks['filter_2_side'], strides=[1, 1, 1, 1], padding='VALID')
    res1 = output_1_side * mask_1 + output_2_side * mask_2
    padded_input = tf.pad(u * mask_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
    output_1_side_refine = tf.nn.conv2d(input=padded_input, filter=filter_banks['filter_1_side'], strides=[1, 1, 1, 1], padding='VALID')
    padded_input = tf.pad(u * mask_2, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
    output_2_side_refine = tf.nn.conv2d(input=padded_input, filter=filter_banks['filter_2_side'], strides=[1, 1, 1, 1], padding='VALID')
    res2 = output_1_side_refine* mask_1 + output_2_side_refine* mask_1
    side_conv_res = res1 * (tf.round(mask_1-0.1) + tf.round(mask_2-0.1))  + \
                    res2 * (tf.ones_like(mask_1) - tf.round(mask_1-0.1) - tf.round(mask_2-0.1))

    padded_input = tf.pad(u * tf.round(mask_1+0.1), [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
    output_1_corner = tf.nn.conv2d(input=padded_input, filter=filter_banks['filter_1_corner'], strides=[1, 1, 1, 1], padding='VALID')
    padded_input = tf.pad(u * tf.round(mask_2+0.1), [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
    output_2_corner = tf.nn.conv2d(input=padded_input, filter=filter_banks['filter_2_corner'], strides=[1, 1, 1, 1], padding='VALID')
    corner_conv_res = output_1_corner + output_2_corner

    LU_u = corner_conv_res + side_conv_res
    result = d_u + LU_u
    tmp = {'result': result,
           'LU_u': LU_u,
           'd_matrix': d_matrix,
           'res1': res1,
           'res2': res2,
           'side_conv_res': side_conv_res,
           'output_1_side': output_1_side,
           'output_2_side': output_2_side,
           'output_1_side_refine': output_1_side_refine,
           'output_2_side_refine': output_2_side_refine,
           'corner_conv_res': corner_conv_res,
           'output_1_corner': output_1_corner,
           'output_2_corner': output_2_corner,
           }
    return tmp

def masked_conv_v2(u, filter_1, filter_2, mask_1, mask_2, conductivity_1, conductivity_2, eps=0.01, filter_banks=None):
    '''not accurate still around the corner'''
    # center
    mask_filter = np.asarray([[1 / 4., 0., 1 / 4.], [0., 0., 0.], [1 / 4., 0., 1 / 4.]], 'float32')
    mask_filter = tf.constant(mask_filter.reshape((3, 3, 1, 1)))
    padded_mask_1 = tf.pad(mask_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    boundary_weight = tf.nn.conv2d(input=padded_mask_1, filter=mask_filter, strides=[1, 1, 1, 1], padding='VALID')
    w = 1000.
    boundary_weight = 1/4. + 1/4. * tf.sigmoid(w*(boundary_weight-1.25/4.)) + 1/4. * tf.sigmoid(w*(boundary_weight-2.75/4.))
    boundary_mask = tf.round(mask_1 + 0.1) + tf.round(mask_2 + 0.1) - 1
    mat1_mask = (boundary_weight) * boundary_mask * (-8/3. * conductivity_1)
    mat2_mask = (1-boundary_weight) * boundary_mask * (-8/3. * conductivity_2)
    boundary_d_matrix = mat1_mask + mat2_mask
    mat1_d_matrix = tf.round(mask_1-0.1) * (-8/3. * conductivity_1)
    mat2_d_matrix = tf.round(mask_2-0.1)* (-8/3. * conductivity_2)
    d_matrix = boundary_d_matrix + mat1_d_matrix + mat2_d_matrix
    d_u = d_matrix * u

    # surrounding
    # not accurate only on the boundary, two parts from two different material
    padded_u_1 = tf.pad(u * tf.round(mask_1 + eps), [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    output_1 = tf.nn.conv2d(input=padded_u_1, filter=filter_1, strides=[1, 1, 1, 1], padding='VALID')
    padded_u_2 = tf.pad(u * tf.round(mask_2 + eps), [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    output_2 = tf.nn.conv2d(input=padded_u_2, filter=filter_2, strides=[1, 1, 1, 1], padding='VALID')
    res1 = output_1 * mask_1 + output_2 * mask_2
    # accurate only on the boundary, two parts from two different material
    padded_u_1 = tf.pad(u * mask_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    output_1 = tf.nn.conv2d(input=padded_u_1, filter=filter_1, strides=[1, 1, 1, 1], padding='VALID')
    padded_u_2 = tf.pad(u * mask_2, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    output_2 = tf.nn.conv2d(input=padded_u_2, filter=filter_2, strides=[1, 1, 1, 1], padding='VALID')
    res2 = output_1 + output_2
    LU_u = res1 * (tf.round(mask_1-0.1) + tf.round(mask_2-0.1)) + \
          res2 * (tf.ones_like(mask_1) - tf.round(mask_1-0.1) - tf.round(mask_2-0.1))

    result = d_u + LU_u
    tmp = {'result': result,
           'res1': res1,
           'res2': res2,
           'LU_u': LU_u,
           'd_matrix': d_matrix,
           'd_u': d_u,
           'boundary_d_matrix': boundary_d_matrix,
           'boundary_weight': boundary_weight,
           'mat1_d_matrix': mat1_d_matrix,
           'mat2_d_matrix': mat2_d_matrix,
           }
    return tmp


def masked_conv_v1(u, filter_1, filter_2, mask_1, mask_2, conductivity_1, conductivity_2, eps=0.01, filter_banks=None):
    '''
    convolution with two different kernels
    :param u: input image
    :param filter_1:
    :param filter_2:
    :param mask_1: region where filter_1 is applied, with value (0.5-eps,0.5+eps) on the boundary
    :param mask_2: region where filter_2 is applied, with value (0.5-eps,0.5+eps) on the boundary
    :param eps: some numerical consideration
    :return:
    '''
    boundary_mask = tf.round(mask_1 + 0.1) + tf.round(mask_2 + 0.1) - 1
    boundary_d_matrix = boundary_mask * (-8/3. * (conductivity_1+conductivity_2)/2.)
    mat1_d_matrix = tf.round(mask_1-0.1) * (-8/3. * conductivity_1)
    mat2_d_matrix = tf.round(mask_2-0.1)* (-8/3. * conductivity_2)
    d_matrix = boundary_d_matrix + mat1_d_matrix + mat2_d_matrix

    padded_input = tf.pad(u * tf.round(mask_1 + eps), [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
    output_1 = tf.nn.conv2d(input=padded_input, filter=filter_1, strides=[1, 1, 1, 1], padding='VALID')
    padded_input = tf.pad(u * tf.round(mask_2 + eps), [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
    output_2 = tf.nn.conv2d(input=padded_input, filter=filter_2, strides=[1, 1, 1, 1], padding='VALID')
    res1 = output_1 * mask_1 + output_2 * mask_2
    padded_input = tf.pad(u * mask_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
    output_1 = tf.nn.conv2d(input=padded_input, filter=filter_1, strides=[1, 1, 1, 1], padding='VALID')
    padded_input = tf.pad(u * mask_2, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
    output_2 = tf.nn.conv2d(input=padded_input, filter=filter_2, strides=[1, 1, 1, 1], padding='VALID')
    res2 = output_1 + output_2
    LU_u = res1 * (tf.round(mask_1) + tf.round(mask_2))  + \
          res2 * (tf.ones_like(mask_1) - tf.round(mask_1) - tf.round(mask_2))
    result = LU_u + (d_matrix*u)

    tmp = {
           'd_matrix': d_matrix,
           'LU_u': LU_u,
           'result': result,
           }
    return tmp

def test_mask_conv_v1(data_case):
    u, f, mask_1, mask_2, conductivity_1, conductivity_2 = load_data(data_case)

    w_filter_1 = np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], 'float32') * conductivity_1 / 3.
    w_filter_1 = tf.constant(w_filter_1.reshape((3, 3, 1, 1)))
    w_filter_2 = np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], 'float32') * conductivity_2 / 3.
    w_filter_2 = tf.constant(w_filter_2.reshape((3, 3, 1, 1)))
    res = masked_conv_v1(u, w_filter_1, w_filter_2, mask_1, mask_2, conductivity_1, conductivity_2)

    return res, f

def test_mask_conv_v2(data_case):
    u, f, mask_1, mask_2, conductivity_1, conductivity_2 = load_data(data_case)

    w_filter_1 = np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], 'float32') * conductivity_1/ 3.
    w_filter_1 = tf.constant(w_filter_1.reshape((3, 3, 1, 1)))
    w_filter_2 = np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], 'float32') * conductivity_2/ 3.
    w_filter_2 = tf.constant(w_filter_2.reshape((3, 3, 1, 1)))

    res = masked_conv_v2(u, w_filter_1, w_filter_2, mask_1, mask_2, conductivity_1, conductivity_2)
    return res, f


def test_mask_conv_v3(data_case):
    u, f, mask_1, mask_2, conductivity_1, conductivity_2 = load_data(data_case)

    filter_1 = np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], 'float32') * conductivity_1/ 3.
    w_filter_1 = tf.constant(filter_1.reshape((3, 3, 1, 1)))
    filter_1_side = np.zeros_like(filter_1)
    filter_1_side[0, 1] = filter_1[0,1]
    filter_1_side[1, 0] = filter_1[1,0]
    filter_1_side[1, 2] = filter_1[1,2]
    filter_1_side[2, 1] = filter_1[2,1]
    filter_1_side = tf.constant(filter_1_side.reshape((3, 3, 1, 1)))
    filter_1_corner = np.zeros_like(filter_1)
    filter_1_corner[0, 0] = filter_1[0,1]
    filter_1_corner[0, 2] = filter_1[1,0]
    filter_1_corner[2, 0] = filter_1[1,2]
    filter_1_corner[2, 2] = filter_1[2,1]
    filter_1_corner = tf.constant(filter_1_corner.reshape((3, 3, 1, 1)))
    filter_2 = np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], 'float32') * conductivity_2/ 3.
    w_filter_2 = tf.constant(filter_2.reshape((3, 3, 1, 1)))
    filter_2_side = np.zeros_like(filter_2)
    filter_2_side[0, 1] = filter_2[0,1]
    filter_2_side[1, 0] = filter_2[1,0]
    filter_2_side[1, 2] = filter_2[1,2]
    filter_2_side[2, 1] = filter_2[2,1]
    filter_2_side = tf.constant(filter_2_side.reshape((3, 3, 1, 1)))
    filter_2_corner = np.zeros_like(filter_2)
    filter_2_corner[0, 0] = filter_2[0,1]
    filter_2_corner[0, 2] = filter_2[1,0]
    filter_2_corner[2, 0] = filter_2[1,2]
    filter_2_corner[2, 2] = filter_2[2,1]
    filter_2_corner = tf.constant(filter_2_corner.reshape((3, 3, 1, 1)))

    filter_banks = {
        'filter_1_corner': filter_1_corner,
        'filter_1_side': filter_1_side,
        'filter_2_corner': filter_2_corner,
        'filter_2_side': filter_2_side,
    }
    res = masked_conv_v3(u, w_filter_1, w_filter_2, mask_1, mask_2, conductivity_1, conductivity_2, filter_banks=filter_banks)
    return res, f


if __name__ == '__main__':
    import scipy.io as sio
    import matplotlib.pyplot as plt

    res, f = test_mask_conv_v3(data_case=-1)

    ## training starts ###
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    img = sess.run(res)
    # visualize convolution result
    plt.figure()
    img = sess.run(res)['result'][0, 1:-1, 1:-1, 0]
    img[:,31] = 0
    img[:,33] = 0
    plt.imshow(img, cmap='jet')
    plt.colorbar()
    plt.grid('off')
    # visualize ground truth
    plt.figure()
    plt.imshow(f[1:-1, 1:-1], cmap='jet')
    plt.colorbar()
    plt.grid('off')
    plt.show()


