
import os
import json
import logging
from datetime import datetime
import dateutil.tz

def creat_dir(network_type):
    """code from on InfoGAN"""
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    root_log_dir = "./saved_logs/" + network_type
    exp_name = network_type + "_%s" % timestamp
    log_dir = os.path.join(root_log_dir, exp_name)

    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    root_model_dir = "./saved_models/" + network_type
    exp_name = network_type + "_%s" % timestamp
    model_dir = os.path.join(root_model_dir, exp_name)

    for path in [log_dir, model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
    return log_dir, model_dir




def church_plt(pred_res, axis_range, fn=None):
    fn = 'church_plot_s{}_g{}.png'.format(cfg['range'], cfg['gpu_idx'])
    from pylab import *
    import matplotlib.gridspec as gridspec
    import numpy as np
    import matplotlib.pylab as pylab

    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (14, 5),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    pylab.rcParams.update(params)

    axis_range = [-32, 32, -32, 32]
    aa = pred_res
    x_ticks_var = ['{}'.format(axis_range[0]), '0', '{}'.format(axis_range[1])]
    y_ticks_var = ['{}'.format(axis_range[2]), '0', '{}'.format(axis_range[3])]
    mat_size = 192
    x_ticks_loc = [0, mat_size / 2, mat_size]
    y_ticks_loc = [0, mat_size / 2, mat_size]

    fig, axes = plt.subplots(nrows=1, ncols=2)
    cmap = plt.get_cmap('jet', 10)

    ax = axes[0]
    pred_matrix = aa['lenet']
    im = ax.imshow(pred_matrix, cmap=cmap, vmin=0, vmax=9)
    ax.set_title('lenet')
    plt.sca(ax)
    plt.xticks(x_ticks_loc, x_ticks_var, color='black')
    plt.yticks(y_ticks_loc, y_ticks_var, color='black')
    # Minor ticks
    ax.set_xticks([mat_size / 2], minor=True)
    ax.set_yticks([mat_size / 2], minor=True)
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.7)

    ax = axes[1]
    pred_matrix = aa['binarize']
    im = ax.imshow(pred_matrix, cmap=cmap, vmin=0, vmax=9)
    ax.set_title('binarize')
    plt.sca(ax)
    plt.xticks(x_ticks_loc, x_ticks_var, color='black')
    plt.yticks(y_ticks_loc, y_ticks_var, color='black')
    # Minor ticks
    ax.set_xticks([mat_size / 2], minor=True)
    ax.set_yticks([mat_size / 2], minor=True)
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.7)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

    plt.savefig(fn)

def get_animation_heat():
    import matplotlib.pyplot as plt
    import numpy as np
    data_dir = './data/heat_transfer_2phase'
    u_hist = np.load(data_dir+'/jacobi_net_sol.npy')
    import imageio
    images = []
    for i in range(0, 10000, 100):
        plt.figure()
        plt.imshow(u_hist[i], cmap='jet')
        plt.colorbar()
        plt.grid('off')
        filename = data_dir+'/itr_{}'.format(i)
        plt.title(filename)
        plt.savefig(filename)
        images.append(imageio.imread(filename+'.png'))
    imageio.mimsave(data_dir+'/jacobi.gif', images)
    plt.close('all')

def get_animation_elast():
    import matplotlib.pyplot as plt
    import scipy.io as sio
    ux = sio.loadmat('./data/linear_elast_2phase/displacement_x_Steel_Al.mat')['u1']
    uy = sio.loadmat('./data/linear_elast_2phase/displacement_y_Steel_Al.mat')['u1']
    plt.figure()
    plt.imshow(ux)
    plt.figure()
    plt.imshow(uy)

# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.semilogy(test_loss_hist)
# plt.show()
#
# # plt.imshow(sess.run(jacobi_result['u_hist'][-1], feed_dict_train)[0,:,:,0], cmap="hot")
# plt.imshow(y_input[0,:,:,0], cmap="hot")
# plt.colorbar()
# plt.show()
