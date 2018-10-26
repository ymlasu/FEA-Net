import matplotlib.pyplot as plt
import numpy as np




SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def fmg_net_complexity():
    x=np.asarray([64,128,256,512,1024,2048,4096,8192])
    y=np.asarray([16,27,36,49,84,101,120,141])
    plt.semilogx(x*x,y,'o-')
    plt.xlabel('DOF')
    plt.ylabel('# of LU block in FMG-NET')