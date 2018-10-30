import matplotlib.pyplot as plt
import numpy as np

fcn = np.load('FCN.npy')
feanet = np.load('FEA_NET.npy')
feanet6x = np.load('FEA_NET6x.npy')
ref = np.load('REF.npy')

plt.subplot(1, 4, 1)
plt.imshow(fcn, cmap='jet', interpolation='bilinear')
plt.colorbar()
plt.axis('off')
plt.subplot(1, 4, 2)
plt.imshow(feanet, cmap='jet', interpolation='bilinear')
plt.colorbar()
plt.axis('off')
plt.subplot(1, 4, 3)
plt.imshow(feanet6x, cmap='jet', interpolation='bilinear')
plt.colorbar()
plt.axis('off')
plt.subplot(1, 4, 4)
plt.imshow(ref, cmap='jet', interpolation='bilinear')
plt.colorbar()
plt.axis('off')

plt.show()