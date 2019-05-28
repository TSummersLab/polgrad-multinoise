import numpy as np
from numpy import linalg as la
from matrixmath import randn,vec

from ltimultgen import gen_system_mult,gen_system_mult_example
from policygradient import PolicyGradientOptions, run_policy_gradient, Regularizer
from ltimult import dlyap_mult

from plotting import plot_traj, plot_PGO_results
from matplotlib import pyplot as plt
from costsurf import CostSurfaceOptions, plot_cost_surf

from time import time
from copy import copy


import dill


glob_filename = 'globalsave.pkl'
# load the dump  session again
dill.load_session(glob_filename)


# Sparsity visualizations
Kmax = np.max(np.abs(vec(SS.K)))
bin1 = 0.01*Kmax    
bins = np.hstack([np.array([0,bin1]),np.linspace(bin1,Kmax,10)])    
sparsity_required = 0.5
sparsity = np.sum(np.abs(SS.K)<bin1)/SS.K.size    
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches((12, 4))
img = ax[0].imshow(np.abs(SS.K))
fig.colorbar(img,ax=ax[0])
ax[0].set_title('Gain matrix entry abs values')
ax[1].hist(vec(np.abs(SS.K)),bins=bins,rwidth=0.8)  
ax[1].set_ylim(0,SS.K.size)
titlestr = 'Regularizer weight = %.2f, Sparsity = %.2f%%,\nClosed-loop LQR cost = %.2f' % (regweight,100*sparsity,SS.c)
ax[1].set_title(titlestr)
#    plt.ion()
plt.show() 