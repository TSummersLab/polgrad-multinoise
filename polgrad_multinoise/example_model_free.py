import numpy as np
from matrixmath import randn

from ltimultgen import gen_system_mult
from policygradient import PolicyGradientOptions, run_policy_gradient
from ltimult import dlyap_obj

from matplotlib import pyplot as plt

from time import time
import os
from pickle_io import pickle_import,pickle_export


def set_initial_gains(SS,K0_method):
    # Initial gains
    if K0_method=='random':
        K0 = randn(*SS.Kare.shape)
        SS.setK(K0)
        while not SS.c < np.inf:
            K0 = randn(*SS.Kare.shape) # May take forever
            SS.setK(K0)
    elif K0_method=='random_olmss':
        K0 = randn(*SS.Kare.shape)
        SS.setK(K0)
        while not SS.c < np.inf:
            K0 = 0.9*K0 # Only guaranteed to work if sys open loop mean-square stable
            SS.setK(K0)
    elif K0_method=='are':
        K0 = SS.Kare
        print('Initializing at the ARE solution')
    elif K0_method=='are_perturbed':
        perturb_scale = 10.0/SS.n  # scale with 1/n as rough heuristic
        safety_scale = 2.0
        Kp = randn(SS.Kare.shape[0],SS.Kare.shape[1])
        K0 = SS.Kare + perturb_scale*Kp
        SS.setK(K0)
        while not SS.c < safety_scale*SS.ccare:
            perturb_scale *= 0.2
            K0 = SS.Kare + perturb_scale*Kp
            SS.setK(K0)
        while not SS.c > safety_scale*SS.ccare:
            perturb_scale *= 1.05
            K0 = SS.Kare + perturb_scale*Kp
            SS.setK(K0)
        perturb_scale /= 1.05
        K0 = SS.Kare + perturb_scale*Kp
        SS.setK(K0)
    elif K0_method=='zero':
        K0 = np.zeros([SS.m,SS.n])
        SS.setK(K0)
        P = dlyap_obj(SS,algo='iterative',show_warn=False)
        if P is not None:
            print('Initializing with zero gain solution')
        else:
            print('System not open-loop mean-square stable, use a different initial gain setting')
    elif K0_method=='user':
            K0[K1_subs] = 7.2
            K0[K2_subs] = -0.37
    SS.setK(K0)
    return K0


def policy_gradient_setup(SS):
    # Gradient descent options

    # Convergence threshold
    # Scale by number of gain entries as rough heuristic
    # Set low to force max number of iterations
    epsilon = (1e-6)*SS.Kare.size
    eta = 1e-2
    max_iters = 200
    stepsize_method = 'constant'
    step_direction = 'gradient'

    PGO = PolicyGradientOptions(epsilon=epsilon,
                                eta=eta,
                                max_iters=max_iters,
                                disp_stride=1,
                                keep_hist=True,
                                opt_method='gradient',
                                keep_opt='last',
                                step_direction=step_direction,
                                stepsize_method=stepsize_method,
                                exact=False,
                                regularizer=None,
                                regweight=0,
                                stop_crit='gradient',
                                fbest_repeat_max=0,
                                display_output=True,
                                display_inplace=True,
                                slow=False)
    return PGO


def load_system(folderstr,timestr):
    # Import
    dirname_in = os.path.join(folderstr,timestr)
    filename_only = 'system_init.pickle'
    SS = pickle_import(os.path.join(dirname_in,filename_only))
    #Export
    timestr = str(time()).replace('.','p')
    dirname_out = os.path.join('systems',timestr)
    SS.dirname = dirname_out
    filename_out = os.path.join(dirname_out,filename_only)
    pickle_export(dirname_out, filename_out, SS)
    return SS



###############################################################################
if __name__ == "__main__":

    SS = gen_system_mult(n=2,
                         m=1,
                         safety_margin=0.3,
                         noise='olmss_weak',
                         mult_noise_method='random',
                         SStype='random')

    # Policy gradient setup
    t_start = time()
#    K0_method = 'are_perturbed'
    K0_method = 'zero'
    set_initial_gains(SS,K0_method=K0_method)
    PGO = policy_gradient_setup(SS)
    filename_out = 'policy_gradient_options.pickle'
    path_out = os.path.join(SS.dirname,filename_out)
    pickle_export(SS.dirname,path_out,PGO)
    t_end = time()
    print('Initialization completed after %.3f seconds' % (t_end-t_start))

    run_policy_gradient(SS,PGO)