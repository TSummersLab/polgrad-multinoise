import numpy as np
from numpy import linalg as la
from matrixmath import randn,vec,mdot,sympart,is_pos_def

from ltimultgen import gen_system_mult,gen_system_example_suspension
from policygradient import PolicyGradientOptions

from policygradient import run_policy_gradient


from ltimult import dlyap_obj, dlyap_mult

from plotting import plot_traj, plot_PGO_results
from matplotlib import pyplot as plt

from time import time,sleep
from copy import copy

import os
from utility import create_directory

from pickle_io import pickle_import,pickle_export


def plot_sample_traj(SS):
    # Sample plots
    nt = 20
    nr = 10
    Xall_ol = np.zeros([SS.n,nt,nr])
    Xall_cl = np.zeros([SS.n,nt,nr])

    # Simulate trajectories
    for i in range(nr):
        x0 = np.ones(SS.n)
        Xall_ol[:,:,i] = SS.sim_ol(x0,nt)
        Xall_cl[:,:,i] = SS.sim_cl(x0,nt)

    # Plot trajectories
    plot_type='split'
    plot_traj(Xall_ol,plot_type=plot_type)
    plot_traj(Xall_cl,plot_type=plot_type)

def check_olmss(SS):
    # Check if open-loop mss
    K00 = np.zeros([SS.m,SS.n])
    SS00 = copy(SS)
    SS00.setK(K00)
    P = dlyap_obj(SS00,algo='iterative',show_warn=False)
    if P is None:
        print('System is NOT open-loop mean-square stable')
    else:
        print('System is open-loop mean-square stable')


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
        safety_scale = 10.0
        Kp = randn(SS.Kare.shape[0],SS.Kare.shape[1])
        K0 = SS.Kare + perturb_scale*Kp
        SS.setK(K0)
        while not SS.c < safety_scale*SS.ccare:
            perturb_scale *= 0.2
            K0 = SS.Kare + perturb_scale*Kp
            SS.setK(K0)
        while not SS.c > safety_scale*SS.ccare:
            perturb_scale *= 1.01
            K0 = SS.Kare + perturb_scale*Kp
            SS.setK(K0)
        perturb_scale /= 1.01
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
    SS.setK(K0)
    return K0


def policy_gradient_setup(SS):
    # Gradient descent options

    # Convergence threshold
    epsilon = (1e+2)*SS.Kare.size # Scale by number of gain entries as rough heuristic
    eta = 1e-8
    max_iters = 100000
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
                                exact=True,
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

def calc_comparison_costs(SS1,SS2,histlist1,histlist2):

    nt_on_SS1 = histlist1[0].shape[2]
    nt_on_SS2 = histlist2[0].shape[2]

    c1hist_on_SS1 = np.zeros(nt_on_SS1)
    c2hist_on_SS1 = np.zeros(nt_on_SS1)
    c1hist_on_SS2 = np.zeros(nt_on_SS2)
    c2hist_on_SS2 = np.zeros(nt_on_SS2)

    P1 = np.zeros([SS1.n,SS1.n])
    S1 = np.zeros([SS1.n,SS1.n])
    P2 = np.zeros([SS2.n,SS2.n])
    S2 = np.zeros([SS2.n,SS2.n])

    for i in range(nt_on_SS1):
        K = histlist1[0][:,:,i]
        SS1.setK(K)
        SS2.setK(K)
        SS1.calc_PS(P1,S1)
        SS2.calc_PS(P2,S2)
        P1 = SS1.P
        S1 = SS1.S
        P2 = SS2.P
        S2 = SS2.S
        print(i)
        c1hist_on_SS1[i] = SS1.c
        c2hist_on_SS1[i] = SS2.c


    P1 = np.zeros([SS1.n,SS1.n])
    S1 = np.zeros([SS1.n,SS1.n])
    P2 = np.zeros([SS2.n,SS2.n])
    S2 = np.zeros([SS2.n,SS2.n])


    for i in range(nt_on_SS2):
        K = histlist2[0][:,:,i]
        if i > 0:
            if c1hist_on_SS2[i-1] < np.inf:
                SS1.setK(K)
                SS1.calc_PS(P1,S1)
                P1 = SS1.P
                S1 = SS1.S
                c1hist_on_SS2[i] = SS1.c
            else:
                c1hist_on_SS2[i] = np.inf
        else:
            SS1.setK(K)
            SS1.calc_PS(P1,S1)
            P1 = SS1.P
            S1 = SS1.S
            c1hist_on_SS2[i] = SS1.c

        SS2.setK(K)
        SS2.calc_PS(P2,S2)
        P2 = SS2.P
        S2 = SS2.S
        c2hist_on_SS2[i] = SS2.c

        print(i)

    return [c1hist_on_SS1,c2hist_on_SS1,c1hist_on_SS2,c2hist_on_SS2]


def plot_results(SS1,SS2,chist_data,dirname_in):

    img_dirname_out = os.path.join(dirname_in,'analysis_plots')
    create_directory(img_dirname_out)

    c1hist_on_SS1 = np.copy(chist_data[0])
    c2hist_on_SS1 = np.copy(chist_data[1])
    c1hist_on_SS2 = np.copy(chist_data[2])
    c2hist_on_SS2 = np.copy(chist_data[3])

    ccare1 = SS1.ccare
    ccare2 = SS2.ccare

    # Normalize and shift
    c1hist_on_SS1 /= ccare1
    c2hist_on_SS1 /= ccare2
    c1hist_on_SS2 /= ccare1
    c2hist_on_SS2 /= ccare2

    c1hist_on_SS1 -= 1
    c2hist_on_SS1 -= 1
    c1hist_on_SS2 -= 1
    c2hist_on_SS2 -= 1

    plotfun = plt.semilogy


    marker1 = 'o'
    marker2 = 'v'

    ms = 8

    color1 = 'tab:blue'
    color2 = 'tab:red'

    lw1 = 3
    lw2 = 3
    fs = 14

    fig1 = plt.figure(figsize=(4.5,2.5))
    plotfun(c1hist_on_SS1,color=color1,linewidth=lw1,marker=marker1,markersize=ms,markevery=500)
    plotfun(c2hist_on_SS1,color=color2,linewidth=lw2,marker=marker2,markersize=ms,markevery=500)
    plt.xlabel('Iteration',fontsize=fs)
    plt.ylabel('Normalized cost diff.',fontsize=fs)
    plt.legend(['LQRm cost (with noise)','LQR cost (no noise)'])
    filename_out = 'plot_lqrm_cost_vs_iteration_suspension'
    path_out = os.path.join(img_dirname_out,filename_out)
    plt.savefig(path_out,dpi=300,bbox_inches='tight')

    fig2 = plt.figure(figsize=(4.5,2.5))
    plotfun(c1hist_on_SS2,color=color1,linewidth=lw1,marker=marker1,markersize=ms,markevery=1200)
    plotfun(c2hist_on_SS2,color=color2,linewidth=lw2,marker=marker2,markersize=ms,markevery=1200)
    plt.xlabel('Iteration',fontsize=fs)
    plt.ylabel('Normalized cost diff.',fontsize=fs)
    plt.legend(['LQRm cost (with noise)','LQR cost (no noise)'],loc='right')
    filename_out = 'plot_lqr_cost_vs_iteration_suspension'
    path_out = os.path.join(img_dirname_out,filename_out)
    plt.savefig(path_out,dpi=300,bbox_inches='tight')



def routine_gen():
#    SS = gen_system_example_suspension()

    timestr = '1558459899p686552_example_suspension_model_known'
    folderstr = 'systems_keepers'
    SS1 = load_system(folderstr,timestr)

    check_olmss(SS1)

    # Policy gradient setup
    t_start = time()
    K0_method = 'are_perturbed'
    K0 = set_initial_gains(SS1,K0_method=K0_method)
    PGO = policy_gradient_setup(SS1)
    filename_out = 'policy_gradient_options.pickle'
    path_out = os.path.join(SS1.dirname,filename_out)
    pickle_export(SS1.dirname,path_out,PGO)
    t_end = time()
    print('Initialization completed after %.3f seconds' % (t_end-t_start))

    SS1,histlist1 = run_policy_gradient(SS1,PGO)


    # Find optimal control ignoring noise
    SS2 = copy(SS1)
    SS2.set_a(np.zeros_like(SS2.a))
    SS2.set_b(np.zeros_like(SS2.b))
    SS2.setK(K0)

    PGO.eta = 1e-5
    PGO.epsilon = (1e-1)*SS2.Kare.size
    SS2,histlist2 = run_policy_gradient(SS2,PGO)

#    SS1.setK(SS2.Kare)
#    SS1.setK(SS2.K)
#
#    dirname_in = os.path.join(folderstr,timestr)

#    chist_data = calc_comparison_costs(SS1,SS2,histlist1,histlist2)
#    dirname_out = copy(dirname_in)
#    filename_out = 'chist_data.pickle'
#    path_out = os.path.join(dirname_out,filename_out)
#    pickle_export(dirname_out,path_out,chist_data)
#
#    plot_results(SS1,SS2,chist_data,dirname_in)


def routine_load():
    timestr = '1558459899p686552_example_suspension_model_known'
    folderstr = 'systems_keepers'
    dirname_in = os.path.join(folderstr,timestr)

    SS1 = load_system(folderstr,timestr)
    SS2 = copy(SS1)
    SS2.set_a(np.zeros_like(SS2.a))
    SS2.set_b(np.zeros_like(SS2.b))

    filename_in = os.path.join(dirname_in,'chist_data.pickle')
    chist_data = pickle_import(filename_in)
    plot_results(SS1,SS2,chist_data,dirname_in,)




###############################################################################
if __name__ == "__main__":
    routine_gen()
#    routine_load()