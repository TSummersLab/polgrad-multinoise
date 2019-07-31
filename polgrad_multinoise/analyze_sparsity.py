import numpy as np
from numpy import linalg as la
from matrixmath import randn,vec

from ltimultgen import gen_system_mult
from policygradient import PolicyGradientOptions, run_policy_gradient, Regularizer
from ltimult import dlyap_obj, dlyap_mult

from plotting import plot_traj, plot_PGO_results
from matplotlib import pyplot as plt
from costsurf import CostSurfaceOptions, plot_cost_surf

from time import time,sleep
from copy import copy

import os
from utility import create_directory

from pickle_io import pickle_import,pickle_export


def plot_sparsity_data(SS0,sparsity_data,dirname_in,fig_list=[],ax_list=[]):

    rws = []
    spars = []
    Ks = []
    cs = []

    ccare = SS0.ccare

    for s in sparsity_data:
        rws.append(s[0])
        spars.append(s[1])
        Ks.append(s[2])
        cs.append(s[3])

    fs = 16

    fig_idx_list = range(2)
    img_dirname_out = os.path.join(dirname_in,'analysis_plots')
    create_directory(img_dirname_out)

    for fig_idx in fig_idx_list:
        fig,ax = plt.subplots()
        if not fig_list:
            fig_list.append(fig)
        else:
            fig_list[fig_idx] = fig
        if fig_idx==0:
            plt.plot(rws,spars,'o-')
            plt.xlabel('Regularization weight',fontsize=fs)
            plt.ylabel('Sparsity',fontsize=fs)
            filename_out = 'plot_sparsity_vs_regweight'
        elif fig_idx==1:
            plt.plot(spars,cs,'o-')
            plt.plot(spars,np.ones_like(spars)*ccare,'r--')
            plt.xlabel('Sparsity',fontsize=fs)
            plt.ylabel('LQR cost',fontsize=fs)
            filename_out = 'plot_lqr_cost_vs_sparsity'
        path_out = os.path.join(img_dirname_out,filename_out)
        plt.savefig(path_out,dpi=300,bbox_inches='tight')

    return fig_list, ax_list


def analyze_traverse_sparsity(dirname_in):
    filename_in = 'sparsity_data.pickle'
    path_in = os.path.join(dirname_in,filename_in)
    sparsity_data = pickle_import(path_in)

    filename_in = 'system_init.pickle'
    path_in = os.path.join(dirname_in,filename_in)
    SS0 = pickle_import(path_in)

    plot_sparsity_data(SS0,sparsity_data,dirname_in)


def plot_comparisons(SS0,sparsity_data_all,dirname_in,fig_list=[],ax_list=[]):
    img_dirname_out = os.path.join(dirname_in,'comparison_plots')
    create_directory(img_dirname_out)

    nfigs = 5

    fig_idx_list = range(nfigs)
    ax_idx_list = range(nfigs)

    if not fig_list:
        fig_list = [None] * len(fig_idx_list)
    if not ax_list:
        ax_list = [None] * len(ax_idx_list)

    legend_strs = []
    for key in sparsity_data_all.keys():
        legend_strs.append(key.replace('_',' ').capitalize())

    marker_list = ['o','v','x']
    fs = 14
    fsleg = 12
    xlab_xoffs = 0.50
    xlab_yoffs = -0.05

    for fig_idx in fig_idx_list:
#        fig,ax = plt.subplots(figsize=[6,3])
        fig,ax = plt.subplots(figsize=[6,6])
        fig_list[fig_idx] = fig
        ax_list[fig_idx] = ax
        if fig_idx == 0:
            for i,optiongroup in enumerate(sparsity_data_all.keys()):
                nr = len(sparsity_data_all[optiongroup])
                regweights = []
                walltimes = []
                for r in range(nr):
                    regweights.append(sparsity_data_all[optiongroup][r][0])
                    walltimes.append(sparsity_data_all[optiongroup][r][4])
                plt.semilogx(regweights,walltimes,marker=marker_list[i])
            plt.xlabel('Regularization weight',fontsize=fs)
            ax.xaxis.set_label_coords(xlab_xoffs, xlab_yoffs)
            plt.ylabel('Compute time (s)',fontsize=fs)
            plt.legend(legend_strs,prop={'size': fsleg})
            filename_out = 'plot_comparison_walltime_vs_regweight'

        if fig_idx == 1:
            for i,optiongroup in enumerate(sparsity_data_all.keys()):
                nr = len(sparsity_data_all[optiongroup])
                regweights = []
                itercounts = []
                for r in range(nr):
                    regweights.append(sparsity_data_all[optiongroup][r][0])
                    itercounts.append(len(sparsity_data_all[optiongroup][r][5][2]))
                plt.semilogx(regweights,itercounts,marker=marker_list[i])
            plt.xlabel('Regularization weight',fontsize=fs)
            plt.ylabel('Number of iterations',fontsize=fs)
            ax.xaxis.set_label_coords(xlab_xoffs, xlab_yoffs)
            plt.legend(legend_strs,prop={'size': fsleg})
            filename_out = 'plot_comparison_itercount_vs_regweight'

        if fig_idx == 2:
            for i,optiongroup in enumerate(sparsity_data_all.keys()):
                nr = len(sparsity_data_all[optiongroup])
                regweights = []
                objvals = []
                for r in range(nr):
                    regweights.append(sparsity_data_all[optiongroup][r][0])
                    objvals.append(sparsity_data_all[optiongroup][r][5][3][-1])
                plt.semilogx(regweights,objvals,marker=marker_list[i])
            plt.xlabel('Regularization weight',fontsize=fs)
            plt.ylabel('Objective value',fontsize=fs)
            ax.xaxis.set_label_coords(xlab_xoffs, xlab_yoffs)
            plt.legend(legend_strs,prop={'size': fsleg})
            filename_out = 'plot_comparison_objval_vs_regweight'

        if fig_idx == 3:
            for i,optiongroup in enumerate(sparsity_data_all.keys()):
                nr = len(sparsity_data_all[optiongroup])
                regweights = []
                sps = []
                for r in range(nr):
                    regweights.append(sparsity_data_all[optiongroup][r][0])
                    sps.append(sparsity_data_all[optiongroup][r][1])
                plt.semilogx(regweights,sps,marker=marker_list[i])
            plt.xlabel('Regularization weight',fontsize=fs)
            plt.ylabel('Sparsity',fontsize=fs)
            ax.xaxis.set_label_coords(xlab_xoffs, xlab_yoffs)
            plt.legend(legend_strs,prop={'size': fsleg})
            filename_out = 'plot_comparison_sparsity_vs_regweight'


        if fig_idx == 4:
            YSCALE = 1
            spsmin = 1
            spsmax = 0
            for i,optiongroup in enumerate(sparsity_data_all.keys()):
                nr = len(sparsity_data_all[optiongroup])
                sps = []
                cs = []
                for r in range(nr):
                    sps.append(sparsity_data_all[optiongroup][r][1])
                    cs.append(YSCALE*sparsity_data_all[optiongroup][r][3])
                plt.plot(sps,cs,marker=marker_list[i])
                spsmin = np.min([spsmin,np.min(sps)])
                spsmax = np.max([spsmax,np.max(sps)])
            plt.plot([spsmin,spsmax],[YSCALE*SS0.ccare,YSCALE*SS0.ccare],'r--')
            plt.xlabel('Sparsity',fontsize=fs)
            plt.ylabel('LQRm cost',fontsize=fs)
            ax.xaxis.set_label_coords(xlab_xoffs, xlab_yoffs)
            legend_strs.append('Zero regularization')
            plt.legend(legend_strs,prop={'size': fsleg})
            filename_out = 'plot_comparison_LQR_cost_vs_sparsity'


        path_out = os.path.join(img_dirname_out,filename_out)
        plt.savefig(path_out,dpi=300,bbox_inches='tight')



    return fig_list, ax_list


def compare_optmethods(dirname_in):

    filename_in = 'system_init.pickle'
    path_in = os.path.join(dirname_in,filename_in)
    SS0 = pickle_import(path_in)


#    optiongroup_list = ['gradient','subgradient','proximal_gradient']
#    optiongroup_list = ['subgradient']
    optiongroup_list = ['proximal_gradient']

    sparsity_data_all = {}

    for optiongroup in optiongroup_list:
        optiongroup_dir = os.path.join(dirname_in,optiongroup)

        filename_in = 'sparsity_data.pickle'
        path_in = os.path.join(optiongroup_dir,filename_in)
        sparsity_data = pickle_import(path_in)

        sparsity_data_all[optiongroup] = sparsity_data

    plot_comparisons(SS0,sparsity_data_all,dirname_in)



###############################################################################
if __name__ == "__main__":



#    timestr = '1556656014p3178775_n50_olmss_vec1'
#    timestr = '1556654503p616922_n50_olmss_glr'
#    timestr = '1556664178p176347_n50_olmsus_vec1'
    timestr = 'last'

    if timestr == 'last':
        dir_list = next(os.walk('systems'))[1]
        timestr = dir_list[-1]


    folderstr = 'systems'
#    folderstr = 'systems_keepers'
    dirname_in = os.path.join(folderstr,timestr)
#    analyze_traverse_sparsity(dirname_in)
    compare_optmethods(dirname_in)