import numpy as np
from numpy import linalg as la

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import os
import subprocess

###############################################################################
# Plotting functions
###############################################################################
def plot_traj(Xall,plot_type='combined'):
    
    n = Xall.shape[0]
    nt = Xall.shape[1]
    nr = Xall.shape[2]
    Ts = np.arange(nt)
    
    if plot_type=='combined':
        fig, ax = plt.subplots()
        for j in range(nr):
            for i in range(n):
                points = np.array([Ts, Xall[i,:,j]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                e = np.abs(Xall[i,:,j])
                
                # Create a continuous norm to map from data points to colors
                norm = plt.Normalize(0, 1.2)
                lc = LineCollection(segments, cmap='viridis', norm=norm,alpha=0.5)
                # Set the values used for colormapping
                lc.set_array(e)
                lc.set_linewidth(2)
                line = ax.add_collection(lc)    
                
                xlim = list(ax.get_xlim())
                ylim = list(ax.get_ylim())    
                xlim[0] = np.min((xlim[0],Ts.min()))
                xlim[1] = np.max((xlim[1],Ts.max()))
                ylim[0] = np.min((ylim[0],Xall[i,:,j].min()))
                ylim[1] = np.max((ylim[1],Xall[i,:,j].max()))  
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_title('All states vs. time')
        fig.colorbar(line, ax=ax)
        
    elif plot_type=='split':
        fig, axs = plt.subplots(nrows=n,ncols=1,sharex=True)
        
        for j in range(nr):
            lines = []
            for i in range(n):
                ax = axs[i]
                points = np.array([Ts, Xall[i,:,j]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                e = np.abs(Xall[i,:,j])
                
                # Create a continuous norm to map from data points to colors
                norm = plt.Normalize(0, 1.2)
                lc = LineCollection(segments, cmap='viridis', norm=norm,alpha=0.5)
                # Set the values used for colormapping
                lc.set_array(e)
                lc.set_linewidth(2)
                lines.append(ax.add_collection(lc))
                
                xlim = list(ax.get_xlim())
                ylim = list(ax.get_ylim())    
                xlim[0] = np.min((xlim[0],Ts.min()))
                xlim[1] = np.max((xlim[1],Ts.max()))
                ylim[0] = np.min((ylim[0],Xall[i,:,j].min()))
                ylim[1] = np.max((ylim[1],Xall[i,:,j].max()))  
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xlabel('Time')
                ax.set_ylabel('State '+str(i))
        #        ax.set_title('State '+str(i)+' vs. time')        
        for i in range(n):
            fig.colorbar(lines[i], ax=axs[i])
        
    elif plot_type=='2d':
        fig, ax = plt.subplots()                
        for j in range(nr):
            points = np.array([Xall[0,:,j], Xall[1,:,j]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
            e = la.norm(Xall[:,:,j],axis=0)
            
            # Create a continuous norm to map from data points to colors
            norm = plt.Normalize(0, 1.2)
            lc = LineCollection(segments, cmap='viridis', norm=norm,alpha=0.5)
            # Set the values used for colormapping
            lc.set_array(e)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)    
            
            xlim = list(ax.get_xlim())
            ylim = list(ax.get_ylim())    
            xlim[0] = np.min((xlim[0],Xall[0,:,j].min()))
            xlim[1] = np.max((xlim[1],Xall[0,:,j].max()))
            ylim[0] = np.min((ylim[0],Xall[1,:,j].min()))
            ylim[1] = np.max((ylim[1],Xall[1,:,j].max()))  
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)    
        fig.colorbar(line, ax=ax)
    plt.show()
    
    
    
def plot_PGO_results(SS,K_hist,grad_hist,c_hist):
    
    nt = c_hist.size
    
    K_err_hist = np.zeros(nt)
    grad_mag_hist = np.zeros(nt)
    c_err_hist = c_hist - SS.ccare
    for t in range(nt):
        K_err_hist[t] = la.norm(SS.Kare - K_hist[:,:,t],'fro')
        grad_mag_hist[t] = la.norm(grad_hist[:,:,t])    
    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(K_err_hist)
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Gain error')
    
    ax[1].plot(grad_mag_hist)
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Norm of cost gradient')
    
    ax[2].plot(c_err_hist)
    ax[2].set_xlabel('Iteration')
    ax[2].set_ylabel('Cost error')
    
    
    
def vidsave(img_folder,img_pattern,filename_out):
    os.chdir(img_folder)
    subprocess.call(['ffmpeg', '-framerate', '8', '-i', img_pattern,
                     '-r', '30', filename_out])    