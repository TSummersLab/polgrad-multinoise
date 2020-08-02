import numpy as np
import numpy.linalg as la
import numpy.random as npr
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
from pickle_io import pickle_import,pickle_export

folderstr_list = []
folderstr_list.append("1564554983p5677059_1e4")
folderstr_list.append("1564555001p5515425_1e5")
folderstr_list.append("1564555047p6032026_1e6")
folderstr_list.append("1564555255p6612067_1e7")
folderstr_list.append("1564525514p9662921_1e8")

nr_list = [1e4,1e5,1e6,1e7,1e8]
N = len(nr_list)
data_noiseless = []
data_noisy = []
for i,folderstr in enumerate(folderstr_list):
    dirname_in = folderstr

    filename = 'data_noiseless.pickle'
    filename_in = os.path.join(dirname_in,filename)
    data_noiseless.append(pickle_import(filename_in))

    filename = 'data_noisy.pickle'
    filename_in = os.path.join(dirname_in,filename)
    data_noisy.append(pickle_import(filename_in))

# Plotting
mean_error_norm_noiseless = np.zeros(N)
mean_error_norm_noisy = np.zeros(N)

mean_error_angle_noiseless = np.zeros(N)
mean_error_angle_noisy = np.zeros(N)

for i in range(N):
    mean_error_norm_noiseless[i] = np.mean(data_noiseless[i][4])/la.norm(data_noiseless[0][0])
    mean_error_norm_noisy[i] = np.mean(data_noisy[i][4])/la.norm(data_noisy[0][0])
    mean_error_angle_noiseless[i] = np.mean(data_noiseless[i][2])
    mean_error_angle_noisy[i] = np.mean(data_noisy[i][2])

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.figure(figsize=(4,2.2))
#plt.figure(figsize=(7,5))
plt.semilogx(nr_list,mean_error_norm_noiseless,linewidth=4,marker='^',markersize=8,color='tab:blue')
plt.semilogx(nr_list,mean_error_norm_noisy,linewidth=4,marker='o',markersize=8,color='tab:red')
guide_color = 'tab:grey'
plt.semilogx(nr_list,0.1*np.ones(N),color=guide_color,linestyle='--')
#plt.axvline(5*10**5,ymax=0.15,color=guide_color,linestyle='--')
#plt.axvline(10**8,ymax=0.15,color=guide_color,linestyle='--')
plt.yticks(ticks=[0,0.1,0.25,0.50,0.75])
plt.xlabel('Number of rollouts')
plt.ylabel('Normalized gradient estimate error')
plt.ylabel(r'$\|\nabla C(K)-\widehat{\nabla} C_K \|/\|\nabla C(K)\|$')
plt.legend(["Noiseless","Noisy"])
plt.tight_layout()
plt.savefig("plot_gradient_estimation_error.png",dpi=300)
#plt.savefig("fig1alt.png",dpi=300)

#plt.figure()
#plt.semilogx(nr_list,mean_error_angle_noiseless,linewidth=4)
#plt.semilogx(nr_list,mean_error_angle_noisy,linewidth=4)
#plt.xlabel('Number of rollouts')
#plt.ylabel('Gradient estimate error angle (deg)')
#plt.legend(["Noiseless","Noisy"])