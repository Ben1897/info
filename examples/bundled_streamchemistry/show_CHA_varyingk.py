import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pickle
import sys

from info.utils import tigramite_plotting as tp
from info.utils.causal_network import causal_network
# import info.utils.causal_network.search_bundled_components as sbc
# from info.core.info_network import compute_bundled as cb

# Plotting settings
rc('text', usetex=True)
small_size = 15
medium_size = 25
bigger_size = 30
plt.rc('font', size=small_size)          # controls default text sizes
plt.rc('axes', titlesize=small_size)    # fontsize of the axes title
plt.rc('axes', labelsize=small_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=small_size)    # legend fontsize
plt.rc('figure', titlesize=small_size)  # fontsize of the figure title

# Parameters
levels = [0,1]
knns   = [5,6,7,8,9,10,15]
filename = lambda level,knn:'./bundled_info_ph_level' + str(level)  + '_k' + str(knn) + '_taumax5.npy'
parentsfile = 'parents_7hr_instream_strict_7var_raw_anomaly_taumax5.pkl'
fig_folder = "./"
taustart, tauend = 5, 150

pidlabelst = ['$S^{\mathcal{T}}$', '$R^{\mathcal{T}}$',
             '$U^{\mathcal{T}}_m$', '$U^{\mathcal{T}}_n$']
pidlabelsi = ['$S^{\mathcal{J}}$', '$R^{\mathcal{J}}$',
              '$U^{\mathcal{J}}_m$', '$U^{\mathcal{J}}_n$']
pidlabelsd = ['$S^{\mathcal{D}}$', '$R^{\mathcal{D}}$',
              '$U^{\mathcal{D}}_m$', '$U^{\mathcal{D}}_n$']
colors = ['orange', 'dodgerblue', 'cyan', 'darkgrey']
colors1 = ['yellow', 'lawngreen', 'violet', 'lightgrey']
# xlabel, ylabel = r'$\tau$', '$[nats]$'
ylabel = lambda k: '$k=' + str(k) + '$ \n $[nats]$'

# Plot the PID for J and D
fig, axes = plt.subplots(nrows=len(knns), ncols=len(levels), figsize=(8, 10), sharex=True)
for i in range(len(knns)):
    ## LEVEL 0
    ax = axes[i,0]
    result = np.load(filename(0,knns[i]))[()]
    infoset = result['info'][parentsfile]
    pidset = result['pid'][parentsfile]
    start, end = result['taurange']
    taurange = range(start, end)
    citset, pitset, titset = infoset[:, 0], infoset[:, 1], infoset[:, 2]
    citpid, pitpid, titpid = pidset[:, :4], pidset[:, 4:8], pidset[:, 8:]
    base = np.zeros(len(taurange))[taustart:tauend]
    for j in [2,1,3,0]:
        ax.fill_between(taurange[taustart:tauend], base,
                        base+citpid[taustart:tauend,j],
                        color=colors[j], label=pidlabelsi[j])
        base += citpid[taustart:tauend,j]
    ax.plot(taurange[taustart:tauend], base, 'k--', linewidth=1)
    for j in [2,1,3,0]:
        ax.fill_between(taurange[taustart:tauend], base,
                        base+pitpid[taustart:tauend,j],
                        color=colors1[j], label=pidlabelsd[j])
        base += pitpid[taustart:tauend,j]
    ax.set_xticks([5,50,100,150])
    # ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel(knns[i]))
    ax.set_ylim(bottom=-0.1)

    ## LEVEL 1
    ax = axes[i,1]
    result = np.load(filename(1,knns[i]))[()]
    infoset = result['info'][parentsfile]
    pidset = result['pid'][parentsfile]
    start, end = result['taurange']
    taurange = range(start, end)
    citset, pitset, titset = infoset[:, 0], infoset[:, 1], infoset[:, 2]
    citpid, pitpid, titpid = pidset[:, :4], pidset[:, 4:8], pidset[:, 8:]
    base = np.zeros(len(taurange))[taustart:tauend]
    for j in [2,1,3,0]:
        ax.fill_between(taurange[taustart:tauend], base,
                        base+citpid[taustart:tauend,j],
                        color=colors[j], label=pidlabelsi[j])
        base += citpid[taustart:tauend,j]
    ax.plot(taurange[taustart:tauend], base, 'k--', linewidth=1)
    for j in [2,1,3,0]:
        ax.fill_between(taurange[taustart:tauend], base,
                        base+pitpid[taustart:tauend,j],
                        color=colors1[j], label=pidlabelsd[j])
        base += pitpid[taustart:tauend,j]
    ax.set_xticks([5,50,100,150])
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=-0.01)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='lower center', ncol=4,
          facecolor='white', framealpha=0.7,
          bbox_to_anchor=(0.,-1.25),
          fontsize=15,frameon=True)
# plt.savefig(fig_folder + 'BCHA_stream_varyingk.eps', transparent=True, format='eps',bbox_inches='tight')
plt.savefig(fig_folder + 'BCHA_stream_varyingk_legend.eps', transparent=True, format='eps',bbox_inches='tight')
# plt.show()
