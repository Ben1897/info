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
small_size = 20
medium_size = 25
bigger_size = 30
plt.rc('font', size=small_size)          # controls default text sizes
plt.rc('axes', titlesize=small_size)    # fontsize of the axes title
plt.rc('axes', labelsize=small_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=small_size)    # legend fontsize
plt.rc('figure', titlesize=small_size)  # fontsize of the figure title

# Function for get variable names in stream chemistry
def get_varnames(varnames):
    varnames2 = []
    for varn in varnames:
        if 'pH' in varn: varn2 = r'$pH$'
        elif 'alkalinity' in varn: varn2 = r'$ALK$'
        elif varn.startswith('fc'): varn2 = varn.split('_')[1]
        elif varn.startswith('log_flow'): varn2 = r'$lnQ$'
        elif ' ' in varn: varn2 = varn.split()[0]
        varnames2.append(varn2)
    for i in range(len(varnames2)):
        if varnames2[i] == 'Na': varnames2[i] = r'$Na^{+}$'
        elif varnames2[i] == 'Cl': varnames2[i] = r'$Cl^{-}$'
        elif varnames2[i] == 'SO4': varnames2[i] = r'$SO4^{2-}$'
        elif varnames2[i] == 'Ca': varnames2[i] = r'$Ca^{2+}$'
        elif varnames2[i] == 'Al': varnames2[i] = r'$Al^{3+}$'
    return varnames2

# Parameters
level = int(sys.argv[1])
knn = int(sys.argv[2])
print('The approximation level: %d' % level)
print('k = %d' % knn)
filename = './bundled_info_ph_level' + str(level)  + '_k' + str(knn) + '_taumax5.npy'
parentsfile = 'parents_7hr_instream_strict_7var_raw_anomaly_taumax5.pkl'
fig_folder = "./"
taustart, tauend = 5, 150

# Parameters for labels
# pidlabelst = ['$S_{\mathcal{T}}$', '$R_{\mathcal{T}}$',
             # '$U_{self,\mathcal{T}}$', '$U_{cross,\mathcal{T}}$']
# pidlabelsi = ['$S_{\mathcal{J}}$', '$R_{\mathcal{J}}$',
              # '$U_{self,\mathcal{J}}$', '$U_{cross,\mathcal{J}}$']
# pidlabelsd = ['$S_{\mathcal{D}}$', '$R_{\mathcal{D}}$',
              # '$U_{self,\mathcal{D}}$', '$U_{cross,\mathcal{D}}$']
pidlabelst = ['$S^{\mathcal{T}}$', '$R^{\mathcal{T}}$',
             '$U^{\mathcal{T}}_m$', '$U^{\mathcal{T}}_n$']
pidlabelsi = ['$S^{\mathcal{J}}$', '$R^{\mathcal{J}}$',
              '$U^{\mathcal{J}}_m$', '$U^{\mathcal{J}}_n$']
pidlabelsd = ['$S^{\mathcal{D}}$', '$R^{\mathcal{D}}$',
              '$U^{\mathcal{D}}_m$', '$U^{\mathcal{D}}_n$']
colors = ['orange', 'dodgerblue', 'cyan', 'darkgrey']
colors1 = ['yellow', 'lawngreen', 'violet', 'lightgrey']
xlabel, ylabel = r'$\tau$', '$[nats]$'

# Load the information transfer results
result = np.load(filename)[()]
infoset = result['info'][parentsfile]
pidset = result['pid'][parentsfile]
level = result['level']
prtype = result['prtype'][0]
start, end = result['taurange']
taurange = range(start, end)
citset, pitset, titset = infoset[:, 0], infoset[:, 1], infoset[:, 2]
citpid, pitpid, titpid = pidset[:, :4], pidset[:, 4:8], pidset[:, 8:]

# Load the parents
prinfo = pickle.load(open(parentsfile, 'rb'))
pr_old = prinfo['parents'][prtype]

# Reorder
dfn = prinfo['from']
results = pickle.load(open(dfn, 'rb'))
lagfuncs_old=results['results'][0]['parents_xy']
sigthres_old=results['results'][0]['sig_thres_parents_xy']
var_names = ['Na mg/l', 'Al ug/l', 'Ca mg/l', 'pH', 'log_flow', 'SO4 mg/l', 'Cl mg/l']
var_names_old = results['var_names']
order = [var_names_old.index(varele) for varele in var_names]
order2 = [var_names.index(varele) for varele in var_names_old]
lagfuncs, sigthres = lagfuncs_old[order,:,:], sigthres_old[order,:,:]
lagfuncs, sigthres = lagfuncs[:,order,:], sigthres[:,order,:]
pr = dict((order2[key], [(order2[ele[0]],ele[1]) for ele in value]) 
          for (key, value) in pr_old.items())
var_names_easy = get_varnames(var_names)

# Identify the indices of the sources and the target
srcset1, srcset2 = result['src1'][parentsfile], result['src2'][parentsfile]
target = result['target'][parentsfile]
srcset1ind = [var_names.index(src) for src in srcset1]
srcset2ind = [var_names.index(src) for src in srcset2]
targetind  = (var_names.index(target), 0)

print srcset1ind, srcset2ind, targetind

# Create the DAG graph
net = causal_network(pr, lagfuncs, taumax=12)

# The mininum of y-axis of the plot
if level == 0:
    bottom = -0.1
elif level == 1:
    bottom = -0.01
elif level == 2:
    bottom = -0.01

# Plot the PID for T
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
base = np.zeros(len(taurange))[taustart:tauend]
for j in [2,1,3,0]:
    ax.fill_between(taurange[taustart:tauend], base,
                    base+citpid[taustart:tauend,j]+pitpid[taustart:tauend,j],
                    color=colors[j], label=pidlabelst[j])
    base += citpid[taustart:tauend,j]+pitpid[taustart:tauend,j]
# ax.set_xticks([5,200,400])
ax.set_xticks([5,50,100,150])
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_title('$level: %d$' % level)
ax.set_ylim(bottom=bottom)
ax.legend(loc='upper right', facecolor='white',
          framealpha=0.7, frameon=True)

# Plot the PID for test
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
base = np.zeros(len(taurange))[taustart:tauend]
for j in [2]:
    ax.fill_between(taurange[taustart:tauend], base,
                    base+citpid[taustart:tauend,j],
                    color=colors[j], label=pidlabelst[j])
    base += citpid[taustart:tauend,j]
# ax.set_xticks([5,200,400])
ax.set_xticks([5,50,100,150])
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_title('$level: %d$' % level)
ax.set_ylim(bottom=bottom)
ax.legend(loc='upper right', facecolor='white',
          framealpha=0.7, frameon=True)


# Plot the PID for J and D
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
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
# ax.set_xticks([5,200,400])
ax.set_xticks([5,50,100,150])
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
# ax.set_title('$level: %d$' % level)
ax.set_ylim(bottom=bottom)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper right', ncol=1,
          facecolor='white', framealpha=0.7,
          # bbox_to_anchor=(1.,1.),
          fontsize=15,frameon=True)
plt.savefig(fig_folder + 'BCHA_stream_k' +str(knn)+'_level'+str(level)+'.eps', transparent=True, format='eps',bbox_inches='tight')
# plt.show()

# Plot the DAG graph
nvar, taumax = 7, 12
def get_node_number_plot(nodedict, nvar, taumax):
    return taumax-abs(nodedict[1]) + (taumax+1)*abs(nodedict[0])
def filter_neighbor(lagfuncs):
    ndim, ndim, taumax = lagfuncs.shape
    lagfuncs_new = np.copy(lagfuncs)
    for i in range(ndim):
        for j in range(ndim):
            lagfuncs_new[i,j,0] = 0
    return lagfuncs_new

# With MIWTR
w, ptc, f, pwptc, edges, edges2 = net.search_bundled_components(srcset1ind+srcset2ind, targetind, tau=6,
                                                    level=level, transitive=True, returnRemovedEdges=True)
edges = edges + edges2
hnw=[get_node_number_plot(node, nvar, taumax) for node in w]
hntarget=[get_node_number_plot(targetind, nvar, taumax)]
hnptc=[get_node_number_plot(node, nvar, taumax) for node in ptc]
hnf=[get_node_number_plot(node, nvar, taumax) for node in f]
removed_edges=[(get_node_number_plot(n1,nvar,taumax),
                get_node_number_plot(n2,nvar,taumax))
                for (n1,n2) in edges]

fig = plt.figure(figsize=(7, 4), frameon=False)
ax = fig.add_subplot(111, frame_on=False)
tp.plot_time_series_graph2(
    fig=fig, ax=ax,
    lagfuncs=filter_neighbor(np.abs(lagfuncs)),
    sig_thres=sigthres,
    var_names=var_names_easy,
    vmin_edges=0., vmax_edges=.08,
    edge_ticks=.02,
    cmap_edges='Reds',
    label_fontsize=12,
    node_label_size=20,
    node_size=12,
    taumax=taumax,
    link_colorbar_label=r'MIT [nats]',
    highlighted_nodes=hnw,
    highlighted_nodes2=hnptc,
    highlighted_nodes3=hnf,
    highlighted_nodes4=hntarget,
    removed_edges=removed_edges
)
# ax.set_title('w/ MIWTR')
fig.subplots_adjust(left=0.1, right=.98, bottom=.25, top=.9)
plt.savefig(fig_folder + 'DAG_BCHA_noMIWTR.eps', transparent=True, format='eps',bbox_inches='tight')
# print "Print removed edges:"
# print len(removed_edges)
# print removed_edges
# print edges
# print ""

# Without MIWTR
w, ptc, f, pwptc = net.search_bundled_components(srcset1ind+srcset2ind, targetind, tau=6,
                                                    level=level, transitive=False, returnRemovedEdges=True)
hnw=[get_node_number_plot(node, nvar, taumax) for node in w]
hntarget=[get_node_number_plot(targetind, nvar, taumax)]
hnptc=[get_node_number_plot(node, nvar, taumax) for node in ptc]
hnf=[get_node_number_plot(node, nvar, taumax) for node in f]

fig = plt.figure(figsize=(7, 4), frameon=False)
ax = fig.add_subplot(111, frame_on=False)
tp.plot_time_series_graph2(
    fig=fig, ax=ax,
    lagfuncs=filter_neighbor(np.abs(lagfuncs)),
    sig_thres=sigthres,
    var_names=var_names_easy,
    vmin_edges=0., vmax_edges=.08,
    edge_ticks=.02,
    cmap_edges='Reds',
    label_fontsize=12,
    node_label_size=20,
    node_size=12,
    taumax=taumax,
    link_colorbar_label=r'MIT [nats]',
    highlighted_nodes=hnw,
    highlighted_nodes2=hnptc,
    highlighted_nodes3=hnf,
    highlighted_nodes4=hntarget,
    highlighted_edges=removed_edges
)
fig.subplots_adjust(left=0.1, right=.98, bottom=.25, top=.9)
plt.savefig(fig_folder + 'DAG_BCHA_MIWTR.eps', transparent=True, format='eps',bbox_inches='tight')
# ax.set_title('w/o MIWTR')
# plt.show()
