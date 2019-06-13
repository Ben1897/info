import pickle
import copy
import numpy as np
import sys
sys.path.append('../..')

from info.utils.causal_network import causal_network
from info.core.info import info, computeCMIKNN, computeMIKNN
from info.core.info_network import info_network
from info.utils.others import reorganize_data, dropna

##############
# Parameters #
##############
# fns = ['parents_7hr_instream_strict_7var_raw_anomaly_taumax5.pkl',
#        'parents_7hr_instream_strict_6var_SO4_anomaly_taumax5.pkl']
fns = ['parents_7hr_instream_strict_7var_raw_anomaly_taumax5.pkl']

level = int(sys.argv[1])
knn = int(sys.argv[2])
print('The approximation level: %d' % level)
print('k = %d' % knn)
infofilename = './bundled_info_ph_level' + str(level)  + '_k' + str(knn) + '_taumax5.npy'
# level=1
# level = int(input("Please enter the level of approximation: "))
# knn = 5
# taumax = 100
taumax = 20
prtypes = ['parents_nocontemp']
transitive = True
taustart, tauend = 1, 400

srcsets1, srcsets2, targets = {}, {}, {}
srcsets1[fns[0]] = ['Na mg/l', 'Al ug/l', 'Ca mg/l']
srcsets2[fns[0]] = ['Cl mg/l', 'SO4 mg/l']
targets[fns[0]] = 'pH'

# srcsets1[fns[1]] = ['fc_Na_mg/l_UHF', 'fc_Al_ug/l_UHF', 'fc_Ca_mg/l_UHF']
# srcsets2[fns[1]] = ['fc_Cl_mg/l_UHF', 'fc_SO4_mg/l_UHF']
# targets[fns[1]] = 'fc_pH_UHF'

# if level == 1:
    # infofilename = './bundled_info_ph_' +   + 'taumax5.npy'
# elif level == 2:
    # infofilename = './bundled_info_ph_level2_taumax5.npy'
# elif level == 0:
    # infofilename = './bundled_info_ph_level0_taumax5.npy'
# else:
    # raise Exception('Wrong level!')

#'fc_Ca_mg/l_UHF', 'fc_Al_ug/l_UHF', 'fc_Cl_mg/l_UHF', 'fc_Na_mg/l_UHF', 'fc_SO4_mg/l_UHF', 'fc_pH_UHF'
#'log_flow', 'Na mg/l', 'Cl mg/l', 'Al ug/l', 'Ca mg/l', 'SO4 mg/l', 'pH'
###############
# Compute CIT #
###############
info_to_save = {}
pid_to_save = {}
sst_to_save = {}
sizedim_to_save = {}
for i in range(len(fns)):
    # Load the parents
    prinfo = pickle.load(open(fns[i], 'rb'))
    # print prinfo['parents']
    prs = [prinfo['parents'][prtype] for prtype in prtypes]

    # Load the data
    dfn = prinfo['from']
    print(dfn)
    results = pickle.load(open(dfn, 'rb'))
    var_names = results['var_names']

    # Identify the indices of the sources and the target
    srcset1 = [var_names.index(src) for src in srcsets1[fns[i]]]
    srcset2 = [var_names.index(src) for src in srcsets2[fns[i]]]
    target  = (var_names.index(targets[fns[i]]), 0)

    print(srcset1, srcset2, target)

    data    = results['fulldata'][0]

    # Number of variables
    npts, nvar = data.shape

    # Coupling strengths and threshold
    lagfuncs=results['results'][0]['parents_xy']

    # print dfn, fns[i]

    # Compute info metrics
    infoset = np.zeros([tauend-taustart, 5])
    pidset = np.zeros([tauend-taustart, 12])
    datasize = np.zeros([tauend-taustart, 3])
    dimsize = np.zeros([tauend-taustart, 3])

    for j in range(len(prtypes)):
        print(fns[i] + ' -- ' + prtypes[j])
        # Get the maximum lag of the parent
        paset = [pa for pset in prs[j].values() for pa in pset]
        mpl = np.max([-pr[1] for pr in paset])
        print(mpl)
        # Construct the network
        net = causal_network(prs[j], lagfuncs, taumax)

        # Construct the information network
        infonet = info_network(data, prs[j], lagfuncs=lagfuncs, taumax=taumax,
                               approach='knn', k=knn)

        # Compute the metrics
        results = infonet.compute_bundled_set(srcset1, srcset2, target, taumax=tauend-taustart,level=level,
                                              transitive=transitive)
        # info from causal history
        infoset[:,0], infoset[:,1], infoset[:,2] = results['cit'], results['past'], results['tit']

        # PID from causal history
        pidset[:,:4], pidset[:,4:8], pidset[:,8:] = results['pidimmediate'], results['pidpast'], results['pidtotal']

        # Data size and dimensionality
        datasize[:,:], dimsize[:,:] = results['datasize'], results['dimsize']


    info_to_save[fns[i]] = infoset
    pid_to_save[fns[i]] = pidset
    sizedim_to_save[fns[i]] = {"dimsize":dimsize, "datasize":datasize}

results = {"info":info_to_save, "pid":pid_to_save, "sizedim":sizedim_to_save, "level":level,
           "src1":srcsets1,"src2":srcsets2,"target":targets,
           "transitive": transitive, "prtype":prtypes, "knn":knn, "taurange": (taustart,tauend)}


####################
# Save the results #
####################
# infofilename = 'info_instream_new_7var_raw_anomaly_transred_taumax5.pkl'

np.save(infofilename, results)

# dfns = ['stream_cmiknn_parallel_instream_results.pkl',
#         'stream_cmiknn_parallel_instream_strict_results.pkl',
#         'stream_cmiknn_parallel_instream_results.pkl',
#         'stream_cmiknn_parallel_instream_results.pkl',
#         'stream_cmiknn_parallel_instream_results.pkl',
