"""
Parse the excel data file provided by Kirchner and Neal (2013).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from info.core.info import computeCMIKNN, computeMIKNN
from info.utils.others import reorganize_data, dropna
from info.utils.causal_network import exclude_intersection

from matplotlib.ticker import FormatStrFormatter

excel_file = "/home/pjiang6/data/stream/KirchnerPNAS/sd01.xlsx"

sheetname = "7hour UHF data used in paper"

# Function for parse decimal year (to hours)
def parse_decimal_year_to_hr(dy):
    year = int(dy)
    rem  = dy - year

    base = datetime(year, 1, 1)
    # result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)
    # result = base + timedelta(hours=(base.replace(year=base.year + 1) - base).total_hours() * rem)
    hours = (base.replace(year=base.year + 1) - base).total_seconds() / (3600) * rem

    result = base + timedelta(hours=round(hours))

    return result

# Function for parse decimal year (to weeks)
def parse_decimal_year_to_wk(dy):
    year = int(dy)
    rem  = dy - year

    base = datetime(year, 1, 1)
    # result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)
    # result = base + timedelta(hours=(base.replace(year=base.year + 1) - base).total_hours() * rem)
    weeks = (base.replace(year=base.year + 1) - base).total_seconds() / (3600*24*7) * rem

    result = base + timedelta(weeks=round(weeks))

    return result

# Function for parse date time format "%m/%d/%y %H:%M"
def parse_datetime(dt):
    return datetime.strptime(str(dt), "%m/%d/%y %H:%M")

# Read the sheet
def parse_excel_data(filename, sheetname, header, index_col="year", parse_dates=True, date_parser=parse_decimal_year_to_hr, freq='7h'):
    dframe = pd.read_excel(filename, sheetname=sheetname, header=header, index_col=index_col,
                           parse_dates=parse_dates, date_parser=date_parser)

    if freq is None:
        return dframe
    else:
        return dframe.asfreq(freq)


# Filter the sigthres to make sure that
# each variable has at most two contributions towards the other variable
# (i.e, one contemporaneous link and one directed link with the most link strength).
def filter_sigthres(lagfuncs, sigthres):
    lagfuncs, sigthres = np.copy(lagfuncs), np.copy(sigthres)
    shape = sigthres.shape
    sigthres_new = np.ones(shape)  # Let all the new sigthres be one
    nvar = shape[0]
    for i in range(nvar):
        for j in range(nvar):
            lagfs, thres = np.abs(lagfuncs[j, i, :]), np.abs(sigthres[j, i, :])
            # The contemporaneous undirected link
            if i != j and lagfs[0] >= thres[0]:
                sigthres_new[j, i, 0] = thres[0]
            # The directed link
#             index = np.where(np.abs(lagfs) >= thres)[0]
            index = np.where(lagfs < thres)[0]
            # Include the lag-zero if it is self-dependence
            if i == j and 0 not in index:
#                 index = np.concatenate(index, [0])
                index += [0]
            # Make all the insignificant zeros
            lagfs[index] = 0.
            if index.size:
                dl = np.argmax(lagfs)
                sigthres_new[j, i, dl] = thres[dl]

    return sigthres_new


# Filter the sigthres to make sure that
# each variable has at most two contributions towards the other variable
# (i.e, one contemporaneous link and one directed link with the smallest lag).
def filter_sigthres_1storder(lagfuncs, sigthres):
    lagfuncs, sigthres = np.copy(lagfuncs), np.copy(sigthres)
    shape = sigthres.shape
    sigthres_new = np.ones(shape)  # Let all the new sigthres be one
    nvar = shape[0]
    for i in range(nvar):
        for j in range(nvar):
            lagfs, thres = np.abs(lagfuncs[j, i, :]), np.abs(sigthres[j, i, :])
            # The contemporaneous undirected link
            if i != j and lagfs[0] >= thres[0]:
                sigthres_new[j, i, 0] = thres[0]
            # The directed link
#             index = np.where(np.abs(lagfs) >= thres)[0]
            index = np.where(lagfs < thres)[0]
            # Include the lag-zero if it is self-dependence
            if i == j and 0 not in index:
#                 index = np.concatenate(index, [0])
                index += [0]
            # Make all the insignificant link zeros
            lagfs[index] = 0.
            if index.size:
                dl = np.where(lagfs>0)[0]
                if dl != []:
                    sigthres_new[j, i, dl] = thres[dl[0]]

    return sigthres_new



# Filter the sigthres to make sure that
# the lagfuncs below the level are insignificant
def filter_sigthres_level(lagfuncs, sigthres, level=0):
    lagfuncs, sigthres = np.copy(lagfuncs), np.copy(sigthres)
    shape = sigthres.shape
    sigthres_new = np.copy(sigthres)  # Let all the new sigthres be one
    nvar = shape[0]

    highest = lagfuncs.max() + 1e-05

    sigthres_new[lagfuncs < level] = highest
    # for i in range(nvar):
        # for j in range(nvar):
            # lagfs, thres = np.abs(lagfuncs[j, i, :]), np.abs(sigthres[j, i, :])
#             # The contemporaneous undirected link
#             if i != j and lagfs[0] >= thres[0]:
#                 sigthres_new[j, i, 0] = thres[0]
#             # The directed link
# #             index = np.where(np.abs(lagfs) >= thres)[0]
#             index = np.where(lagfs < thres)[0]
#             # Include the lag-zero if it is self-dependence
#             if i == j and 0 not in index:
# #                 index = np.concatenate(index, [0])
#                 index += [0]
#             # Make all the insignificant zeros
#             lagfs[index] = 0.
#             if index.size:
#                 dl = np.argmax(lagfs)
#                 sigthres_new[j, i, dl] = thres[dl]

    return sigthres_new


# Generate the parents and neighbors based on the lagfuncs and sigthres
# Initialize the parents and neighbors
def generate_parents_neighbors(lagfuncs, sigthres):
    parents_neighbors = {}
    nvar, nlag = lagfuncs.shape[0], lagfuncs.shape[2]
    lags = np.arange(nlag)
    for i in range(nvar):
        parents_neighbors[i] = []
        # Get the parents or neighbors whose lagfuncs are larger than sigthres
        for j in range(nvar):
            lagfs, thres = lagfuncs[j, i, :], sigthres[j, i, :]
            index = np.where(np.abs(lagfs) > thres)[0]
            if i == j and 0 in index:
                index = index[1:]
            for k in lags[np.where(np.abs(lagfs) > thres)]:
                    parents_neighbors[i] += [(j, -k)]

    return parents_neighbors

# Filter the contemporaneous links by forcing their coupling strength to be zero
def filter_neighbor(lagfuncs):
    ndim, ndim, taumax = lagfuncs.shape

    lagfuncs_new = np.copy(lagfuncs)

    for i in range(ndim):
        for j in range(ndim):
            lagfuncs_new[i,j,0] = 0

    return lagfuncs_new


# Function for generating cumulative information transfer and the corresponding mutual information & momentary information transfer along causal paths
def generate_info(network, data, sources, target, taustart, tauend, lagfuncs=None, maxdim=None, knn=5, maxparentlag=1, verbosity=1):
    citset   = np.zeros(tauend-taustart)
    pastset  = np.zeros(tauend-taustart)
    titset   = np.ones(tauend-taustart)
    datasize = np.zeros(tauend-taustart)
    dimsize  = np.zeros(tauend-taustart)

    # Get the parents of the target
    pt = network.search_parents(target)
    print "parents of the target:"
    print pt

    # Compute the total information
    # Reorganize the data
    data1 = reorganize_data(data, [target]+pt)
    data1 = dropna(data1)
    if data1.shape[0] < 100:
        print 'Not enough time series datapoints (<100)!'
        return citset, mitset, miset
    tit = computeMIKNN(data1, k=knn, xyindex=[1])
    titset = tit*titset

    # Compute I and P
    for i in range(tauend-taustart):
        print ""
        print ""
        print "Target and sources:"
        print target, sources
        print ""
        if i <= maxparentlag-1:
            print str(i) + ' search condition sets'
            try:
                w, ptc, cpaths = network.search_cit_components(sources, target, verbosity=verbosity)
            except:
                print "Warning:!!!"
                print network.search_cit_components(sources, target)
                continue
                # return citset, mitset, miset, indiset

            # Choose the maxdim maximum parents
            if maxdim is not None:
                print 'Select the %d most significant parents... ' % maxdim
                # Get the parents not in the causal subgraph
                ptnc = list(set(pt) - set(ptc))
                # Get the maxdim maximum parents in w
                w1 = list(set(w) - set(ptnc))
                if len(w1) > maxdim:
                    strengths = np.zeros(len(w1))
                    for j in range(len(w1)):
                        # Get the children of wele
                        children = network.search_children(w1[j])
                        # Get the children in the causal subgraph
                        ccs = list(set(children) & set(cpaths))
                        # Get the coupling strengths between w and ccs
                        wstrengths = np.zeros(len(ccs))
                        for k in range(len(ccs)):
                            wstrengths[k] = lagfuncs[w1[j][0],ccs[k][0],ccs[k][1]-w1[j][1]]
                        # Select the maximum
                        strengths[j] = wstrengths.max()
                    # Find the maxdim largest
                    indices = strengths.argsort()[-maxdim-1:]
                    print "Proportion of the strengths of the selected parents: %.3f" % (np.sum(strengths[indices]) / np.sum(strengths),)
                    w1 = [w1[ind] for ind in indices]
                    # update W
                    w = w1 + ptnc
        else:
            print i
            # Get the parents not in the causal subgraph
            ptnc = list(set(pt) - set(ptc))
            # Get the maxdim maximum parents in w
            w1 = list(set(w) - set(ptnc))
            # Update w1
            w1 = [(wele[0], wele[1]-1) for wele in w1]
            # Combine w1 and ptnc
            w = w1 + ptnc

        # Reorganize the data
        data21   = reorganize_data(data, [target]+ptc+w)
        data22   = reorganize_data(data, [target]+w)
        # Drop the nan values
        data21   = dropna(data21)
        data22   = dropna(data22)
        print data21.shape
        datasize[i], dimsize[i] = data21.shape
        xyindex1 = [1,1+len(ptc)]
        if data21.shape[0] < 100 or data22.shape[0] < 100:
            print 'Not enough time series datapoints (<100)!'
            return citset, mitset, miset
        citset[i]  = computeCMIKNN(data21, k=knn, xyindex=xyindex1)
        pastset[i] = computeMIKNN(data22, k=knn, xyindex=[1])

        sources = [(s[0], s[1]-1) for s in sources]

    # return citset, mitset, miset, indiset
    return citset, pastset, titset, datasize, dimsize

# Function for loading the cit data
def load_cit_data(filename, datasrc, prtype, start):
    import pickle

    infoall = pickle.load(open(filename, 'rb'))
    infoset = infoall['info'][datasrc][prtype,:,:,:]
    sizedim = infoall['sizedim'][datasrc][prtype,:,:,:]
    taustart, tauend = infoall['taurange']
    citset, pitset = infoset[:,0,start:], infoset[:,1,start:]
    nvar, _ = citset.shape

    taurange = np.array(range(taustart+start,tauend))

    try:
        titset = infoset[:,2,start:]
        return citset, pitset, titset, taurange
    except:
        return citset, pitset, taurange
    
def load_cit_data2(filename, datasrc, prtype, start):
    import pickle

    infoall = pickle.load(open(filename, 'rb'))
    infoset = infoall['info'][datasrc][prtype,:,:,:]
    dimsize = infoall['sizedim'][datasrc]['dimsize'][prtype,:,:,:]
    datasize = infoall['sizedim'][datasrc]['datasize'][prtype,:,:,:]
    taustart, tauend = infoall['taurange']
    citset, pitset, titset = infoset[:,start:,0], infoset[:,start:,1], infoset[:,start:,2]
    pidpast, pidcit = infoset[:,start:,7:], infoset[:,start:,3:7]
    try:
        sstpast = infoall['sstpast'][datasrc][prtype,:,:,:]
    except:
        print "No sst for pit"
        sstpast = None
    nvar, _ = citset.shape

    taurange = np.array(range(taustart+start,tauend))
    
    return citset, pitset, titset, pidcit, pidpast, sstpast, taurange, dimsize, datasize

def load_cit_data3(filename, datasrc, prtype, start):
    import pickle

    infoall = pickle.load(open(filename, 'rb'))
    infoset = infoall['info'][datasrc][prtype,:,:,:]
    pidset  = infoall['pid'][datasrc][prtype,:,:,:]
    #print infoset
    dimsize = infoall['sizedim'][datasrc]['dimsize'][prtype,:,:,:]
    datasize = infoall['sizedim'][datasrc]['datasize'][prtype,:,:,:]
    taustart, tauend = infoall['taurange']
    citset, pitset, titset = infoset[:,start:,0], infoset[:,start:,1], infoset[:,start:,2]
    #print infoset[:,start:,7:]
    #pidpast, pidcit = pidset[:,start:,4:], pidset[:,start:,:4]
    pidpast, pidcit = pidset[:,start:,4:8], pidset[:,start:,:4]
    try:
        sstpast = infoall['sstpast'][datasrc][prtype,:,:,:]
    except:
        print "No sst for pit"
        sstpast = None
    nvar, _ = citset.shape

    taurange = np.array(range(taustart+start,tauend))
    
    return citset, pitset, titset, pidcit, pidpast, sstpast, taurange, dimsize, datasize

# Function for loading the network
def load_net(netfile, datasrc, taumax, approach, k):
    import pickle
    from info.core.info_network import info_network
    # Read the data
    results = pickle.load(open(netfile, 'rb'))
    rawdata = results['rawdata']
    data = results['fulldata'][0]
    varnames = results['var_names']

    # Read the parents
    causalDict = pickle.load(open(datasrc, 'rb'))['parents']['parents_nocontemp']

    # Coupling strengths and threshold
    lagfuncs=results['results'][0]['parents_xy']
    sigthres=results['results'][0]['sig_thres_parents_xy']

    # Construct the network
    net = info_network(data, causalDict, taumax=taumax, lagfuncs=lagfuncs, approach=approach, k=k)
    
    return rawdata, data, net, varnames, lagfuncs, sigthres

# Function for generally plotting information
def plot_info_general(citset, pitset, taurange, labels, labels2, xlabel, ylabel, ylabel2,
                      colors, markers, alphas, fig=None, axes=None, figsize=(18,12), sharex=False):
    import matplotlib.pyplot as plt

    nvar, _ = citset.shape
    
    if axes is None:
        fig, axes = plt.subplots(nrows=nvar,ncols=2,figsize=figsize,sharex=sharex)
    for i in range(nvar):
        ax, infoset = axes[i,0], [citset[i,:],pitset[i,:]]
        # ax, infoset = axes[i,0], [citset[i,:],pitset[i,:],
        #                           citset[i,:]+pitset[i,:]]
        # plot_t(citset[i,:], pitset[i,:], taurange=taurange, labels=labels, ax=ax, logx=False)
        plot_info(infoset, taurange=taurange, labels=labels, ax=ax,
                  colors=colors, markers=markers, alphas=alphas, logx=False)
        ax = axes[i,1]
        plot_povert2(citset[i,:], pitset[i,:], taurange=taurange, labels=labels, ax=ax, logx=False)
        # plot_povert(citset[i,:], pitset[i,:], taurange=taurange, labels=labels, ax=ax, logx=False)
        # ax = axes[i,2]
        # plot_citvmi(citset[i,:]/pitset[i,:], taurange=taurange,
                    # ax=ax, labels=labels2, logx=True, logy=True)

    # axes[-1,1].legend(ncol=2, bbox_to_anchor=(0.8, -0.35))
    # axes[-1,0].legend(ncol=4, bbox_to_anchor=(0.9, -0.35))
    axes[-1,0].legend(ncol=3, bbox_to_anchor=(1.6, -0.4), fontsize=15,frameon=False)

    for i in range(nvar):
        axes[i,0].set_ylabel(ylabel2[i])

    for j in range(2):
        axes[-1,j].set_xlabel(xlabel)

    axes[0,0].set_title(r'$\mathcal{T} = \mathcal{J} + \mathcal{D}$')
    axes[0,1].set_title(r'$\mathcal{D} / \mathcal{T}$')
    # axes[0,2].set_title(r'$\mathcal{I} / \mathcal{P}$')

    plt.subplots_adjust(wspace=0.1)

    return fig, axes

# Function for plotting the information metrics derived from the previous one
def plot_t(citset, pitset, taurange=None, labels=None, xlabel=None, ylabel=None, logx=False,
           title=None, ax=None, colors=None, markers=None, alphas=None):
    from matplotlib.ticker import FormatStrFormatter

    if ax is None:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)

    if taurange is None:
        taurange = range(infoset[0].size)

    if colors is None:
        colors = ['k', 'r',]

    if markers is None:
        markers = '.'

    ax.plot(taurange, citset+pitset, markers, color=colors[0])

    if title:
        ax.set_title(title)

    if xlabel:
        ax.set_ylabel('[nats]')

    if logx:
        ax.set_xscale("log")

    ax.tick_params(direction='in')

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # if labels is not None:
        # ax.legend(loc='lower left', bbox_to_anchor=(1.05, 1.))

    return ax

# Function for plotting the information metrics derived from the previous one
def plot_povert2(citset, pitset, taurange=None, labels=None, xlabel=None, ylabel=None, logx=False,
                title=None, ax=None, colors=None, markers=None, alphas=None):

    if ax is None:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)

    if taurange is None:
        taurange = range(infoset[0].size)

    if colors is None:
        colors = ['k', 'r']

    if markers is None:
        markers = '.'

    # ax.plot(taurange, pitset/(citset+pitset), markers, color=colors[1])
    ax.fill_between(taurange, np.zeros(taurange.size), pitset/(citset+pitset))
    ax.set_ylim([0, 1.1])
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels([r'$50\%$', r'$100\%$'])
    ax.yaxis.tick_right()

    ax.axhline(y=1, linewidth=0.5, linestyle='--', c='k')
    ax.axhline(y=0.5, linewidth=0.5, linestyle='--', c='k')
    # ax.axhline(y=0, linewidth=0.5, linestyle='--', c='k')

    if title:
        ax.set_title(title)

    if xlabel:
        ax.set_ylabel('[nats]')

    if logx:
        ax.set_xscale("log")

    ax.tick_params(direction='in')

    # if labels is not None:
        # ax.legend(loc='lower left', bbox_to_anchor=(1.05, 1.))

    return ax

# Function for plotting the information metrics derived from the previous one
def plot_povert(citset, pitset, taurange=None, labels=None, xlabel=None, ylabel=None, logx=False,
                title=None, ax=None, colors=None, markers=None, alphas=None):
    from matplotlib.ticker import FormatStrFormatter

    if ax is None:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)

    if taurange is None:
        taurange = range(infoset[0].size)

    if colors is None:
        colors = ['k', 'r',]

    if markers is None:
        markers = '.'

    ax.plot(taurange, citset+pitset, markers, color=colors[0])
    ax2 = ax.twinx()
    ax2.plot(taurange, pitset/(citset+pitset), markers, color=colors[1])
    ax2.tick_params('y', colors=colors[1])
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_yticklabels([r'$0\%$', r'$50\%$', r'$100\%$'])

    ax.set_ylim(bottom=-0.1)

    if title:
        ax.set_title(title)

    if xlabel:
        ax.set_ylabel('[nats]')

    if ylabel:
        ax.set_xlabel(r'$\tau$')

    if logx:
        ax.set_xscale("log")

    ax.tick_params(direction='in')
    ax2.tick_params(direction='in')

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # if labels is not None:
        # ax.legend(loc='lower left', bbox_to_anchor=(1.05, 1.))

    return ax

# Function for plotting the information metrics derived from the previous one
def plot_info(infoset, taurange=None, labels=None, xlabel=None, ylabel=None, logx=False,
              title=None, ax=None, colors=None, markers=None, alphas=None):
    if ax is None:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)

    if taurange is None:
        taurange = range(infoset[0].size)

    if colors is None:
        colors = ['b.', 'r.', 'k.', 'g.', 'y.']

    if markers is None:
        markers = ['.' for i in range(len(infoset))]

    if alphas is None:
        alphas = [1. for i in range(len(infoset))]

    lines = ['-', '-', '-']

    for i in range(len(infoset)):
        if labels is None:
            # ax.plot(taurange, infoset[i], markers[i], color=colors[i], alpha=alphas[i])
            ax.plot(taurange, infoset[i], lines[i], color=colors[i], alpha=alphas[i])
        else:
            # ax.plot(taurange, infoset[i], markers[i], color=colors[i], alpha=alphas[i], label=labels[i])
            ax.plot(taurange, infoset[i], lines[i], color=colors[i], alpha=alphas[i], label=labels[i])

    infomean = np.array(infoset).mean()
    infomax = np.array(infoset).max()
    if infomax > 1:
        ax.set_ylim([0, max(1., infomax + 0.3)])
    else:
        ax.set_ylim([0, .6+infomean/2])
    ax.set_ylim(bottom=0)

    if title:
        ax.set_title(title)

    if xlabel:
        ax.set_ylabel('[nats]')

    if ylabel:
        ax.set_xlabel(r'$\tau$')

    if logx:
        ax.set_xscale("log")

    # if labels is not None:
        # ax.legend(loc='lower left', bbox_to_anchor=(1.05, 1.))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    return ax


# Function for plotting the information metrics derived from the previous one
def plot_citvmi(citvmi, taurange=None, labels=None, xlabel=None, ylabel=None, logx=False, logy=False,
                title=None, ax=None, ls=None):
    if ax is None:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)

    if taurange is None:
        taurange = range(citvmi.size)

    citdom = citvmi >= 1
    midom  = citvmi < 1
    # ax.semilogy(taurange, citvmi, 'k.')
    ax.plot(taurange[citdom], citvmi[citdom], '.', color='blue', label=labels[0])
    ax.plot(taurange[midom], citvmi[midom], '.', color='orange', label=labels[1])
    ax.axhline(y=1, linewidth=0.5, c='k')

    if logx:
        ax.set_xscale("log")

    if logy:
        ax.set_yscale("log")

    ax.set_ylim([0.05, 20])
    # ax.set_ylim([0.1, max(10, citvmi.max())])
    ax.set_yticks([0.1, 1, 10])
    ax.set_yticklabels([0.1, 1, 10])
    # ax.yaxis.tick_right()
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels([1, 10, 100])
    ax.tick_params(direction='in')

    if title:
        ax.set_title(title)

    if xlabel:
        ax.set_xlabel(r'$\tau$')

    return ax
