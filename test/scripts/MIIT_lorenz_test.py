'''
This is a test of computing miit and the conditional pid for the Lorenz model.

author: peishi jiang
date: 2017.5.11
'''
# import matplotlib
# matplotlib.use('agg')

import pickle
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.axes3d import Axes3D

from info.core.info import info
from info.utils.pdf_computer import pdfComputer
from info.models.others import Lorenze_model
from info.plot.plot_sur import plot_sr_comparison1d, plot_sur_1d, plot_ii1d

# Figure folder
figureFolder = './figures/'

# Simulation settings
n = 10000
nx, ny, nz = [10, 10, 10]
approach = 'kde_cuda'

# Plotting settings
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Parameters
r_set = np.arange(0.1, 28.1, 1.)
nr = r_set.size
sigma, b = 10., 8./3.
dt = .01
Nt = 10000
x0, y0, z0 = 10., 10., 10.

def main():
    pid11set, pid12set = [], []
    pid21set, pid22set = [], []
    for r in r_set:
        print '*** r = %.1f ***' % r
        # Simulation
        x, y, z = Lorenze_model(x0, y0, z0, Nt, dt, sigma, r, b)

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, label='parametric curve')
        ax.set_title('sigma: %.1f; b: %.1f; r: %.1f' % (sigma, b, r))
        plt.savefig(figureFolder + 'LorenzPlot(r=%.1f).png' % r, dpi=150)

        # Calculate interation information
        pid11, pid12, pid21, pid22 = interaction_information(x, y, z)

        # save them
        pid11set.append(pid11)
        pid12set.append(pid12)
        pid21set.append(pid21)
        pid22set.append(pid22)

    # save all these to file
    with open('pids_Lorenz.pickle', 'w') as f:
        pickle.dump([pid11set, pid12set, pid21set, pid22set], f)

    # print pid11.allInfo
    # print pid12.allInfo
    # print pid21.allInfo
    # print pid22.allInfo


def plot_analysis():
    # Read the INFO
    with open('pids_Lorenz.pickle') as f:
        pid11set, pid12set, pid21set, pid22set = pickle.load(f)

    # Get the info
    rc1, r1, rc2, r2 = np.zeros(nr), np.zeros(nr), np.zeros(nr), np.zeros(nr)
    sc1, s1, sc2, s2 = np.zeros(nr), np.zeros(nr), np.zeros(nr), np.zeros(nr)
    uxc1, ux1, uxc2, ux2 = np.zeros(nr), np.zeros(nr), np.zeros(nr), np.zeros(nr)
    uyc1, uy1, uyc2, uy2 = np.zeros(nr), np.zeros(nr), np.zeros(nr), np.zeros(nr)
    itc1, it1, itc2, it2 = np.zeros(nr), np.zeros(nr), np.zeros(nr), np.zeros(nr)
    iic1, ii1, iic2, ii2 = np.zeros(nr), np.zeros(nr), np.zeros(nr), np.zeros(nr)

    for i in range(nr):
        pid11, pid12, pid21, pid22 = pid11set[i], pid12set[i], pid21set[i], pid22set[i]

        rc1[i], r1[i], rc2[i], r2[i] = pid11.r, pid12.r, pid21.r, pid22.r
        sc1[i], s1[i], sc2[i], s2[i] = pid11.s, pid12.s, pid21.s, pid22.s
        uxc1[i], ux1[i], uxc2[i], ux2[i] = pid11.uxz, pid12.uxz, pid21.uxz, pid22.uxz
        uyc1[i], uy1[i], uyc2[i], uy2[i] = pid11.uyz, pid12.uyz, pid21.uyz, pid22.uyz
        itc1[i], it1[i], itc2[i], it2[i] = pid11.itot, pid12.itot, pid21.itot, pid22.itot
        iic1[i], ii1[i], iic2[i], ii2[i] = pid11.ii, pid12.ii, pid21.ii, pid22.ii

    # Plot interaction information
    plot_ii1d(r_set, iic1, ii1, itc1, it1, xlabel='information (bit)', title='Lorenz case 1', ylabel='r')
    plt.savefig(figureFolder + 'II_comparison_Lorenz_case1.png', dpi=150)
    plot_ii1d(r_set, iic2, ii2, itc2, it2, xlabel='information (bit)', title='Lorenz case 2', ylabel='r')
    plt.savefig(figureFolder + 'II_comparison_Lorenz_case2.png', dpi=150)

    # Plot SUR partition
    plot_sur_1d(r_set, rc1, sc1, uxc1, uyc1, xlabel='r', title='Lorenz case 1 (w/ conditions)', proportion=False)
    plt.savefig(figureFolder + 'SUR_Lorenz_case1_wc.png')
    plot_sur_1d(r_set, r1, s1, ux1, uy1, xlabel='r', title='Lorenz case 1 (w/o conditions)', proportion=False)
    plt.savefig(figureFolder + 'SUR_Lorenz_case1_woc.png')
    plot_sur_1d(r_set, rc2, sc2, uxc2, uyc2, xlabel='r', title='Lorenz case 2 (w/ conditions)', proportion=False)
    plt.savefig(figureFolder + 'SUR_Lorenz_case2_wc.png')
    plot_sur_1d(r_set, r2, s2, ux2, uy2, xlabel='r', title='Lorenz case 2 (w/o conditions)', proportion=False)
    plt.savefig(figureFolder + 'SUR_Lorenz_case2_woc.png')
    plot_sr_comparison1d(r_set, rc1, r1, sc1, s1, xlabel='r', title='Lorenz case 1', proportion=False)
    plt.savefig(figureFolder + 'SR_prop_Lorenz_case1.png')
    plot_sr_comparison1d(r_set, rc2, r2, sc2, s2, xlabel='r', title='Lorenz case 2', proportion=False)
    plt.savefig(figureFolder + 'SR_prop_Lorenz_case2.png')

    # Plot the proportion of SUR information
    plot_sur_1d(r_set, rc1/itc1, sc1/itc1, uxc1/itc1, uyc1/itc1, xlabel='r', title='Lorenz case 1 (w/ conditions)', proportion=True)
    plt.savefig(figureFolder + 'SUR_prop_Lorenz_case1_wc.png')
    plot_sur_1d(r_set, r1/it1, s1/it1, ux1/it1, uy1/it1, xlabel='r', title='Lorenz case 1 (w/o conditions)', proportion=True)
    plt.savefig(figureFolder + 'SUR_prop_Lorenz_case1_woc.png')
    plot_sur_1d(r_set, rc2/itc2, sc2/itc2, uxc2/itc2, uyc2/itc2, xlabel='r', title='Lorenz case 2 (w/ conditions)', proportion=True)
    plt.savefig(figureFolder + 'SUR_prop_Lorenz_case2_wc.png')
    plot_sur_1d(r_set, r2/it2, s2/it2, ux2/it2, uy2/it2, xlabel='r', title='Lorenz case 2 (w/o conditions)', proportion=True)
    plt.savefig(figureFolder + 'SUR_prop_Lorenz_case2_woc.png')
    plot_sr_comparison1d(r_set, rc1/itc1, r1/it1, sc1/itc1, s1/it1, xlabel='r', title='Lorenz case 1', proportion=True)
    plt.savefig(figureFolder + 'SR_prop_Lorenz_case1.png')
    plot_sr_comparison1d(r_set, rc2/itc2, r2/it2, sc2/itc2, s2/it2, xlabel='r', title='Lorenz case 2', proportion=True)
    plt.savefig(figureFolder + 'SR_prop_Lorenz_case2.png')


def interaction_information(x, y, z):
    xt3, xt2, xt1 = x[:-3], x[1:-2], x[2:-1]
    yt, yt1, yt2  = y[3:], y[2:-1], y[1:-2]
    zt1, zt2 = z[2:-1], z[1:-2]

    ######################################################
    # MIIT for the Lorenz model                          #
    # II[X(t-2);X(t-1);Y(t)|Y(t-1),Z(t-1),X(t-3),Y(t-2)] #
    ######################################################
    print 'II[X(t-2);X(t-1);Y(t)|Y(t-1),Z(t-1),X(t-3),Y(t-2)]'
    data11 = np.array([xt2,xt1,yt,yt1,zt1,xt3,yt2]).T
    pdfsolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
    t11, pdf11, coords11 = pdfsolver.computePDF(data11, nbins=[nx, nx, ny, ny, nz, nx, ny])
    pid11 = info(pdf11)
    print t11

    #######################################################
    # II for the Lorenz model                          #
    # II[X(t-2);X(t-1);Y(t)] #
    ######################################################
    print 'II[X(t-2);X(t-1);Y(t)]'
    data12 = np.array([xt2,xt1,yt]).T
    pdfsolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
    t12, pdf12, coords12 = pdfsolver.computePDF(data12, nbins=[nx, nx, ny])
    pid12 = info(pdf12)
    print t12

    #####################################################
    # MIIT for the Lorenz model                          #
    # II[X(t-1);Z(t-1);Y(t)|Y(t-1),Z(t-2),X(t-2),Y(t-2)] #
    ######################################################
    print 'II[X(t-1);Z(t-1);Y(t)|Y(t-1),Z(t-2),X(t-2),Y(t-2)]'
    data21 = np.array([xt1,zt1,yt,yt1,zt2,xt2,yt2]).T
    pdfsolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
    t21, pdf21, coords21 = pdfsolver.computePDF(data21, nbins=[nx, nz, ny, ny, nz, nx, ny])
    pid21 = info(pdf21)
    print t21

    #######################################################
    # II for the Lorenz model                          #
    # II[X(t-1);Z(t-1);Y(t)] #
    ######################################################
    print 'II[X(t-1);Z(t-1);Y(t)]'
    data22 = np.array([xt1,zt1,yt]).T
    pdfsolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
    t22, pdf22, coords22 = pdfsolver.computePDF(data22, nbins=[nx, nz, ny])
    pid22 = info(pdf22)
    print t22

    return pid11, pid12, pid21, pid22

if __name__ == "__main__":
    # main()
    plot_analysis()
