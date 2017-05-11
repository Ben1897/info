'''
This is a test of computing miit and the conditional pid for the Lorenz model.

author: peishi jiang
date: 2017.5.11
'''
# import matplotlib
# matplotlib.use('agg')

import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.axes3d import Axes3D

from info.core.info import info
from info.utils.pdf_computer import pdfComputer
from info.models.others import Lorenze_model
from info.plot.plot_sur import plot_sr_comparison3d, plot_sr_prop_comparison3d, plot_ii, plot_sur_prop

n = 10000
nx, ny, nz = [10, 10, 10]
approach = 'kde_cuda'

# Parameters
sigma, b, r = 10., 8./3., 28.
dt = .01
Nt = 10000
x0, y0, z0 = 10., 10., 10.

# Simulation
x, y, z = Lorenze_model(x0, y0, z0, Nt, dt, sigma, r, b)

xt3, xt2, xt1 = x[:-3], x[1:-2], x[2:-1]
yt, yt1, yt2  = y[3:], y[2:-1], y[1:-2]
zt1, zt2 = z[2:-1], z[1:-2]

######################################################
# MIIT for the Lorenz model                          #
# II[X(t-2);X(t-1);Y(t)|Y(t-1),Z(t-1),X(t-3),Y(t-2)] #
######################################################
data11 = np.array([xt2,xt1,yt,yt1,zt1,xt3,yt2]).T
pdfsolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
t11, pdf11, coords11 = pdfsolver.computePDF(data11, nbins=[nx, nx, ny, ny, nz, nx, ny])
pid11 = info(pdf11)
print 'II[X(t-2);X(t-1);Y(t)|Y(t-1),Z(t-1),X(t-3),Y(t-2)]'
print pid11.allInfo

#######################################################
# II for the Lorenz model                          #
# II[X(t-2);X(t-1);Y(t)] #
######################################################
data12 = np.array([xt2,xt1,yt]).T
pdfsolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
t12, pdf12, coords12 = pdfsolver.computePDF(data12, nbins=[nx, nx, ny])
pid12 = info(pdf12)
print 'II[X(t-2);X(t-1);Y(t)]'
print pid12.allInfo

#####################################################
# MIIT for the Lorenz model                          #
# II[X(t-1);Z(t-1);Y(t)|Y(t-1),Z(t-2),X(t-2),Y(t-2)] #
######################################################
data21 = np.array([xt1,zt1,yt,yt1,zt2,xt2,yt2]).T
pdfsolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
t21, pdf21, coords21 = pdfsolver.computePDF(data21, nbins=[nx, nz, ny, ny, nz, nx, ny])
pid21 = info(pdf21)
print 'II[X(t-1);Z(t-1);Y(t)|Y(t-1),Z(t-2),X(t-2),Y(t-2)]'
print pid21.allInfo

#######################################################
# II for the Lorenz model                          #
# II[X(t-1);Z(t-1);Y(t)] #
######################################################
data22 = np.array([xt1,zt1,yt]).T
pdfsolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
t22, pdf22, coords22 = pdfsolver.computePDF(data22, nbins=[nx, nz, ny])
pid22 = info(pdf22)
print 'II[X(t-1);Z(t-1);Y(t)]'
print pid22.allInfo
