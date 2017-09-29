'''
this is a test of computing miit and the conditional pid for the nonlinear common driver model

author: peishi jiang
date: 2017.5.9
'''
# import matplotlib
# matplotlib.use('agg')

import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.axes3d import Axes3D

from time import time

from info.core.info import info
from info.utils.pdf_computer import pdfComputer
from info.models.others import common_driver_linear, common_driver_nonlinear
from info.plot.plot_sur import plot_sr_comparison3d, plot_sr_prop_comparison3d, plot_ii, plot_sur_prop, plot_pid

n = 10000
nx, ny, nz, nw = [20, 20, 20, 20]
approach = 'kde_cuda'
PLOT = 1
# Figure folder
figureFolder = '/projects/users/pjiang6/figures/'

##############################################
# miit for the nonlinear common driver model #
# ii[x(t-1);y(t-1);z(t)|w(t-2)]              #
##############################################
print 'nonlinear common driver model...'
# parameter settings
ncwx, ncwy = 10, 10
cwx_min, cwx_max = -2, 2
cwy_min, cwy_max = -2, 2
cwx_set = np.linspace(cwx_min, cwx_max, ncwx)
cwy_set = np.linspace(cwy_min, cwy_max, ncwy)
cz = .5
sw, sx, sy, sz = 1., 1., 1., 1.

rc, r = np.zeros([ncwx, ncwy]), np.zeros([ncwx, ncwy])
sc, s = np.zeros([ncwx, ncwy]), np.zeros([ncwx, ncwy])
miit, ii = np.zeros([ncwx, ncwy]), np.zeros([ncwx, ncwy])
uxc, ux = np.zeros([ncwx, ncwy]), np.zeros([ncwx, ncwy])
uyc, uy = np.zeros([ncwx, ncwy]), np.zeros([ncwx, ncwy])
itc, it = np.zeros([ncwx, ncwy]), np.zeros([ncwx, ncwy])
iic, ii = np.zeros([ncwx, ncwy]), np.zeros([ncwx, ncwy])

start = time()
for i in range(ncwx):
    cwx = cwx_set[i]
    for j in range(ncwy):
        cwy = cwy_set[j]
        # simulation
        w, x, y, z = common_driver_nonlinear(n, cwx, cwy, cz, sw, sy, sx, sz)
        w = w[:-2]
        x = x[1:-1]
        y = y[1:-1]
        z = z[2:]
        data1 = np.array([x, y, z, w]).T
        data2 = np.array([x, y, z]).T

        # compute the pdf
        # miit
        pdfsolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
        t1, pdf1, coords1 = pdfsolver.computePDF(data1, nbins=[nx, ny, nz, nw])
        # print t1
        # ii
        pdfsolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
        t2, pdf2, coords2 = pdfsolver.computePDF(data2, nbins=[nx, ny, nz])
        # print t2

        # compute pid
        # miit
        pid1 = info(ndim=3, pdfs=pdf1, base=np.e, conditioned=True)
        # ii
        pid2 = info(ndim=3, pdfs=pdf2, base=np.e)

        # get sur values
        rc[i,j], r[i,j]    = pid1.r, pid2.r
        sc[i,j], s[i,j]    = pid1.s, pid2.s
        miit[i,j], ii[i,j] = pid1.ii, pid2.ii
        uxc[i,j], ux[i,j]  = pid1.uxz, pid2.uxz
        uyc[i,j], uy[i,j]  = pid1.uyz, pid2.uyz
        itc[i,j], it[i,j]  = pid1.itot, pid2.itot
        iic[i,j], ii[i,j]  = pid1.ii, pid2.ii

        # print 'hi ---'

print "Time usage %.2f" % (time() - start)

# plot
xlabel, ylabel = 'cwx', 'cwy'
xlabeltex, ylabeltex = '$c_{WX}$', '$c_{WY}$'
extent = [cwx_min, cwx_max, cwy_max, cwy_min]
xv, yv = np.meshgrid(cwx_set, cwy_set, indexing='ij')
small_size = 30
medium_size = 30
bigger_size = 30

plt.rc('text', usetex=True)              # usage of tex
plt.rc('font', size=small_size)          # controls default text sizes
plt.rc('axes', titlesize=bigger_size)    # fontsize of the axes title
plt.rc('axes', labelsize=bigger_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=small_size)    # legend fontsize
plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

# plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01)

if PLOT == 1:
    # plot interaction information and total information (comparison)
    plot_ii(xv, yv, iic, ii, itc, it, xlabel, ylabel)
    plt.savefig(figureFolder + 'ncd_interaction_info_comp.png', dpi=150)

    # plot sur (w/ conditions, percentage)
    plot_sur_prop(xv, yv, rc/itc, sc/itc, uxc/itc, uyc/itc, xlabel, ylabel, 'w/ cond\'t')
    plt.savefig(figureFolder + 'ncd_sur_miit.png', dpi=150)

    # plot sur (w/o conditions, percentage)
    plot_sur_prop(xv, yv, r/it, s/it, ux/it, uy/it, xlabel, ylabel, 'w/o cond\'t')
    plt.savefig(figureFolder + 'ncd_sur_ii.png', dpi=150)

    # plot sr comparison
    plot_sr_comparison3d(xv, yv, rc, r, sc, s, xlabel, ylabel)
    plt.savefig(figureFolder + 'ncd_sr_info_comparison.png', dpi=150)

    # plot the proportion of sr in the total information
    plot_sr_prop_comparison3d(xv, yv, rc/itc, r/it, sc/itc, s/itc, xlabel, ylabel)
    plt.savefig(figureFolder + 'ncd_prop_sr_info_comparison.png', dpi=150)

    vmin, vmax = np.min([iic, ii]), np.max([iic, ii])
    # plot all the figures from GMII
    plot_pid(xv, yv, iic, rc/itc, sc/itc, uxc/itc, uyc/itc, xlabeltex, ylabeltex,
             vmin1=vmin, vmax1=vmax, option='MPID', prop=True)
    # plt.tight_layout(pad=0)
    plt.savefig(figureFolder + 'ncd_mpid_prop.png', bbox_inches='tight', dpi=250)

    # plot all the figures from II
    plot_pid(xv, yv, ii, r/it, s/it, ux/it, uy/it, xlabeltex, ylabeltex,
             vmin1=vmin, vmax1=vmax, option='II', prop=True)
    # plt.tight_layout(pad=0)
    plt.savefig(figureFolder + 'ncd_pid_prop.png', bbox_inches='tight', dpi=250)

#    # plot all the figures from GMII
    plot_pid(xv, yv, iic, rc, sc, uxc, uyc, xlabeltex, ylabeltex,
             vmin1=vmin, vmax1=vmax, option='MPID', prop=False)
    # plt.tight_layout(pad=0)
    plt.savefig(figureFolder + 'ncd_mpid.png', bbox_inches='tight', dpi=250)

    # plot all the figures from II
    plot_pid(xv, yv, ii, r, s, ux, uy, xlabeltex, ylabeltex,
             vmin1=vmin, vmax1=vmax, option='II', prop=False)
    # plt.tight_layout(pad=0)
    plt.savefig(figureFolder + 'ncd_pid.png', bbox_inches='tight', dpi=250)

#############################################
# ii for the linear common driver model #
# ii[x(t-1);y(t-1);z(t)|w(t-2)]              #
##############################################
# print 'linear common driver model...'
# # parameter settings
# ncwx, ncwy = 10, 10
# cwx_min, cwx_max = -2, 2
# cwy_min, cwy_max = -2, 2
# cwx_set = np.linspace(cwx_min, cwx_max, ncwx)
# cwy_set = np.linspace(cwy_min, cwy_max, ncwy)
# cxz, cyz = .5, .5
# sw, sx, sy, sz = 1., 1., 1., 1.

# rc, r = np.zeros([ncwx, ncwy]), np.zeros([ncwx, ncwy])
# sc, s = np.zeros([ncwx, ncwy]), np.zeros([ncwx, ncwy])
# miit, ii = np.zeros([ncwx, ncwy]), np.zeros([ncwx, ncwy])
# uxc, ux = np.zeros([ncwx, ncwy]), np.zeros([ncwx, ncwy])
# uyc, uy = np.zeros([ncwx, ncwy]), np.zeros([ncwx, ncwy])
# itc, it = np.zeros([ncwx, ncwy]), np.zeros([ncwx, ncwy])
# iic, ii = np.zeros([ncwx, ncwy]), np.zeros([ncwx, ncwy])

# for i in range(ncwx):
#     cwx = cwx_set[i]
#     for j in range(ncwy):
#         cwy = cwy_set[j]
#         # simulation
#         w, x, y, z = common_driver_linear(n, cwx, cwy, cxz, cyz, sw, sy, sx, sz)
#         w = w[:-2]
#         x = x[1:-1]
#         y = y[1:-1]
#         z = z[2:]
#         data1 = np.array([x, y, z, w]).t
#         data2 = np.array([x, y, z]).t

#         # compute the pdf
#         # miit
#         pdfsolver = pdfcomputer(ndim='m', approach=approach, bandwidth='silverman')
#         t1, pdf1, coords1 = pdfsolver.computepdf(data1, nbins=[nx, ny, nz, nw])
#         # print t1
#         # ii
#         pdfsolver = pdfcomputer(ndim='m', approach=approach, bandwidth='silverman')
#         t2, pdf2, coords2 = pdfsolver.computepdf(data2, nbins=[nx, ny, nz])
#         # print t2

#         # compute pid
#         # miit
#         pid1 = info(pdf1)
#         # ii
#         pid2 = info(pdf2)

#         # get sur values
#         rc[i,j], r[i,j]    = pid1.r, pid2.r
#         sc[i,j], s[i,j]    = pid1.s, pid2.s
#         miit[i,j], ii[i,j] = pid1.ii, pid2.ii
#         uxc[i,j], ux[i,j]  = pid1.uxz, pid2.uxz
#         uyc[i,j], uy[i,j]  = pid1.uyz, pid2.uyz
#         itc[i,j], it[i,j]  = pid1.itot, pid2.itot
#         iic[i,j], ii[i,j]  = pid1.ii, pid2.ii

# # plot
# xlabel, ylabel = 'cwx', 'cwy'
# extent = [cwx_min, cwx_max, cwy_max, cwy_min]
# xv, yv = np.meshgrid(cwx_set, cwy_set, indexing='ij')
# SMALL_SIZE = 16
# MEDIUM_SIZE = 18
# BIGGER_SIZE = 22

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# if PLOT == 1:
#     # Plot interaction information and total information (comparison)
#     plot_ii(xv, yv, iic, ii, itc, it, xlabel, ylabel)
#     plt.savefig('lcd_interaction_info_comp.png', dpi=150)

#     # Plot SUR (w/ conditions, percentage)
#     plot_sur_prop(xv, yv, rc/itc, sc/itc, uxc/itc, uyc/itc, xlabel, ylabel, 'w/ cond\'t')
#     plt.savefig('lcd_SUR_MIIT.png', dpi=150)

#     # Plot SUR (w/o conditions, percentage)
#     plot_sur_prop(xv, yv, r/it, s/it, ux/it, uy/it, xlabel, ylabel, 'w/o cond\'t')
#     plt.savefig('lcd_SUR_II.png', dpi=150)

#     # Plot SR comparison
#     plot_sr_comparison3d(xv, yv, rc, r, sc, s, xlabel, ylabel)
#     plt.savefig('lcd_SR_info_comparison.png', dpi=150)

#     # Plot the proportion of SR in the total information
#     plot_sr_prop_comparison3d(xv, yv, rc/itc, r/it, sc/itc, s/itc, xlabel, ylabel)
#     plt.savefig('lcd_Prop_SR_info_comparison.png', dpi=150)


# #################################
# # Test on a simple linear model #
# #################################

# # Parameter settings
# cwx, cwy, cxz, cyz = -10., -10., .5, .5
# sw, sx, sy, sz = 1., 1., 1., 1.
# nx, ny, nz, nw = [10, 10, 10, 10]

# # Simulation
# w, x, y, z = common_driver_linear(N, cwx, cwy, cxz, cyz, sw, sy, sx, sz)
# w = w[:-2]
# x = x[1:-1]
# y = y[1:-1]
# z = z[2:]
# data1 = np.array([x, y, z, w]).T
# data2 = np.array([x, y, z]).T

# # Compute the PDF
# # MIIT
# pdfSolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
# t1, pdf1, coords1 = pdfSolver.computePDF(data1, nbins=[nx, ny, nz, nw])
# # II
# pdfSolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
# t2, pdf2, coords2 = pdfSolver.computePDF(data2, nbins=[nx, ny, nz])


# # Compute PID
# # MIIT
# pid1 = info(pdf1)
# print '----------MIIT---------'
# print pid1.ii, pid1.itot
# print pid1.ixy_w, pid1.hxw, pid1.hyw, pid1.isource, pid1.ixz_w, pid1.iyz_w
# print pid1.hw, pid1.hxw, pid1.hyw, pid1.hw, pid1.hxyw
# print pid1.hx, pid1.hy, pid1.hz
# print pid1.ixy, pid1.iyz, pid1.ixz
# # II
# pid2 = info(pdf2)
# print '----------II---------'
# print pid2.ii, pid2.itot
# print pid2.ixy, pid2.hx, pid2.hy, pid2.isource, pid2.iyz, pid2.ixz
# print pid2.hx, pid2.hy, pid2.hz
# print pid2.ixy, pid2.iyz, pid2.ixz

# print '---------ixy---------'
# data3 = np.array([x, y]).T
# pdfSolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
# t3, pdf3, coords3 = pdfSolver.computePDF(data3, nbins=[nx, ny])
# pid3 = info(pdf3)
# print pid3.ixy, pid3.hx, pid3.hy

# pdfxy1 = np.sum(pdf1, axis=(2,3))
# pdfxy2 = np.sum(pdf2, axis=(2))

# plt.figure(figsize=(10, 24))
# gs = gridspec.GridSpec(3, 1)
# gs.update(wspace=.4, hspace=.4)

# ax = plt.subplot(gs[0, 0])
# cax = ax.imshow(pdfxy2, interpolation='bilinear', cmap=plt.get_cmap('jet'))
# plt.colorbar(cax, ax=ax)
# # ax.scatter(x, z)
# ax = plt.subplot(gs[1, 0])
# cax = ax.imshow(pdf3, interpolation='bilinear', cmap=plt.get_cmap('jet'))
# plt.colorbar(cax, ax=ax)
# # ax.scatter(y, z)
# ax = plt.subplot(gs[2, 0])
# cax = ax.scatter(x, y)
# # cax = ax.imshow(pdfxy1-pdfxy2, interpolation='bilinear', cmap=plt.get_cmap('jet'))
# # plt.colorbar(cax, ax=ax)
# plt.savefig('Test.png')
