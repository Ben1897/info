'''
This is a test of computing MIIT and the conditional PID for the nonlinear common driver model

Author: Peishi Jiang
Date: 2017.5.9
'''

import numpy as np
from info.core.info import info
from info.utils.pdf_computer import pdfComputer
from info.models.others import common_driver_linear, common_driver_nonlinear, Lorenze_model


N = 10000
approach = 'kde_c'

##############################################
# MIIT for the nonlinear common driver model #
# II[X(t-1);Y(t-1);Z(t)|W(t-2)]              #
##############################################

# Parameter settings
cwx, cwy, cz = 1.9, 1.9, .5
sw, sx, sy, sz = 1., 1., 1., 1.
nx, ny, nz, nw = [10, 10, 10, 10]

# Simulation
w, x, y, z = common_driver_nonlinear(N, cwx, cwy, cz, sw, sy, sx, sz)
w = w[:-2]
x = x[1:-1]
y = y[1:-1]
z = z[2:]
data1 = np.array([x, y, z, w]).T
data2 = np.array([x, y, z]).T

# Compute the PDF
# MIIT
pdfSolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
t1, pdf1, coords1 = pdfSolver.computePDF(data1, nbins=[nx, ny, nz, nw])
print t1
# II
pdfSolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
t2, pdf2, coords2 = pdfSolver.computePDF(data2, nbins=[nx, ny, nz])
print t2

# Compute PID
# MIIT
pid1 = info(pdf1)
# II
pid2 = info(pdf2)

print pid1.allInfo
print pid2.allInfo


# ##############################################
# # II for the linear common driver model #
# # II[X(t-1);Y(t-1);Z(t)|W(t-2)]              #
# ##############################################
# # Parameter settings
# cwx, cwy, cxz, cyz = 1.9, 1.9, .5, .5
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
# print t1
# # II
# pdfSolver = pdfComputer(ndim='m', approach=approach, bandwidth='silverman')
# t2, pdf2, coords2 = pdfSolver.computePDF(data2, nbins=[nx, ny, nz])
# print t2

# # Compute PID
# # MIIT
# pid1 = info(pdf1)
# # II
# pid2 = info(pdf2)

# print pid1.allInfo
# print pid2.allInfo
