import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import entropy, multivariate_normal
# import np.random.multivariate_normal as mnormal
from info.utils.pdf_computer import pdfComputer
from info.core.info import info


ndim = 'm'
approach = 'kde_c'
base = 2
pdfsolver = pdfComputer(ndim, approach=approach, bandwidth='silverman')

##########################
# 2D normal distribution #
##########################
n = 10000
nbins = [10, 12]
mean = [1, 2]
cov = [[1, .5], [.5, 1]]

# Generate the data and calculate the sampled PDF
data = np.random.multivariate_normal(mean, cov, n)
t, pdf, coord = pdfsolver.computePDF(data, nbins)
print pdf.shape
dx = coord[0][1] - coord[0][0]
dy = coord[1][1] - coord[1][0]

#Calculate the true pdf
rv = multivariate_normal(mean=mean, cov=cov)
x, y = np.meshgrid(coord[0], coord[1], indexing='ij')
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
pdf_t = rv.pdf(pos)

# Calculate the empirical joint entropy and the theoretical entropy
info2d = info(pdf, base=base)
He1 = info2d.hxy + info2d.hy
logpdf = np.ma.log(pdf) / np.log(base); logpdf = logpdf.filled(0)
He2 = -np.sum(pdf*logpdf)
print He1, He2

# Plot
fig, axes = plt.subplots(1, 2)
cax1 = axes[0].imshow(pdf, interpolation='bilinear', cmap=plt.get_cmap('jet'))
fig.colorbar(cax1, ax=axes[0])
# plt.grid(False)
cax2 = axes[1].imshow(pdf_t, interpolation='bilinear', cmap=plt.get_cmap('jet'))
fig.colorbar(cax2, ax=axes[1])
# plt.grid(False)
plt.show()


##########################
# 4D normal distribution #
##########################
nbins = [20, 20, 20, 20]
