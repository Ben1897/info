"""
This file is used for generating the preliminary DAG time-series graph by using Tigramite package, and is modified from:
https://github.com/jakobrunge/tigramite_old/blob/master/tigramite_analysis_ensemble_parallel.py.

Case: raw stream chemistry data.
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# TiGraMITe -- Time Series Graph Based Measures of Information Transfer
#
# Methods are described in:
#    J. Runge et al., Nature Communications, 6, 8502 (2015)
#    J. Runge, J. Heitzig, V. Petoukhov, and J. Kurths,
#       Phys. Rev. Lett. 108, 258701 (2012)
#    J. Runge, J. Heitzig, N. Marwan, and J. Kurths,
#       Phys. Rev. E 86, 061121 (2012)
#    J. Runge, V. Petoukhov, and J. Kurths, Journal of Climate, 27.2 (2014)
#
# Please cite all references when using the method.
#
# Copyright (C) 2012-2016 Jakob Runge <jakobrunge@posteo.de>
# https://github.com/jakobrunge/tigramite.git
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Script to estimate time series graph and measures like MIT, ITY...
"""

#
#  Import essential tigramite modules
#
from tigramite_src import tigramite_preprocessing as pp
# import tigramite_preprocessing_geo as ppgeo

from tigramite_src import tigramite_estimation_beta as tigramite_estimation
from tigramite_src import tigramite_plotting

# import Parallel module (based on mpi4py)
from tigramite_src import mpi

# Import modules for data processing
import sys
sys.path.append('../..')

# from stream_analysis_Kirchner.parse_excel_data import parse_excel_data
from info.utils.others import butter_filter, aggregate, normalize
from utils import parse_excel_data

#  Import NumPy for the array object and fast numerics
import numpy
import pandas as pd
import matplotlib.pyplot as plt

from time import time

# import file handling packages
import os
import sys
import pickle

###
# Which operations to perform
###

# Estimate parents/neighbors and lag funcdionts;
# If False, these will be taken from results dict with name:
# os.path.expanduser(save_folder) + project_name + '_results.pkl'

###
# Some parameters used in different steps of the script
###

##############
# parameters #
##############
# file info
save_folder   = "./networks/"
figure_folder = save_folder
verbosity     = 3

# Porject name
project_name = 'stream_cmiknn_parallel_instream_strict_7var_raw_anomaly_taumax5'

fn    = "./sd01.xlsx"

snraw  = "7hour edited data"
var_names = ["log_flow", "Na mg/l", "Cl mg/l", "Al ug/l", "Ca mg/l", "SO4 mg/l", "pH"]


# knn parameters
k1 = 5
k2 = 100
tau_max = 5
ntest = 100
alpha = 0.05
sig_method = 'full_shuffle'
ensemble_number = 1

initial_conds = 2
max_conds = 10
max_trials = 20

#################
# load the data #
#################
dfraw = pd.read_excel(fn, sheetname=snraw, header=0, index_col="date_time")
dfcon = dfraw[dfraw['Site'] == 'UHF'][var_names]

#################
# normalization #
#################
data = normalize(dfcon.values)

###
# Data preparation: provide numpy arrays "fulldata" (float) and
# "sample_selector" (bool), both of shape
# (Time, Variables)
# and datatime (float array) of shape (Time,)
###
# Generate the outputs & Ensembel
data_final = data
T, N = data_final.shape
fulldata_list = [data_final for i in range(ensemble_number)]

###
# Possibly supply mask as a boolean array. Samples with a "0" are masked out.
# The variable sample_selector needs to be of the same shape as fulldata.
###

sample_selector_list = [numpy.ones(data.shape).astype('bool')
                        for data in fulldata_list]
# sample_selector[fulldata < -3] = False        # example of masking by value

##
# Possibly construct symbolic time series for use with measure = 'symb'
##

# (fulldata, sample_selector, T) = pp.ordinal_patt_array(
#                                   fulldata, sample_selector,
#                                   dim=2, step=1, verbosity=0)

# fulldata = pp.quantile_bin_array(fulldata, bins = 3)
# print fulldata

##
# Define time sequence (only used for plotting)
##
datatime_list = [numpy.arange(0, data.shape[0], 1.)
                 for data in fulldata_list]


# Initialize results dictionary with important variables that are used
# in different analysis steps and should be saved to the results dictionary.
# All less important (eg plotting) parameters can be local...

d = {
    # Data
    'fulldata': fulldata_list,
    'rawdata': dfcon,
    'N': fulldata_list[0].shape[1],
    'T': [data.shape[0] for data in fulldata_list],
    'datatime': datatime_list,

    # Analyze only masked samples
    # selector_type needs to be a list containing 'x' or 'y'or 'z' or any
    # combination. This will ignore masked values if they are in the
    # lagged variable X, the 'driven' variable Y and/or the condition Z in
    # the association measure I(X;Y | Z), which enables to, e.g., only
    # consider the impacton summer months. More use cases will bediscussed
    # in future papers...
    'selector': False,
    'sample_selector': sample_selector_list,
    'selector_type': ['y'],

    # Measure of association and params
    # - 'par_corr': linear partial correlation,
    # - 'reg': linear standardized partial regression
    # - 'cmi_knn': conditional mutual information (CMI)
    #   estimated using nearest neighbors
    # - 'cmi_symb': CMI using symbolic time series
    #   (from binning or ordinal patterns, the data
    #   can be converted using the functions in
    #   the module "pp")
    'measure': 'cmi_knn',

    # Quantities to estimate using estimated parents
    # 'none' for MI/cross correlation
    # 'parents_xy' for MIT
    # 'parents_y' for ITY
    # 'parents_x' for ITX
    # These measures are described in Runge et al. PRE (2012).
    'cond_types': ['none', 'parents_xy'],

    # for measure='cmi_knn': nearest neighbor parameter
    # used in causal algorithm (higher k reduce the
    # variance of the estimator, better for
    # independence tests).
    # Recommended: 5% - 50% of time series length
    'measure_params_algo': {'knn': k2},

    # nearest neighbor parameter used in the
    # subsequent estimation of MI, MIT, ITY, ...
    # (smaller k has smaller bias)
    # Recommended: 5..10, independent of T
    'measure_params_lagfuncs': {'knn': k1},

    # Causal Algorithm for estimation of parents/neighbors
    # Maximum time lag up to which links are tested
    'tau_max': tau_max,

    # Initial number of conditions to use, corresponds
    # to n_0 in Runge et al. PRL (2012).
    # Larger initial_conds speeds up the algorithm,
    # but leads to slightly more false positives.
    'initial_conds': initial_conds,

    # Maximum number of conditions to use
    # (parameter n_max in my phd thesis)
    # Recommended: 4..6  for CMI estimation,
    # for 'par_corr' or 'reg' more can be used.
    'max_conds': max_conds,

    # Maximum number of combinations of conditions
    # to check in algorithm (corresponds to number i
    # of iterations in n-loop in Runge PRL (2012))
    # Recommended: 3..6
    'max_trials': max_trials,

    # True for solid links as defined in Runge PRL + PRE (2012)
    # Recommended is  "True".
    'solid_contemp_links': True,

    # Significance testing in algorithm and lag functions estimation
    # - 'fixed': fixed threshold (specified below)
    # - 'analytic': sig_lev for analytical sample
    #  distribution of partial correlation or
    #  regression (Student's t)
    # - 'full_shuffle': shuffle test as described in
    #   Runge et al. PRL (2012)
    # - 'block_shuffle': block shuffle test works better for serially
    #   dependent data. Block length determined using approach in Mader (2013)
    #   [Eq. (6)]
    # Recommended for CMI: 'full_shuffle' or 'fixed'
    # Recommended for par_corr and reg: 'analytic'
    'significance': sig_method,

    # significance level (1-alpha). Note that for
    # 'par_corr' or 'reg' the test is two-sided,
    # such that 0.95 actually corresponds to a 90%
    # significance level
    # Here the divisor "/ 2." accounts for a two-sided level
    # 'sig_lev': (1. - .1 / 2.),
    'sig_lev': (1. - alpha),

    # Higher significance levels require a larger
    # number of shuffle test samples, i.e. 0.9 needs
    # about 50 samples, 0.95 about 100, .98 about 500.
    'sig_samples': ntest,

    # fixed threshold for CMI. I recommend to use a
    # shuffle test for CMI to get an idea of typical
    # values (see output in command line). Note that
    # shuffle significance thresholds depend on the
    # estimation dimension.
    'fixed_thres': 0.05,

    # Confidence bounds to be displayed
    # in lag functions (not used in the algorithm)
    # - False: no bounds
    # - 'analytic': conf_lev for analytical sample
    #  distribution (Student's t)
    # - 'bootstrap': bootstrap confidence bounds
    # Recommended for CMI: 'bootstrap'
    # Recommended for par_corr and reg: 'analytic'
    'confidence': False,

    # 0.9 corresponds to 90% confidence interval.
    'conf_lev': .95,
    'conf_samples': 100,

    # Variable names and node positions for graph plots (in figure coords)
    # These can be adapted to basemap plots in plot section below
    # 'var_names': ['0', '1', '2', '3'],
    'var_names': var_names,
    'node_pos': None,
               # {'y': numpy.array([0.5,  1.,  0., 0.5]),
               # 'x': numpy.array([0., 0.5,  0.5, 1.])},
}


###
# Space for operations on the data using functions in tigramite modules
###

def master():


    ###
    # Estimate parents and neighbors
    ###

    ensemble_members = range(len(fulldata_list))
    print ''
    print 'Start the project ' + project_name + '...'
    start = time()

    d['ensemble_members'] = ensemble_members

    for ens in ensemble_members:

        print ens

        tigramite_estimation._sanity_checks(
            which='pc_algo',
            data=d['fulldata'][ens],
            selector=d['selector'],
            selector_type=d['selector_type'],
            sample_selector=d['sample_selector'][ens],

            measure=d['measure'],
            measure_params=d['measure_params_algo'],

            estimate_parents_neighbors='both',
            tau_max=d['tau_max'],
            initial_conds=d['initial_conds'],
            max_conds=d['max_conds'],
            max_trials=d['max_trials'],
            significance=d['significance'],
            sig_lev=d['sig_lev'],
            fixed_thres=d['fixed_thres'],

            verbosity=verbosity)

    d['results'] = {}

    if verbosity > 0:
        print("\n" + "-" * 60 +
              "\nEstimating parents for all variables:"
              "\n" + "-" * 60)

    job_index = 0
    for ens in ensemble_members:
        for j in range(d['N']):

            mpi.submit_call(
                "tigramite_estimation._pc_algo",
                kwargs={
                    'data': d['fulldata'][ens],
                    'j': j,
                    'parents_or_neighbors': 'parents',
                    'all_parents': None,
                    'tau_max': d['tau_max'],
                    'initial_conds': d['initial_conds'],
                    'max_conds': d['max_conds'],
                    'max_trials': d['max_trials'],
                    'measure': d['measure'],
                    'measure_params': d['measure_params_algo'],
                    'significance': d['significance'],
                    'sig_lev': d['sig_lev'],
                    'fixed_thres': d['fixed_thres'],
                    'sig_samples': d['sig_samples'],
                    'selector': d['selector'],
                    'selector_type': d['selector_type'],
                    'sample_selector': d['sample_selector'][ens],
                    'verbosity': verbosity
                },
                id=job_index)
            job_index += 1

    job_index = 0
    for ens in ensemble_members:
        d['results'][ens] = {'parents_neighbors': {}}
        for j in range(d['N']):
            d['results'][ens]['parents_neighbors'][j] = \
                mpi.get_result(id=job_index)
            job_index += 1

    if d['solid_contemp_links']:
        if verbosity > 0:
            print("\n" + "-" * 60 +
                  "\nEstimating neighbors for all variables:"
                  "\n" + "-" * 60)

        job_index = 0
        for ens in ensemble_members:
            for j in range(d['N']):

                mpi.submit_call(
                    "tigramite_estimation._pc_algo",
                    kwargs={
                        'data': d['fulldata'][ens],
                        'j': j,
                        'parents_or_neighbors': 'neighbors',
                        'all_parents': d['results'][ens]['parents_neighbors'],
                        'tau_max': d['tau_max'],
                        'initial_conds': d['initial_conds'],
                        'max_conds': d['max_conds'],
                        'max_trials': d['max_trials'],
                        'measure': d['measure'],
                        'measure_params': d['measure_params_algo'],
                        'significance': d['significance'],
                        'sig_lev': d['sig_lev'],
                        'fixed_thres': d['fixed_thres'],
                        'sig_samples': d['sig_samples'],
                        'selector': d['selector'],
                        'selector_type': d['selector_type'],
                        'sample_selector': d['sample_selector'][ens],
                        'verbosity': verbosity
                    },
                    id=job_index)
                job_index += 1

        job_index = 0
        for ens in ensemble_members:
            for j in range(d['N']):
                d['results'][ens]['parents_neighbors'][j] += \
                    mpi.get_result(id=job_index)
                job_index += 1

    ###
    # Estimate lag functions for MIT, ITY, ...
    ###

    # 'none' for MI/cross correlation
    # 'parents_xy' for MIT
    # 'parents_y' for ITY
    # 'parents_x' for ITX
    # These measures are described in Runge et al. PRE (2012).
    # cond_types = ['none', 'parents_xy']
    # d['cond_types'] = cond_types

    if verbosity > 0:
        print("\n" + "-" * 60 +
              "\nEstimating lag functions for all variables:"
              "\n" + "-" * 60)

    job_index = 0
    for ens in ensemble_members:
        for which in d['cond_types']:

            # if verbosity > 0:
            #     print("Estimating lag functions for ensemble "
            #           "member %d and condition type %s" % (ens, which))

            for j in range(d['N']):

                mpi.submit_call(
                    "tigramite_estimation.get_lagfunctions",
                    kwargs={
                        'selected_variables': [j],
                        'data': d['fulldata'][ens],
                        'selector': d['selector'],
                        'selector_type': d['selector_type'],
                        'sample_selector': d['sample_selector'][ens],
                        'parents_neighbors': d['results'][ens][
                            'parents_neighbors'],
                        'cond_mode': which,
                        'solid_contemp_links': d['solid_contemp_links'],
                        'tau_max': d['tau_max'],
                        'max_conds': d['max_conds'],
                        'measure': d['measure'],
                        'measure_params': d['measure_params_lagfuncs'],

                        'significance': d['significance'],
                        'sig_lev': d['sig_lev'],
                        'sig_samples': d['sig_samples'],
                        'fixed_thres': d['fixed_thres'],

                        'confidence': d['confidence'],
                        'conf_lev': d['conf_lev'],
                        'conf_samples': d['conf_samples'],

                        'verbosity': verbosity
                    },
                    id=job_index)

                job_index += 1

    job_index = 0
    for ens in ensemble_members:
        for which in d['cond_types']:

            d['results'][ens][which] = numpy.zeros(
                (d['N'], d['N'], d['tau_max'] + 1))
            d['results'][ens]['sig_thres_' + which] = numpy.zeros(
                (d['N'], d['N'], d['tau_max'] + 1))
            d['results'][ens]['conf_' + which] = numpy.zeros(
                (d['N'], d['N'], d['tau_max'] + 1, 2))

            for j in range(d['N']):

                res = mpi.get_result(id=job_index)

                (d['results'][ens][which][:, j, :],
                 d['results'][ens]['sig_thres_' + which][:, j, :],
                 d['results'][ens]['conf_' + which][:, j, :]
                 ) = res[0][:, j, :], res[1][:, j, :], res[2][:, j, :]

                job_index += 1

    if verbosity > 0:
        print("Saving results as %s" % (os.path.expanduser(save_folder) +
                                        project_name +
                                        '_results.pkl'))
        print d['results'][0]['parents_neighbors']

    print time() - start

    pickle.dump(d, open(os.path.expanduser(save_folder) + project_name +
                        '_results.pkl', 'w'))


mpi.run(verbose=False)
