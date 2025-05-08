#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs MCMCs to the spectral light curve channels of JWST/NIRSpec G395H transmission spectra using emcee.

INPUT
----------
spec_lc : dictionary
    Dictionary containing the spectral light curve channels
broad_fit : lmfit results object
    Results for the fit parameters of the broad band fit.
prelim_fits: lmfit results object
    Result of the preliminary least squares fit with free limb darkening coefficients of the spectral light curve channels.
trans_spec: dictionary
    Preliminary transmission spectrum for finding typical uncertainty values for the fit parameters

OUTPUT
-------
mcmc_results : dictionaries
    Contains the flattened chains of the mcmcs as well as a lists of fit parameters and the acceptance fractions of all walkers.
mcmc_params : dictionaries
    Fit parameters' medians, errors etc.
trans_spec : dictionary
    Dictionary containing the results for the fit parameters of all spectral light curve channels.

"""

import os, dynesty, sys, h5py
from dynesty import plotting as dyplot
import numpy as np
import Utils, Plot
import parameters as pr
import matplotlib.pyplot as plt

#%% COMPARISON TRANSLATIONAL & ROTATIONAL MODEL

px = '10'

# Set directories
rot_dir = os.path.join( pr.ASTRONODE_DIR, '250317' )
trans_dir = os.path.join( pr.WORK_DIR, '240805', px + '_pixel_columns', pr.TRANS_DIR, pr.MCMC_DIR )

# Input
rot_spec = Utils.pickle_read( os.path.join(rot_dir, px + 'px_lc-fit') )
trans_spec = Utils.pickle_read( os.path.join(trans_dir, px + 'px_lc-fit') )

rot_spec['RpRs'] = rot_spec['R0']

# Plot.planet_rotation( rot_spec, opath = os.path.join(rot_dir, 'rotational_spectrum') )
Plot.Transmission_Spectrum_2runs_errorbars_residuals( {'Translational model' : trans_spec, 'Rotational model' : rot_spec}, \
                                                      transit_depth_unit=1.0E-2, transit_depth_limits=[1.4, 1.65], uncertainty_unit=1.0E-6, \
                                                          uncertainty_limits=[50.,800.], residual_limits=[-100., 100.], opath = os.path.join(rot_dir, 'transmission_spectrum') )

#%% PLOT ROTATIONAL SIGNAL

# Data
trans_folders = ['250429_linear', '250429_quadratic_ldc-priors', '250429_quadratic']
trans_labels = ['$R_2/R_*=0$, Uniform priors on LDCs', '$R_2/R_*$ free, Gaussian priors on LDCs', '$R_2/R_*$ free, Uniform priors on LDCs']
model_files = ['output_nominal']#, 'output_co_tweak', 'output_t_tweak']
model_labels = ['SPARC/MITgcm', 'CO removed from morning', 'Temperature tweak']

trans_spec = {}
for i, trans_folder in enumerate(trans_folders):
    trans_dir = os.path.join( pr.WORK_DIR, trans_folder, 'spec', pr.TRANS_DIR, pr.MCMC_DIR)
    trans_spec[trans_labels[i]] = Utils.pickle_read( os.path.join(trans_dir, pr.TRANS_FILE ) )

models = {}
for i, model_file in enumerate(model_files):
    f=h5py.File(os.path.join(pr.WORK_DIR, "aurelien/" + model_file + ".h5") )
    time = f["Model/lightcurve/times"][...]
    rprs = np.sqrt(f["Output/lightcurve/flux"][...])
    wvls = 10000./f["Output/lightcurve/wns"][...]
    
    # Trim from T2 to T3
    T23 = 2.25*3600.
    time, rprs = Utils.trim_spectrum( time, rprs, -T23/2., T23/2. )
    
    models[model_labels[i]] = Utils.twoD_lc2coefficients( time, rprs, wvls )

Plot.planet_rotation_nruns( trans_spec, models=models, opath = os.path.join(trans_dir, 'rotational_spectrum') )

# #%% VARIATION WITHIN NRS2

# rot_dir = os.path.join( pr.ASTRONODE_DIR, '250317' )
# rot_spec = Utils.pickle_read( os.path.join(rot_dir, '10px_lc-fit') )

# # Cut out NRS2
# i_cut = np.argmin( np.abs(rot_spec['wavelength'] - 3.72) ) + 1
# x = rot_spec['wavelength'][i_cut:]
# y = rot_spec['R1']['value'][i_cut:]
# yerr = rot_spec['R1']['stderr'][i_cut:]
# x_min = np.min(x)
# x_max = np.max(x)

# plt.clf()
# plt.errorbar( x, y, yerr, ls='none', marker='o', markerfacecolor='white', zorder=1 )

# # This is the function we want to maximize
# def log_likelihood( pars ):
#     pars_dict = Utils.arrays2dict( pars, varied_parameters )
    
#     if 'c' in varied_parameters:
#         model = pars_dict['c']
#         if 'c1' in varied_parameters:
#             model += pars_dict['c1'] * (x-x_min)
            
#     else:
#         model = np.zeros(len(x))
#         for i in range(len(model)):
#             if x[i] < 4.3:
#                 model[i] = pars_dict['lower']
#             else:
#                 model[i] = pars_dict['upper']
#     if 'sigma_sys' in varied_parameters:
#         summand = (y-model)**2 / (yerr**2 + pars_dict['sigma_sys']**2) + np.log( 2. * np.pi * (yerr**2 + pars_dict['sigma_sys']**2) )
#     else:
#         summand = (y-model)**2 / yerr**2 + np.log( 2. * np.pi * yerr**2 )
    
#     return -0.5*np.sum( summand )


# print('Constant fit')
# # varied_parameters = ['c', 'sigma_sys']
# # priors = { 'c': {'lower_limit' : 0., 'upper_limit' : 0.05}, \
# #           'sigma_sys': {'lower_limit' : 0., 'upper_limit' : 0.05} }
# varied_parameters = ['c']
# priors = { 'c': {'lower_limit' : -0.1, 'upper_limit' : 0.1} }
# ndim = len(varied_parameters)

# # "Static" nested sampling.
# sampler = dynesty.NestedSampler( log_likelihood, Utils.prior_transform, ndim, ptform_args=[priors], nlive=pr.LIVE_POINTS )
# sampler.run_nested()

# results = sampler.results
# samples = results.samples_equal()

# results.summary()
# medians = Utils.samples2medians(samples, varied_parameters)
# plt.hlines( medians['c'], np.min(x), np.max(x), color='red', zorder=2 )

# print('Linear fit')
# # varied_parameters = ['c', 'c1', 'sigma_sys']
# # priors = { 'c': {'lower_limit' : 0., 'upper_limit' : 0.05}, \
# #            'c1': {'lower_limit' : -0.05, 'upper_limit' : 0.05}, \
# #            'sigma_sys': {'lower_limit' : 0., 'upper_limit' : 0.05} }
# varied_parameters = ['c', 'c1']
# priors = { 'c': {'lower_limit' : -0.1, 'upper_limit' : 0.1}, \
#             'c1': {'lower_limit' : -0.1, 'upper_limit' : 0.1} }
# ndim = len(varied_parameters)

# # "Static" nested sampling.
# sampler = dynesty.NestedSampler( log_likelihood, Utils.prior_transform, ndim, ptform_args=[priors], nlive=pr.LIVE_POINTS )
# sampler.run_nested()

# results = sampler.results
# samples = results.samples_equal()
# # cfig, caxes = dyplot.cornerplot(results, labels=varied_parameters)
# # cfig.show()
# results.summary()
# medians = Utils.samples2medians(samples, varied_parameters)
# plt.plot( x, medians['c'] + medians['c1']*(x-x_min), color='orange', zorder=2 )

# print('Step function')
# step = 4.3
# # varied_parameters = ['lower', 'upper', 'sigma_sys']
# # priors = { 'lower': {'lower_limit' : 0., 'upper_limit' : 0.05}, \
# #            'upper': {'lower_limit' : 0., 'upper_limit' : 0.05}, \
# #            # 'step': {'lower_limit' : x_min, 'upper_limit' : x_max}, \
# #            'sigma_sys': {'lower_limit' : 0., 'upper_limit' : 0.05} }
# varied_parameters = ['lower', 'upper']
# priors = { 'lower': {'lower_limit' : -0.1, 'upper_limit' : 0.1}, \
#             # 'step': {'lower_limit' : x_min, 'upper_limit' : x_max}, \
#             'upper': {'lower_limit' : -0.1, 'upper_limit' : 0.1} }
# ndim = len(varied_parameters)

# # "Static" nested sampling.
# sampler = dynesty.NestedSampler( log_likelihood, Utils.prior_transform, ndim, ptform_args=[priors], nlive=pr.LIVE_POINTS )
# sampler.run_nested()

# results = sampler.results
# samples = results.samples_equal()
# # cfig, caxes = dyplot.cornerplot(results, labels=varied_parameters)
# # cfig.show()
# results.summary()
# medians = Utils.samples2medians(samples, varied_parameters)
# plt.hlines( medians['lower'], x_min, step, color='green', zorder=2 )
# plt.hlines( medians['upper'], step, x_max, color='green', zorder=2 )

# plt.ylim( -0.05, 0.1 )
# plt.xlabel('Wavelength ($\mu$m)')
# plt.ylabel('$R_1$ [days$^{(-1)}$]')
# plt.savefig( os.path.join( rot_dir, 'nrs2.pdf') )
# plt.show()
# plt.close()

# #%%
# print('Bayes factor, sigma significance')
# B = np.exp(9638.968-9636.367)
# print( B, Utils.bayes2Nsigma(B))
