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
from scipy.stats import binned_statistic

#%% PLOT ROTATIONAL SIGNAL

# Data
trans_folders = ['250429_linear', '250429_quadratic', '250429_quadratic_ldc-priors']
trans_labels = ['$R_2/R_*=0$, Uniform priors on LDCs', '$R_2/R_*$ free, Uniform priors on LDCs', '$R_2/R_*$ free, Gaussian priors on LDCs']
lc = Utils.pickle_read( os.path.join(pr.WORK_DIR, trans_folders[0], 'spec', pr.LC_FILE) )

lc_keys = list(lc.keys())
ldcs = {'u1' : [], 'u1_err' : [], 'u2' : [], 'u2_err' : []}
for ldc in ldcs.keys():
    for lc_key in lc_keys:
        ldcs[ldc] += [lc[lc_key][ldc]]

# Models
model_files = ['output_nominal', 'output_t_tweak']#, 'output_T-modification_CtoO-modification']
model_labels = ['SPARC/MITgcm, nominal', 'SPARC/MITgcm, muted leading limb', 'SPARC/MITgcm, muted leading limb and C/O=0.92']

trans_spec = {}
for i, trans_folder in enumerate(trans_folders):
    trans_dir = os.path.join( pr.WORK_DIR, trans_folder, 'spec', pr.TRANS_DIR, 'all_' + pr.MCMC_DIR)
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
# Plot.all_fit_parameters( trans_spec[trans_labels[0]], LDcoefficients=ldcs, opath=os.path.join(pr.WORK_DIR, trans_folders[0], 'spec', pr.TRANS_DIR, '' + pr.MCMC_DIR, 'all_fit_params') )

# DATA: Transit depth
h2o_R0 = Utils.error_weighted_mean( trans_spec[trans_labels[0]]['R0']['value'][:6], trans_spec[trans_labels[0]]['R0']['stderr'][:6] )
co_R0 = Utils.error_weighted_mean( trans_spec[trans_labels[0]]['R0']['value'][-5:], trans_spec[trans_labels[0]]['R0']['stderr'][-5:] )
print('H2O:', h2o_R0)
print('CO:', co_R0)

# DATA: R1
conversion = 1.0E6/24.
h2o_R1 = Utils.error_weighted_mean( trans_spec[trans_labels[0]]['R1']['value'][:6]*conversion, trans_spec[trans_labels[0]]['R1']['stderr'][:6]*conversion )
h2o_R1_err = Utils.gaussian_error_propagation( trans_spec[trans_labels[0]]['R1']['stderr'][:6]*conversion )
co_R1 = Utils.error_weighted_mean( trans_spec[trans_labels[0]]['R1']['value'][-5:]*conversion, trans_spec[trans_labels[0]]['R1']['stderr'][-5:]*conversion )
co_R1_err = Utils.gaussian_error_propagation( trans_spec[trans_labels[0]]['R1']['stderr'][-5:]*conversion )
sio_R1 = Utils.error_weighted_mean( trans_spec[trans_labels[0]]['R1']['value'][7:9]*conversion, trans_spec[trans_labels[0]]['R1']['stderr'][7:9]*conversion )
sio_R1_err = Utils.gaussian_error_propagation( trans_spec[trans_labels[0]]['R1']['stderr'][7:9]*conversion )
print('H2O:', h2o_R1, '+/-', h2o_R1_err, 'ppm/hr')
print('CO:', co_R1, '+/-', co_R1_err, 'ppm/hr')
print('SiO:', sio_R1, '+/-', sio_R1_err, 'ppm/hr')

# MODEL
h2o_interval = [trans_spec[trans_labels[0]]['wavelength'][0] - trans_spec[trans_labels[0]]['wavelength_halfwidth'][0], trans_spec[trans_labels[0]]['wavelength'][5] + trans_spec[trans_labels[0]]['wavelength_halfwidth'][5]]
h2o_model, applied_bin_edges, indices = binned_statistic( models[model_labels[0]]['wavelength'], models[model_labels[0]]['R1']*conversion, bins=h2o_interval )
print('Model, H2O:', h2o_model[0], 'ppm/hr')

co_interval = [trans_spec[trans_labels[0]]['wavelength'][9] - trans_spec[trans_labels[0]]['wavelength_halfwidth'][9], trans_spec[trans_labels[0]]['wavelength'][13] + trans_spec[trans_labels[0]]['wavelength_halfwidth'][13]]
co_model, applied_bin_edges, indices = binned_statistic( models[model_labels[0]]['wavelength'], models[model_labels[0]]['R1']*conversion, bins=co_interval )
print('Model, CO:', co_model[0], 'ppm/hr')

sio_interval = [trans_spec[trans_labels[0]]['wavelength'][7] - trans_spec[trans_labels[0]]['wavelength_halfwidth'][7], trans_spec[trans_labels[0]]['wavelength'][8] + trans_spec[trans_labels[0]]['wavelength_halfwidth'][8]]
sio_model, applied_bin_edges, indices = binned_statistic( models[model_labels[0]]['wavelength'], models[model_labels[0]]['R1']*conversion, bins=sio_interval )
print('Model, SiO:', sio_model[0], 'ppm/hr')

# for i in range(len(broad_lc['NRS2']['bjd_tdb'])):
#     print((broad_lc['NRS2']['bjd_tdb'][i]-broad_lc['NRS1']['bjd_tdb'][i])*24.*3600.)

#%% PLOT DIFFERENT REDUCTIONS

# Data
trans_folders = ['250429_linear', '250729_linear']
trans_labels = ['Firefly', 'EUREKA!']

# Models
model_files = ['output_nominal', 'output_t_tweak']#, 'output_co_tweak']
model_labels = ['SPARC/MITgcm, nominal', 'SPARC/MITgcm, T modification', 'SPARC/MITgcm, CO modification']#'#C/O=0.92']

trans_spec = {}
for i, trans_folder in enumerate(trans_folders):
    if i == 0:
        trans_dir = os.path.join( pr.WORK_DIR, trans_folder, 'spec', pr.TRANS_DIR, 'all_' + pr.MCMC_DIR)
    else:
        trans_dir = os.path.join( pr.WORK_DIR, trans_folder, 'spec', pr.TRANS_DIR, '' + pr.MCMC_DIR)
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

Plot.planet_rotation_nruns( trans_spec, models=models, opath = os.path.join(trans_dir, 'nirspec_reductions_spectra') )
# Plot.all_fit_parameters( trans_spec[trans_labels[0]], LDcoefficients=ldcs, opath=os.path.join(pr.WORK_DIR, trans_folders[0], 'spec', pr.TRANS_DIR, '' + pr.MCMC_DIR, 'all_fit_params') )

# #%% ALL MODELS

# # Data
# trans_folders = ['250429_linear', '250429_quadratic_ldc-priors', '250429_quadratic']
# trans_labels = ['$R_2/R_*=0$, Uniform priors on LDCs', '$R_2/R_*$ free, Gaussian priors on LDCs', '$R_2/R_*$ free, Uniform priors on LDCs']
# lc = Utils.pickle_read( os.path.join(pr.WORK_DIR, trans_folders[0], 'spec', pr.LC_FILE) )

# lc_keys = list(lc.keys())
# ldcs = {'u1' : [], 'u1_err' : [], 'u2' : [], 'u2_err' : []}
# for ldc in ldcs.keys():
#     for lc_key in lc_keys:
#         ldcs[ldc] += [lc[lc_key][ldc]]

# # Models
# model_files = ['output_nominal']#, 'output_co_tweak', 'output_t_tweak']
# model_labels = ['SPARC/MITgcm', 'CO removed from morning', 'Temperature tweak']

# trans_spec = {}
# for i, trans_folder in enumerate(trans_folders):
#     trans_dir = os.path.join( pr.WORK_DIR, trans_folder, 'spec', pr.TRANS_DIR, 'all_' + pr.MCMC_DIR)
#     trans_spec[trans_labels[i]] = Utils.pickle_read( os.path.join(trans_dir, pr.TRANS_FILE ) )

# models = {}
# for i, model_file in enumerate(model_files):
#     f=h5py.File(os.path.join(pr.WORK_DIR, "aurelien/" + model_file + ".h5") )
#     time = f["Model/lightcurve/times"][...]
#     rprs = np.sqrt(f["Output/lightcurve/flux"][...])
#     wvls = 10000./f["Output/lightcurve/wns"][...]
    
#     # Trim from T2 to T3
#     T23 = 2.25*3600.
#     time, rprs = Utils.trim_spectrum( time, rprs, -T23/2., T23/2. )
    
#     models[model_labels[i]] = Utils.twoD_lc2coefficients( time, rprs, wvls )

# Plot.planet_rotation_nruns( trans_spec, models=models, opath = os.path.join(trans_dir, 'rotational_spectrum') )
# Plot.all_fit_parameters( trans_spec[trans_labels[2]], LDcoefficients=ldcs )


# # Data
# trans_folders = ['250429_linear']#, '250429_quadratic_ldc-priors', '250429_quadratic']
# trans_labels = ['$R_2/R_*=0$, Uniform priors on LDCs', '$R_2/R_*$ free, Gaussian priors on LDCs', '$R_2/R_*$ free, Uniform priors on LDCs']
# model_files = ['output_nominal', 'output_co_tweak', 'output_t_tweak']
# model_labels = ['Nominal SPARC/MITgcm', 'CO modification', 'Temperature modification']

# trans_spec = {}
# for i, trans_folder in enumerate(trans_folders):
#     trans_dir = os.path.join( pr.WORK_DIR, trans_folder, 'spec', pr.TRANS_DIR, 'all_' + pr.MCMC_DIR)
#     trans_spec[trans_labels[i]] = Utils.pickle_read( os.path.join(trans_dir, pr.TRANS_FILE ) )

# models = {}
# for i, model_file in enumerate(model_files):
#     f=h5py.File(os.path.join(pr.WORK_DIR, "aurelien/" + model_file + ".h5") )
#     time = f["Model/lightcurve/times"][...]
#     rprs = np.sqrt(f["Output/lightcurve/flux"][...])
#     wvls = 10000./f["Output/lightcurve/wns"][...]
    
#     # Trim from T2 to T3
#     T23 = 2.25*3600.
#     time, rprs = Utils.trim_spectrum( time, rprs, -T23/2., T23/2. )
    
#     models[model_labels[i]] = Utils.twoD_lc2coefficients( time, rprs, wvls )

# Plot.planet_rotation_nruns( trans_spec, models=models, opath = os.path.join(trans_dir, 'gcm_tweaks') )

# # wavelengths = models[model_labels[0]]['wavelength']
# # for i in range(len(wavelengths) - 1):
# #     print((wavelengths[i+1]-wavelengths[i])/(wavelengths[i+1]+wavelengths[i]))

#%% T2 AND T3 SPECTRA

T23 = 2.25/24.
print('Rotation during half a transit:', T23/2./pr.ORBITAL_PERIOD['median']*360.)

# Data
trans_dir = os.path.join( pr.WORK_DIR, '250429_linear', 'spec', pr.TRANS_DIR, 'all_' + pr.MCMC_DIR)
rot_trans_spec = Utils.pickle_read( os.path.join(trans_dir, pr.TRANS_FILE ) )
variables = list(rot_trans_spec.keys())

trans_spec = {}
contact_points = ['Contact point 2', 'Contact point 3']
signs = [-1., 1.]

for i, point in enumerate(contact_points):
    sign = signs[i]
    trans_spec[point] = {}
    for key in variables:
        trans_spec[point][key] = rot_trans_spec[key]
    trans_spec[point]['RpRs'] = {}
    trans_spec[point]['RpRs']['value'] = rot_trans_spec['R0']['value'] + sign * T23/2.*rot_trans_spec['R1']['value']
    trans_spec[point]['RpRs']['stderr'] = np.sqrt( rot_trans_spec['R0']['stderr']**2 + (T23/2.*rot_trans_spec['R1']['stderr'])**2 )

# Models
model_file = 'output_nominal'# output_t_tweak

models = {}
f=h5py.File(os.path.join(pr.WORK_DIR, "aurelien/" + model_file + ".h5") )
time = f["Model/lightcurve/times"][...]
rprs = np.sqrt(f["Output/lightcurve/flux"][...])
wvls = 10000./f["Output/lightcurve/wns"][...]
    
# Trim from T2 to T3
T23 = 2.25*3600.
time, rprs = Utils.trim_spectrum( time, rprs, -T23/2., T23/2. )

models['Contact point 2'] = {}
models['Contact point 2']['wavelength'] = wvls
models['Contact point 2']['transit_depth'] = rprs[0,:]**2

models['Contact point 3'] = {}
models['Contact point 3']['wavelength'] = wvls
models['Contact point 3']['transit_depth'] = rprs[-1,:]**2

Plot.Transmission_Spectrum_nruns_npanels( trans_spec, transit_depth_unit=1.0E-2, opath=os.path.join(trans_dir, 'trans_spec') )
# Plot.Transmission_Spectrum_nruns_meansubtracted_npanels( trans_spec, transit_depth_unit=1.0E-6 )

# #%% COMPARISON TRANSLATIONAL & ROTATIONAL MODEL

# px = '10'

# # Set directories
# rot_dir = os.path.join( pr.ASTRONODE_DIR, '250317' )
# trans_dir = os.path.join( pr.WORK_DIR, '240805', px + '_pixel_columns', pr.TRANS_DIR, pr.MCMC_DIR )

# # Input
# rot_spec = Utils.pickle_read( os.path.join(rot_dir, px + 'px_lc-fit') )
# trans_spec = Utils.pickle_read( os.path.join(trans_dir, px + 'px_lc-fit') )

# rot_spec['RpRs'] = rot_spec['R0']

# # Plot.planet_rotation( rot_spec, opath = os.path.join(rot_dir, 'rotational_spectrum') )
# Plot.Transmission_Spectrum_2runs_errorbars_residuals( {'Translational model' : trans_spec, 'Rotational model' : rot_spec}, \
#                                                       transit_depth_unit=1.0E-2, transit_depth_limits=[1.4, 1.65], uncertainty_unit=1.0E-6, \
#                                                           uncertainty_limits=[50.,800.], residual_limits=[-100., 100.], opath = os.path.join(rot_dir, 'transmission_spectrum') )

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
