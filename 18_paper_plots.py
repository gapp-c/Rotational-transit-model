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

import os, h5py
import numpy as np
import Utils, Plot
import parameters as pr
import matplotlib.pyplot as plt


# #%% Plot white phase curves

# lc = {}
# for nrs in ['NRS1', 'NRS2']:
#     lc[nrs] = {}
#     lc[nrs]['t'], lc[nrs]['lcVals'], lc[nrs]['lcUncs'] = np.loadtxt( os.path.join(pr.WORK_DIR, 'Tom/white', nrs+'_white_data_corrected.txt'), unpack=True )

# Plot.lightcurves(lc)

#%% WHITE LIGHT CURVE FITS, FIREFLY

fit_directories = [os.path.join(pr.WORK_DIR, '250429_translational'), os.path.join(pr.WORK_DIR, '250429_quadratic')]
models = ['$\mathcal{M}_{tra}$', '$\mathcal{M}_{rot}$']

lc = Utils.pickle_read( os.path.join(fit_directories[0], 'white', pr.LC_FILE) )

# Read all models' best fit parameters and sample chains
mcmc_chains = {}
mcmc_params = {}
plot_params = {}
for i, model in enumerate(models):
    fit_directory = fit_directories[i]
    
    prelim_fit = Utils.pickle_read( os.path.join(fit_directory, 'white', pr.PRELIM_DIR, pr.FIT_FILE) )
    mcmc_chains[model] = Utils.pickle_read( os.path.join(fit_directory, 'white', pr.MCMC_DIR, pr.CHAIN_FILE) )
    mcmc_params[model] = Utils.emcee_distributions_into_dictionaries( mcmc_chains[model] )
    plot_params[model] = mcmc_params[model]['max_probabilities']
    plot_params[model] = Utils.merge_dictionaries(plot_params[model], prelim_fit.params.valuesdict())
    # plot_params[model] = prelim_fit.params.valuesdict()
    
    # Plot.white_light_curves( lc, plot_params[model], emcee_result=mcmc_chains[model] )
    
Plot.white_lightcurves_n_models_zoom( lc, plot_params, emcee_result=mcmc_chains, scale_errorbars=True, opath=os.path.join(fit_directories[0], 'white', 'white_lightcurve_fits_firefly') )

#%% WHITE LIGHT CURVE FITS, EUREKA!

fit_directories = [os.path.join(pr.WORK_DIR, '250514_translational'), os.path.join(pr.WORK_DIR, '250514_quadratic')]
models = ['$\mathcal{M}_{tra}$', '$\mathcal{M}_{rot}$']

lc = Utils.pickle_read( os.path.join(fit_directories[0], 'white', 'eureka', pr.LC_FILE) )

# Read all models' best fit parameters and sample chains
mcmc_chains = {}
mcmc_params = {}
plot_params = {}
for i, model in enumerate(models):
    fit_directory = fit_directories[i]
    
    prelim_fit = Utils.pickle_read( os.path.join(fit_directory, 'white', 'eureka', pr.PRELIM_DIR, pr.FIT_FILE) )
    mcmc_chains[model] = Utils.pickle_read( os.path.join(fit_directory, 'white', 'eureka', pr.MCMC_DIR, pr.CHAIN_FILE) )
    mcmc_params[model] = Utils.emcee_distributions_into_dictionaries( mcmc_chains[model] )
    plot_params[model] = mcmc_params[model]['max_probabilities']
    plot_params[model] = Utils.merge_dictionaries(plot_params[model], prelim_fit.params.valuesdict())
    # plot_params[model] = prelim_fit.params.valuesdict()
    
    # Plot.white_light_curves( lc, plot_params[model], emcee_result=mcmc_chains[model] )
    
Plot.white_lightcurves_n_models_zoom( lc, plot_params, emcee_result=mcmc_chains, scale_errorbars=True, opath=os.path.join(fit_directories[0], 'white', 'eureka', 'white_lightcurve_fits_eureka') )

#%% CO LIGHT CURVE FITS

directories = [os.path.join(pr.WORK_DIR + '_soss', '250514'), os.path.join(pr.WORK_DIR, '250429')]
reductions = ['JWST/NIRISS', 'JWST/NIRSpec']
models = ['translational', 'linear']

# Read all models' best fit parameters and sample chains
lcs = {}
mcmc_chains = {}
mcmc_params = {}
plot_params = {}
for i, reduction in enumerate(reductions):
    lcs[reduction] = Utils.pickle_read( os.path.join(directories[i] + '_' + models[0], 'single', pr.LC_FILE) )
    
    mcmc_chains[reduction] = {}
    plot_params[reduction] = {}
    for j, model in enumerate(models):
        fit_directory = directories[i] + '_' + model
        
        prelim_fit = Utils.pickle_read( os.path.join(fit_directory, 'single', pr.PRELIM_DIR, pr.FIT_FILE) )
        mcmc_chains[reduction][model] = Utils.pickle_read( os.path.join(fit_directory, 'single', pr.MCMC_DIR, pr.CHAIN_FILE) )
        mcmc_params = Utils.emcee_distributions_into_dictionaries( mcmc_chains[reduction][model] )
        max_likelihood_params = mcmc_params['max_probabilities']
        plot_params[reduction][model] = Utils.merge_dictionaries(max_likelihood_params, prelim_fit.params.valuesdict())
        
        # Add orbital parameters into NIRSpec reduction
        if reduction == 'JWST/NIRSpec':
            white_fit = Utils.pickle_read( os.path.join(fit_directory, 'white', pr.MCMC_DIR, pr.PARAMETERS_FILE) )
            plot_params[reduction][model] = Utils.merge_dictionaries( plot_params[reduction][model], white_fit['max_probabilities'] )

Plot.n_lightcurves_m_models_zoom( lcs, plot_params, emcee_result=mcmc_chains, scale_errorbars=True, opath=os.path.join(directories[-1] + '_' + models[0], 'single', 'co_lightcurve_fits') )


# #%% Significances transit-only

# lcs = ['Firefly', 'Eureka!']
# lnZ0 = [9626.58, 9463.618]
# lnZ1 = [9640.75, 9489.482]
# lnZ2 = [9636.56, 9484.989]
# print('Data reduction, Bayes factor(1-0), sigma significance (1-0), Bayes factor(2-0), sigma significance (2-0), Bayes factor(1-2), sigma significance (1-2)')
# for i in range(len(lcs)):
#     B1 = np.exp(lnZ1[i]-lnZ0[i])
#     sigma1 = Utils.bayes2Nsigma(B1)
#     B2 = np.exp(lnZ2[i]-lnZ0[i])
#     sigma2 = Utils.bayes2Nsigma(B2)
#     B12 = np.exp(lnZ1[i]-lnZ2[i])
#     sigma12 = Utils.bayes2Nsigma(B12)
#     print( lcs[i], B1, sigma1, B2, sigma2, B12, sigma12)

#%% Significances transit+eclipses

lcs = ['Firefly', 'Eureka!']
lnZ0 = [15122.641, 14892.17]
lnZ1 = [15136.765, 14914.49]
lnZ2 = [15134.797, 14914.91]
print('Data reduction, Bayes factor(2-0), sigma significance (2-0), Bayes factor(1-0), sigma significance (1-0), Bayes factor(1-2), sigma significance (1-2)')
for i in range(len(lcs)):
    B2 = np.exp(lnZ2[i]-lnZ0[i])
    sigma2 = Utils.bayes2Nsigma(B2)
    B1 = np.exp(lnZ1[i]-lnZ0[i])
    sigma1 = Utils.bayes2Nsigma(B1)
    B12 = np.exp(lnZ1[i]-lnZ2[i])
    sigma12 = Utils.bayes2Nsigma(B12)
    print( lcs[i], B2, sigma2, B1, sigma1, B12, sigma12)

#%% R_p(t) plots

def Rp_uncertainty(R0_unc, R1, R1_unc, R2, R2_unc, t0, t0_unc, t):
    return np.sqrt( R0_unc**2 + (R1_unc * t)**2 + (R1*t0_unc)**2 + (R2_unc*t**2)**2 + (2.*R2*t*t0*t0_unc)**2 )
def Rp_R0_uncertainty(R0, R0_unc, R1, R1_unc, t0_unc, t):
    return np.sqrt( (R1 * t * R0**(-2) * R0_unc)**2 + (R1_unc * t / R0 )**2 + (R1 * t0_unc / R0)**2 )

T14 = 2.91/24.
t = np.linspace(-T14/2., T14/2., 100)
molecules = ['co', 'h2o']
line_colors = ['purple', 'red', 'green']
shade_colors = ['magenta', 'red', 'lightgreen']
line_styles = ['-', '--', '-.']
sigma_shading = [1.]#, 2., 3.]

# Which observations to plot
fit_dirs = ['250320_linear', '250320_quadratic', '250320_fixLD_quadratic']
fit_labels = ['Free LD, linear $R_p/R_*(t)$', 'Free LD, quadratic $R_p/R_*(t)$', 'Fixed LD, quadratic $R_p/R_*(t)$']

# Read models
model_files = ["nominal"]#, "co_removed_from_morning"]
model_labels = ["GCM"]#, "CO removed from morning"]
model_styles = ['--', ':']
models = {}
for i, model_file in enumerate(model_files):
    f=h5py.File(os.path.join(pr.WORK_DIR, "aurelien/" + model_file + ".h5") )
    label = model_labels[i]
    
    models[label] = {}
    models[label]['time'] = f["Model/lightcurve/times"][...]
    models[label]['rprs'] = np.sqrt(f["Output/lightcurve/flux"][...])
    models[label]['wvls'] = 10000./f["Output/lightcurve/wns"][...]

plt.clf()
fig, ax = plt.subplots(2, 1, figsize=[6,6], sharex=True)
for i, molecule in enumerate(molecules):
    
    # Observations
    for j, fit_dir in enumerate(fit_dirs):
        data_dir = os.path.join(pr.WORK_DIR, fit_dir, molecule)
        
        # # MCMC
        # fit_parameters = Utils.pickle_read( os.path.join(data_dir, pr.MCMC_DIR, pr.PARAMETERS_FILE) )
        
        # Rp = fit_parameters['medians']['R0'] + fit_parameters['medians']['R1']*t + fit_parameters['medians']['R2']*t**2
        # ax[i].plot(t*24.*3600., Rp, label=fit_dir, color=line_colors[i])
        
        # for sigma in sigma_shading:
        #     upper_boundary = Rp + sigma * Rp_uncertainty( fit_parameters['stderrs']['R0'], fit_parameters['medians']['R1'], \
        #                                   fit_parameters['stderrs']['R1'], fit_parameters['medians']['R2'], fit_parameters['stderrs']['R2'], fit_parameters['medians']['Tmid'], fit_parameters['stderrs']['Tmid'], t )
        #     lower_boundary = Rp - sigma * Rp_uncertainty( fit_parameters['stderrs']['R0'], fit_parameters['medians']['R1'], \
        #                                   fit_parameters['stderrs']['R1'], fit_parameters['medians']['R2'], fit_parameters['stderrs']['R2'], fit_parameters['medians']['Tmid'], fit_parameters['stderrs']['Tmid'], t )
        
        #     ax[i].fill_between(t*24.*3600., lower_boundary, upper_boundary, color=shade_colors[i], alpha=0.4)
        
        # LSQ
        fit_result = Utils.pickle_read( os.path.join(data_dir, pr.PRELIM_DIR, pr.FIT_FILE) )
        fit_parameters = fit_result.params
        print(molecule)
        fit_parameters.pretty_print()
        Rp = fit_parameters['R0'].value + fit_parameters['R1'].value*t + fit_parameters['R2'].value*t**2
        ax[i].plot(t*24.*3600., Rp, line_styles[j], label=fit_labels[j], color=line_colors[j])
        
        for sigma in sigma_shading:
            uncertainty = Rp_uncertainty( fit_parameters['R0'].stderr, fit_parameters['R1'].value, \
                                          fit_parameters['R1'].stderr, fit_parameters['R2'].value, fit_parameters['R2'].stderr, fit_parameters['Tmid'].value, fit_parameters['Tmid'].stderr, t )
            upper_boundary = Rp + sigma * uncertainty
            lower_boundary = Rp - sigma * uncertainty
            
            ax[i].fill_between(t*24.*3600., lower_boundary, upper_boundary, color=shade_colors[j], alpha=0.4)
    
    # # Normalized by R0
    # Rp_R0 = 1. + fit_parameters['medians']['R1']*t/fit_parameters['medians']['R0']
    # ax[1].plot(t*24.*3600., Rp_R0, label=molecule, color=line_colors[i])
    #
    # for sigma in sigma_shading:
    #     upper_boundary = Rp_R0 + Rp_R0_uncertainty( fit_parameters['medians']['R0'], fit_parameters['stderrs']['R0'],\
    #                   fit_parameters['medians']['R1'], fit_parameters['stderrs']['R1'], fit_parameters['stderrs']['Tmid'], t )
    #     lower_boundary = Rp_R0 - Rp_R0_uncertainty( fit_parameters['medians']['R0'], fit_parameters['stderrs']['R0'],\
    #                   fit_parameters['medians']['R1'], fit_parameters['stderrs']['R1'], fit_parameters['stderrs']['Tmid'], t )
    #
    #     ax[1].fill_between(t*24.*3600., lower_boundary, upper_boundary, color=shade_colors[i], alpha=0.2)
    
    # Models
    lc = Utils.pickle_read( os.path.join(pr.WORK_DIR, fit_dirs[0], molecule, pr.LC_FILE ) )
    wavelength_range = lc['wavEdgesMicr']
    for j, key in enumerate(model_labels):
        i_min = np.argmin( np.abs(models[key]['wvls']-wavelength_range[1]) )
        i_max = np.argmin( np.abs(models[key]['wvls']-wavelength_range[0]) )
        i0 = np.argmin( np.abs(models[key]['time']) )
        rprs_integrated = np.mean( models[key]['rprs'][:,i_min:i_max], axis=1 )
        ax[i].plot(models[key]['time'], rprs_integrated, ':', label=key, color='black')
        # ax[i].plot(models[key]['time'], rprs_integrated + (fit_parameters['medians']['R0']-rprs_integrated[i0]), model_styles[j], label=key, color='black')
        if molecule == 'co':
            ax[i].set_ylabel('$R_p/R_*$ (CO band)')
        elif molecule == 'h2o':
            ax[i].set_ylabel('$R_p/R_*$ (H$_2$O band)')
        else:
            ax[i].set_ylabel('$R_p/R_*$ (' + molecule + ' band)')
        # ax[i].grid()
        # ax[1].plot(time, rprs_integrated/rprs_integrated[i0], '--', color=line_colors[i], label='quadratic fit to GCM')
        # print(molecule,rprs_integrated)
        
        ax[i].legend( loc='lower left', fontsize='x-small' )

ax[-1].set_xlabel('Time from mid-transit [s]')
ax[0].set_xlim(-4000.,4000.)
ax[0].set_ylim(0.12,0.13)
ax[1].set_ylim(0.12,0.126)
plt.savefig( os.path.join(pr.RESULTS_DIR, 'measurements.pdf'), bbox_inches='tight', dpi=300 )
plt.show()
plt.close()

#%% R_p(t) plots Astronodes

def Rp_uncertainty(R0_unc, R1, R1_unc, R2, R2_unc, t0, t0_unc, t):
    return np.sqrt( R0_unc**2 + (R1_unc * t)**2 + (R1*t0_unc)**2 + (R2_unc*t**2)**2 + (2.*R2*t*t0*t0_unc)**2 )

T23 = 2.25/24.
t = np.linspace(-T23/2., T23/2., 100)
colors = ['blue', 'green', 'purple']
observation_styles = ['-', '--', '-.']
sigma_shading = [1.]#, 2., 3.]
alpha_shading = 0.2

# Which observations to plot
fit_dirs = ['250429_linear', '250429_quadratic_ldc-priors', '250429_quadratic']
fit_labels = ['$R_2/R_*=0$, uniform priors on LDCs', '$R_2/R_*$ free, Gaussian priors on LDCs', '$R_2/R_*$ free, uniform priors on LDCs']

# Read models
model_files = ["output_nominal"]#, "co_removed_from_morning"]
model_labels = ["SPARC/MITgcm"]#, "CO removed from morning"]
model_styles = ['--', ':', '-.']
models = {}
for i, model_file in enumerate(model_files):
    f=h5py.File(os.path.join(pr.WORK_DIR, "aurelien/" + model_file + ".h5") )
    label = model_labels[i]
    
    models[label] = {}
    models[label]['time'] = f["Model/lightcurve/times"][...]/24./3600.
    models[label]['rprs'] = np.sqrt(f["Output/lightcurve/flux"][...])
    models[label]['wvls'] = 10000./f["Output/lightcurve/wns"][...]

plt.clf()
fig, ax = plt.subplots(2, 1, figsize=[6,6], sharex=True)
lc = Utils.pickle_read( os.path.join(pr.WORK_DIR, fit_dirs[0], 'white', pr.LC_FILE) )
detectors = list(lc.keys())

Rp_ref = {}
# Observations
for i, fit_dir in enumerate(fit_dirs):
    fit_parameters = Utils.pickle_read( os.path.join(pr.WORK_DIR, fit_dir, 'white', pr.MCMC_DIR, pr.PARAMETERS_FILE) )
    
    for j, nrs in enumerate(detectors):
        
        Rp = fit_parameters['medians']['R0_' + nrs] + fit_parameters['medians']['R1_' + nrs]*t + fit_parameters['medians']['R2_' + nrs]*t**2
        ax[j].plot(t*24.*3600., Rp, observation_styles[i], label=fit_labels[i], color=colors[i], zorder=1)
        if i == 0:
            Rp_ref[nrs] = Rp
        for sigma in sigma_shading:
            uncertainty = Rp_uncertainty( fit_parameters['stderrs']['R0_' + nrs], fit_parameters['medians']['R1_' + nrs], \
                                          fit_parameters['stderrs']['R1_' + nrs], fit_parameters['medians']['R2_' + nrs], fit_parameters['stderrs']['R2_' + nrs], fit_parameters['medians']['Tmid_' + nrs], fit_parameters['stderrs']['Tmid_' + nrs], t )
            upper_boundary = Rp + sigma * uncertainty
            lower_boundary = Rp - sigma * uncertainty
            
            ax[j].fill_between(t*24.*3600., lower_boundary, upper_boundary, color=colors[i], alpha=alpha_shading)

# Models
for i, key in enumerate(model_labels):
    for j, nrs in enumerate(detectors):
        # Integrate model over wavelength range
        wavelength_range = lc[nrs]['wavEdgesMicr']
        i_min = np.argmin( np.abs(models[key]['wvls']-wavelength_range[1]) )
        i_max = np.argmin( np.abs(models[key]['wvls']-wavelength_range[0]) )
        rprs_integrated = np.mean( models[key]['rprs'][:,i_min:i_max], axis=1 )
        
        # Trim from T2 to T3
        time, rprs = Utils.trim_spectrum( models[key]['time'], rprs_integrated, np.min(t), np.max(t) )
        ax[j].plot( time*24.*3600., rprs - np.mean(rprs) + np.mean(Rp_ref[nrs]), ':', label=key, color='black', zorder=3)
        ax[j].set_ylabel('$R_p/R_*$ (' + nrs + ')')
        
        # Fit quadratic to model light curves
        a2, a1, a0 = np.polyfit( time, rprs, 2)
        ax[j].plot( time*3600.*24., (a2*time**2 + a1*time + a0) - np.mean(rprs) + np.mean(Rp_ref[nrs]), color='magenta', label='Quadratic fit to SPARC/MITgcm', zorder=2)
        print(nrs, 'a1:', a1, 'a2:', a2 )
        ax[j].legend( fontsize='x-small' )

ax[-1].set_xlabel('Time from mid-transit [s]')
ax[0].set_xlim(np.min(t)*24.*3600.,np.max(t)*24.*3600.)
plt.savefig( os.path.join(pr.WORK_DIR, fit_dirs[0], 'white', pr.MCMC_DIR, 'radius_functions.pdf'), bbox_inches='tight', dpi=300 )
plt.show()
plt.close()

# #%% SINGLE LIGHT CURVE FITS

# ######## Set directories ########

# molecules = ['h2o', 'co']

# lcs = {}
# prelim_params = {}
# mcmc_chains = {}
# plot_params = {}

# # Read the white fit parameters
# white_fit_params = {}
# white_fit = Utils.pickle_read( os.path.join( pr.BROAD_DIR, pr.MCMC_DIR, pr.PARAMETERS_FILE) )
# white_fit_params['Rotational model'] = white_fit['max_probabilities']
# white_fit = Utils.pickle_read(os.path.join( pr.RESULTS_DIR[:-1] + '_translational', 'white', pr.MCMC_DIR, pr.PARAMETERS_FILE ))
# white_fit_params['Translational model'] = white_fit['max_probabilities']

# for i, molecule in enumerate(molecules):
    
#     lc_directory = os.path.join(pr.RESULTS_DIR, molecule)
#     lcs[molecule] = Utils.pickle_read( os.path.join( lc_directory, pr.LC_FILE ) )
#     prelim_params[molecule] = {}
#     mcmc_chains[molecule] = {}
#     plot_params[molecule] = {}
    
#     model = 'Translational model'
#     # Set directories
#     translational_dir = os.path.join( pr.RESULTS_DIR[:-1] + '_translational', lc_directory.split('/')[-2] )
#     prelim_dir = os.path.join( translational_dir, pr.PRELIM_DIR )
#     mcmc_dir = os.path.join( translational_dir, pr.MCMC_DIR )
    
#     prelim_fit = Utils.pickle_read( os.path.join(prelim_dir, pr.FIT_FILE) )
#     prelim_params[molecule][model] = prelim_fit.params.valuesdict()
#     mcmc_chains[molecule][model] = Utils.pickle_read( os.path.join(mcmc_dir, pr.CHAIN_FILE) )
#     mcmc_params = Utils.emcee_distributions_into_dictionaries( mcmc_chains[molecule][model] )
#     plot_params[molecule][model] = mcmc_params['max_probabilities']
    
#     model = 'Rotational model'
#     # Set directories
#     prelim_dir = os.path.join( lc_directory, pr.PRELIM_DIR )
#     mcmc_dir = os.path.join( lc_directory, pr.MCMC_DIR )
    
#     prelim_fit = Utils.pickle_read( os.path.join(prelim_dir, pr.FIT_FILE) )
#     prelim_params[molecule][model] = prelim_fit.params.valuesdict()
#     mcmc_chains[molecule][model] = Utils.pickle_read( os.path.join(mcmc_dir, pr.CHAIN_FILE) )
#     mcmc_params = Utils.emcee_distributions_into_dictionaries( mcmc_chains[molecule][model] )
#     plot_params[molecule][model] = mcmc_params['max_probabilities']

# Plot.n_lightcurves_m_models( lcs, plot_params, prelim_params, white_fit_params, emcee_result=mcmc_chains, unit=1.0E-6, opath=os.path.join(pr.RESULTS_DIR, 'lightcurve_fits') )
