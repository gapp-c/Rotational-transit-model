#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs an MCMC to the white light curves of JWST/NIRSpec G395H transmission spectra using emcee.

INPUT
----------
broad_lc : dictionary
    Dictionary containing the broad band light curves of both NRS1 & NRS2
prelim_fit: lmfit results object
    Result of the preliminary least squares fit with free limb darkening coefficients.

OUTPUT
-------
mcmc_result : dictionary
    Contains the flattened chain of the mcmc as well as a list of fit parameters and the acceptance fractions of all walkers.
mcmc_params : lmfit parameters
    Fit parameters' most likely values and half widths between 16th and 84th percentiles.
    
"""

import os, emcee
import numpy as np
import Utils, Plot, initialParameters, calculateBatmanModel
import parameters as pr
if pr.USE_ASTRONODES:
    from multiprocessing import Pool

######## Set directories ########

fit_in_dir = os.path.join( pr.BROAD_DIR, 'eureka', pr.PRELIM_DIR )
mcmc_dir = os.path.join( pr.BROAD_DIR, 'eureka', pr.MCMC_DIR )

######## Input ########

number_of_walkers = 500
thin_number = 100
if pr.USE_ASTRONODES:
    number_of_steps = 50000
    burn_number = 20000
else:
    number_of_steps = 20000
    burn_number = 4000

detectors = ['NRS1', 'NRS2']
broad_lc = {}
for detector in detectors:
    broad_lc[detector] = {}
    broad_lc[detector]['t'], broad_lc[detector]['lcVals'], broad_lc[detector]['lcUncs'] = np.loadtxt( os.path.join(pr.WORK_DIR, 'eureka', 'WASP-121b_' + detector + '_WL.txt'), unpack=True )
    broad_lc[detector]['bjd_tdb'] = broad_lc[detector]['t'] + pr.TIME_0 + pr.TRANSIT_MID_TIME
    broad_lc[detector] = Utils.remove_nans( broad_lc[detector] )

broad_lc['NRS1']['u1'] = 0.06585945520225178
broad_lc['NRS1']['u2'] = 0.11265248382653993
broad_lc['NRS1']['u1_err'] = 0.011569608991181486
broad_lc['NRS1']['u2_err'] = 0.015043755034295784

broad_lc['NRS2']['u1'] = 0.05605071953777346
broad_lc['NRS2']['u2'] = 0.09230747753839463
broad_lc['NRS2']['u1_err'] = 0.009485355297039812
broad_lc['NRS2']['u2_err'] = 0.012333638781348886
prelim_fit = Utils.pickle_read( os.path.join(fit_in_dir, pr.FIT_FILE) )

# MCMC. Caution with the paralellization, this part has to be run outside of a function!
p0 = prelim_fit.params

for nrs in detectors:
    if pr.ADD_SYSTEMATIC_UNCERTAINTY:
        p0.add('sigma_sys_' + nrs, value = 0., min = 0. )

priors = initialParameters.lmfitparams2priors( p0 )

if not(pr.KIPPING_LD_PARAMETERS) and pr.LDC_PRIORS:
    if pr.FIT_1ST_LD_PARAMETER:
        for nrs in detectors:
            priors['u1_' + nrs] = Utils.make_prior_gaussian( broad_lc[nrs]['u1'], broad_lc[nrs]['u1_err'] )
    if pr.FIT_2ND_LD_PARAMETER:
        for nrs in detectors:
            priors['u2_' + nrs] = Utils.make_prior_gaussian( broad_lc[nrs]['u2'], broad_lc[nrs]['u2_err'] )

# Transfer the parameters from lmfit to emcee format. Parameters that do not get varied have to be added back to the dictionary later!
varied_parameters = Utils.varied_parameters_lmfit( p0 )

# Initialize a batman model for every data point to change the transit radius as a function of time
batpar = {}
batmodel = {}
for nrs in detectors:
    batpar[nrs], batmodel[nrs] = calculateBatmanModel.create_Batman_arrays( broad_lc[nrs]['bjd_tdb'] )

# Spread the initial positions of the walkers around the best-fit values from the preliminary fit
walker_positions = Utils.gaussian_ball( p0, number_of_walkers, priors )

# This is the function we want to maximize
def log_posterior( pars ):
    
    pars_dict = Utils.arrays2dict( pars, varied_parameters )
    log_prior = Utils.evaluate_priors(priors, pars_dict)
    
    if (not np.isfinite(log_prior)):
        return -np.inf
    
    # If the log_prior is not -inf, then calculate the log likelihood and add it to the log_prior to get the log_posterior
    pars_dict = Utils.add_fixed_parameters2dict( pars_dict, p0 )
    
    summand = np.asarray([])
    for nrs in detectors:
        model = calculateBatmanModel.Broad( pars_dict, broad_lc[nrs], batpar[nrs], batmodel[nrs], nrs )
        if pr.ADD_SYSTEMATIC_UNCERTAINTY:
            summand = np.append( summand, (broad_lc[nrs]['lcVals']-model)**2 / (broad_lc[nrs]['lcUncs']**2 + (pars_dict['sigma_sys_' + nrs]*1.0E-6)**2) + np.log( 2. * np.pi * (broad_lc[nrs]['lcUncs']**2 + (pars_dict['sigma_sys_' + nrs]*1.0E-6)**2) ) )
        else:
            summand = np.append( summand, (broad_lc[nrs]['lcVals']-model)**2 / broad_lc[nrs]['lcUncs']**2 + np.log( 2. * np.pi * broad_lc[nrs]['lcUncs']**2 ) )
    
    return log_prior - 0.5*np.sum( summand )

if pr.USE_ASTRONODES:
    pool = Pool(pr.NUMBER_OF_CORES)
    sampler = emcee.EnsembleSampler(number_of_walkers, len(varied_parameters), log_posterior, pool=pool )
else:
    sampler = emcee.EnsembleSampler(number_of_walkers, len(varied_parameters), log_posterior )

sampler.run_mcmc(walker_positions, number_of_steps, progress=not(pr.USE_ASTRONODES))

if pr.USE_ASTRONODES:
    pool.close()
    pool.join()

# Output
Utils.pickle_save( mcmc_dir, pr.CHAIN_FILE, Utils.emcee_sampler2dictionary( sampler, varied_parameters, burn_number, thin_number ) )
Plot.emcee_walkers( sampler.get_chain(), varied_parameters, os.path.join(mcmc_dir, 'emcee_walkers'))

