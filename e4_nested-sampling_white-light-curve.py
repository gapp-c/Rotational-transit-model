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

import os, dynesty, sys
import numpy as np
import Utils, Plot, initialParameters, calculateBatmanModel
import parameters as pr
from dynesty import plotting as dyplot
if pr.USE_ASTRONODES:
    from multiprocessing import Pool

######## Set directories ########

fit_in_dir = os.path.join( pr.BROAD_DIR, 'eureka', pr.PRELIM_DIR )
fit_out_dir = os.path.join( pr.BROAD_DIR, 'eureka', pr.NESTED_DIR )

######## Input ########

prelim_fit = Utils.pickle_read( os.path.join(fit_in_dir, pr.FIT_FILE) )
broad_lc = Utils.pickle_read( os.path.join(pr.BROAD_DIR, 'eureka', pr.LC_FILE) )
detectors = list(broad_lc.keys())

# MCMC. Caution with the paralellization, this part has to be run outside of a function!
p0 = prelim_fit.params

if pr.ADD_SYSTEMATIC_UNCERTAINTY:
    for nrs in detectors:
        p0.add('sigma_sys_' + nrs, value = 0., min = 0. )

# Transfer the parameters from lmfit to emcee format. Parameters that do not get varied have to be added back to the dictionary later!
varied_parameters = Utils.varied_parameters_lmfit( p0 )
priors = initialParameters.dynesty_uniform_priors( p0 )
ndim = len(varied_parameters)

if not(pr.KIPPING_LD_PARAMETERS) and pr.LDC_PRIORS:
    if pr.FIT_1ST_LD_PARAMETER:
        for nrs in detectors:
            priors['u1_' + nrs] = Utils.make_prior_gaussian( broad_lc[nrs]['u1'], broad_lc[nrs]['u1_err'] )
    if pr.FIT_2ND_LD_PARAMETER:
        for nrs in detectors:
            priors['u2_' + nrs] = Utils.make_prior_gaussian( broad_lc[nrs]['u2'], broad_lc[nrs]['u2_err'] )

# Initialize a batman model for every data point to change the transit radius as a function of time
batpar = {}
batmodel = {}
for nrs in detectors:
    batpar[nrs], batmodel[nrs] = calculateBatmanModel.create_Batman_arrays( broad_lc[nrs]['bjd_tdb'] )

# This is the function we want to maximize
def log_likelihood( pars ):
    
    pars_dict = Utils.arrays2dict( pars, varied_parameters )
    
    # If the log_prior is not -inf, then calculate the log likelihood and add it to the log_prior to get the log_posterior
    pars_dict = Utils.add_fixed_parameters2dict( pars_dict, p0 )
    
    summand = np.asarray([])
    for nrs in detectors:
        model = calculateBatmanModel.Broad( pars_dict, broad_lc[nrs], batpar[nrs], batmodel[nrs], nrs )
        if pr.ADD_SYSTEMATIC_UNCERTAINTY:
            summand = np.append( summand, (broad_lc[nrs]['lcVals']-model)**2 / (broad_lc[nrs]['lcUncs']**2 + (pars_dict['sigma_sys_' + nrs]*1.0E-6)**2) + np.log( 2. * np.pi * (broad_lc[nrs]['lcUncs']**2 + (pars_dict['sigma_sys_' + nrs]*1.0E-6)**2) ) )
        else:
            summand = np.append( summand, (broad_lc[nrs]['lcVals']-model)**2 / broad_lc[nrs]['lcUncs']**2 + np.log( 2. * np.pi * broad_lc[nrs]['lcUncs']**2 ) )
    
    return -0.5*np.sum( summand )

if pr.USE_ASTRONODES:
    pool = Pool(pr.NUMBER_OF_CORES)
    sampler = dynesty.NestedSampler( log_likelihood, Utils.prior_transform, ndim, ptform_args=[priors], nlive=pr.LIVE_POINTS, pool=pool, queue_size=pr.NUMBER_OF_CORES )
else:
    sampler = dynesty.NestedSampler( log_likelihood, Utils.prior_transform, ndim, ptform_args=[priors], nlive=pr.LIVE_POINTS )

sampler.run_nested( print_progress=not(pr.USE_ASTRONODES) )

if pr.USE_ASTRONODES:
    pool.close()
    pool.join()

results = sampler.results
samples = results.samples_equal()
medians = Utils.add_fixed_parameters2dict( Utils.samples2medians(samples, varied_parameters), p0 )

### Output
if os.path.isdir( fit_out_dir )==False:
    os.makedirs( fit_out_dir )

# Write summary to file rather than terminal
old_stdout = sys.stdout
with open(os.path.join(fit_out_dir, 'sampling_results.txt'), 'w') as sys.stdout:
    results.summary()
sys.stdout = old_stdout

### Output ###

Utils.pickle_save( fit_out_dir, pr.FIT_FILE, results )
Utils.pickle_save( fit_out_dir, pr.SAMPLES_FILE, Utils.samples2dictionary(samples, varied_parameters) )
Plot.white_light_curves(broad_lc, medians, opath=os.path.join(fit_out_dir, 'white-fit'))

tfig, taxes = dyplot.traceplot(results, labels=varied_parameters)
tfig.savefig( os.path.join(fit_out_dir, 'traces.png'), dpi=300 )

cfig, caxes = dyplot.cornerplot(results, labels=varied_parameters)
cfig.savefig( os.path.join(fit_out_dir, 'corner.pdf') )