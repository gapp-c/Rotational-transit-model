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

import os, dynesty, sys
from dynesty import plotting as dyplot
import numpy as np
import Utils, initialParameters, calculateBatmanModel, Plot
import parameters as pr
if pr.USE_ASTRONODES:
    from multiprocessing import Pool

######## Set directories ########

broad_fit_dir = os.path.join( pr.BROAD_DIR, pr.MCMC_DIR )
broad_fit = Utils.pickle_read( os.path.join( broad_fit_dir, pr.PARAMETERS_FILE) )
broad_fit_params = broad_fit['max_probabilities']

# fit_in_dir = os.path.join( pr.BROAD_DIR, pr.PRELIM_DIR )
# broad_fit = Utils.pickle_read( os.path.join(fit_in_dir, pr.FIT_FILE) )
# broad_fit_params = broad_fit.params.valuesdict()

for i, molecule in enumerate(pr.MOLECULES):
    
    specfit_in_dir = os.path.join( pr.RESULTS_DIR, molecule, pr.PRELIM_DIR )
    specfit_out_dir = os.path.join( pr.RESULTS_DIR, molecule, pr.NESTED_DIR )
    
    ######## Input ########

    lc = Utils.pickle_read( os.path.join( pr.RESULTS_DIR, molecule, pr.LC_FILE ) )

    prelim_fit = Utils.pickle_read( os.path.join(specfit_in_dir, pr.FIT_FILE) )
    p0 = prelim_fit.params

    # Add systematic uncertainty to the fit parameters and run the MCMC
    if pr.ADD_SYSTEMATIC_UNCERTAINTY:
        p0.add('sigma_sys', value = 0., min = 0. )

    priors = initialParameters.dynesty_uniform_priors( p0 )
    
    varied_parameters = Utils.varied_parameters_lmfit(p0)
    ndim = len(varied_parameters)
    
    if not(pr.KIPPING_LD_PARAMETERS) and pr.LDC_PRIORS:
        if pr.FIT_1ST_LD_PARAMETER:
            priors['u1'] = Utils.make_prior_gaussian( lc['u1'], lc['u1_err'] )
        if pr.FIT_2ND_LD_PARAMETER:
            priors['u2'] = Utils.make_prior_gaussian( lc['u2'], lc['u2_err'] )
    
    # First, initialize a batman model for every data point to change the transit radius as a function of time
    batpar, batmodel = calculateBatmanModel.create_Batman_arrays( lc['bjd_tdb'] )
    for j in range(len(batpar)):
        batpar[j] = calculateBatmanModel.fix_broad_fit_parameters(batpar[j], broad_fit_params)
        
    # This is the function we want to maximize
    def log_likelihood( pars ):
        pars_dict = Utils.arrays2dict( pars, varied_parameters )
    
        pars_dict = Utils.add_fixed_parameters2dict( pars_dict, p0 )
        model = calculateBatmanModel.Spec( pars_dict, lc, batpar, batmodel )
    
        if pr.ADD_SYSTEMATIC_UNCERTAINTY:
            summand = (lc['lcVals']-model)**2 / (lc['lcUncs']**2 + (pars_dict['sigma_sys']*1.0E-6)**2) + np.log( 2. * np.pi * (lc['lcUncs']**2 + (pars_dict['sigma_sys']*1.0E-6)**2) )
        else:
            summand = (lc['lcVals']-model)**2 / lc['lcUncs']**2 + np.log( 2. * np.pi * lc['lcUncs']**2 )
    
        return -0.5*np.sum( summand )
    
    if pr.USE_ASTRONODES:
        pool = Pool(pr.NUMBER_OF_CORES_SINGLE)
        sampler = dynesty.NestedSampler( log_likelihood, Utils.prior_transform, ndim, ptform_args=[priors], nlive=pr.LIVE_POINTS, pool=pool, queue_size=pr.NUMBER_OF_CORES_SINGLE )
    else:
        sampler = dynesty.NestedSampler( log_likelihood, Utils.prior_transform, ndim, ptform_args=[priors], nlive=pr.LIVE_POINTS )

    sampler.run_nested()
    
    if pr.USE_ASTRONODES:
        pool.close()
        pool.join()
    
    results = sampler.results
    samples = results.samples_equal()
    
    ### Output
    if os.path.isdir( specfit_out_dir )==False:
        os.makedirs( specfit_out_dir )
    
    # Write summary to file rather than terminal
    old_stdout = sys.stdout
    with open(os.path.join(specfit_out_dir, 'sampling_results.txt'), 'w') as sys.stdout:
        results.summary()
    sys.stdout = old_stdout
    
    tfig, taxes = dyplot.traceplot(results, labels=varied_parameters)
    tfig.savefig( os.path.join(specfit_out_dir, 'traces.png'), dpi=300 )
    
    cfig, caxes = dyplot.cornerplot(results, labels=varied_parameters)
    cfig.savefig( os.path.join(specfit_out_dir, 'corner.pdf') )
    
    medians = Utils.add_fixed_parameters2dict( Utils.samples2medians(samples, varied_parameters), p0 )
    Plot.spectral_light_curve_jason( lc, medians, broad_fit_params, unit=1.0E-6, opath=os.path.join(specfit_out_dir, 'lc_fit'), label=molecule )
