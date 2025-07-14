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

import os
import numpy as np
import Utils, initialParameters, calculateBatmanModel, Plot
import parameters as pr


######## Set directories ########

broadfit_in_dir = os.path.join( pr.BROAD_DIR, pr.MCMC_DIR )
specfit_in_dir = os.path.join( pr.SPEC_DIR, pr.PRELIM_DIR )
trans_in_dir = os.path.join( pr.SPEC_DIR, pr.TRANS_DIR, pr.PRELIM_DIR )

mcmc_dir = os.path.join( pr.SPEC_DIR, pr.MCMC_DIR )
trans_out_dir = os.path.join( pr.SPEC_DIR, pr.TRANS_DIR, pr.MCMC_DIR )

######## Input ########
broad_fit = Utils.pickle_read( os.path.join( broadfit_in_dir, pr.PARAMETERS_FILE) )
broad_fit_params = broad_fit['max_probabilities']
spec_lc = Utils.pickle_read( os.path.join( pr.SPEC_DIR, pr.LC_EDITED_FILE ) )

keys = list(spec_lc.keys())
nchan = len(keys)

# These lines only serve for getting the list of fit parameters out of the parameter function for plotting
fit_parameters = initialParameters.Spec( broad_fit_params, 'NRS1', 0.1, 0.1 )

if pr.ADD_SYSTEMATIC_UNCERTAINTY:
    fit_parameters.add('sigma_sys', value = 0., min = 0. )

trans_spec = Utils.prepare_transmission_dictionary( fit_parameters, nchan, only_varied_params=True, asymmetric_errors=True, bic=True )


for i, k in enumerate(keys):
    
    prelim_fit = Utils.pickle_read( os.path.join(specfit_in_dir, k) )
    
    print( '--------- Working on ' + k + ', MCMC POST PROCESSING ---------' )
    
    mcmc_result = Utils.pickle_read( os.path.join(mcmc_dir, k) )
    mcmc_params = Utils.emcee_distributions_into_dictionaries( mcmc_result )
    
    # To compare data and best-fitting model, first calculate the best-fitting model
    best_params = mcmc_params['medians']                # Which parameters to take for the comparison
    best_params = Utils.add_fixed_parameters2dict(best_params, prelim_fit.params)
    
    # Set up batman model
    batpar, batmodel = calculateBatmanModel.create_Batman_arrays( spec_lc[k]['bjd_tdb'] )
    for j in range(len(batpar)):
        batpar[j] = calculateBatmanModel.fix_broad_fit_parameters(batpar[j], broad_fit_params)
    
    Plot.corner_plot( mcmc_result, Utils.dictionary2array(mcmc_params['max_probabilities']), os.path.join(mcmc_dir, 'corner_' + k) )
    
    # trans_spec['chisquare'][i] = Utils.compare_data_and_model_spec( spec_lc[k], best_params, broad_fit_params, mcmc_dir, k, mcmc_result )
    Utils.pickle_save( mcmc_dir, k + '_' + pr.PARAMETERS_FILE, mcmc_params )
    
    trans_spec['wavelength'][i] = np.sum(spec_lc[k]['wavEdgesMicr']) / 2.
    trans_spec['wavelength_halfwidth'][i] = (spec_lc[k]['wavEdgesMicr'][1] - spec_lc[k]['wavEdgesMicr'][0]) / 2.
    
    n_var = 0
    for var in fit_parameters.keys():
        if fit_parameters[var].vary:
            trans_spec[var]['value'][i] = mcmc_params['medians'][var]
            trans_spec[var]['stderr'][i] = mcmc_params['stderrs'][var]
            trans_spec[var]['upperr'][i] = mcmc_params['upperrs'][var]
            trans_spec[var]['lowerr'][i] = mcmc_params['lowerrs'][var]
            n_var += 1
    
    trans_spec['bic'][i] = float(n_var) * np.log( float(len(spec_lc[k]['bjd_tdb'])) ) - 2.*np.max(mcmc_result['flat_log_probability'])


######## Output ########
Utils.pickle_save( trans_out_dir, pr.TRANS_FILE, trans_spec)
