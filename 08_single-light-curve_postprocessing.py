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
import Utils, calculateBatmanModel, Plot
import parameters as pr


######## Set directories ########

broad_fit_dir = os.path.join( pr.BROAD_DIR, pr.MCMC_DIR )
broad_fit = Utils.pickle_read( os.path.join( broad_fit_dir, pr.PARAMETERS_FILE) )
broad_fit_params = broad_fit['max_probabilities']

######## Input ########

for i, molecule in enumerate(pr.MOLECULES):
    specfit_in_dir = os.path.join( pr.RESULTS_DIR, molecule, pr.PRELIM_DIR )
    mcmc_dir = os.path.join( pr.RESULTS_DIR, molecule, pr.MCMC_DIR )
    lc = Utils.pickle_read( os.path.join( pr.RESULTS_DIR, molecule, pr.LC_FILE ) )
    
    prelim_fit = Utils.pickle_read( os.path.join(specfit_in_dir, pr.FIT_FILE) )
    mcmc_chain = Utils.pickle_read( os.path.join(mcmc_dir, pr.CHAIN_FILE) )
    mcmc_params = Utils.emcee_distributions_into_dictionaries( mcmc_chain )
    n_var = len(mcmc_params['max_probabilities'])
    Utils.pickle_save( mcmc_dir, pr.PARAMETERS_FILE, mcmc_params )
    Plot.corner_plot( mcmc_chain, Utils.dictionary2array(mcmc_params['max_probabilities']), os.path.join(mcmc_dir, 'corner') )
    plot_params = Utils.merge_dictionaries(mcmc_params['max_probabilities'], prelim_fit.params.valuesdict())
    
    Plot.spectral_light_curve_jason( lc, plot_params, broad_fit_params, unit=1.0E-6, opath=os.path.join(mcmc_dir, 'rotational_model'), label=molecule )
    
    Utils.write_fit_dictionary_to_csv_file(mcmc_params, os.path.join( mcmc_dir, 'fit-parameters') )
    
    # Plot.histogram( radius_change )
    
    batpar, batmodel = calculateBatmanModel.create_Batman_arrays( lc['bjd_tdb'] )
    for j in range(len(batpar)):
        batpar[j] = calculateBatmanModel.fix_broad_fit_parameters(batpar[j], broad_fit_params)
    model = calculateBatmanModel.Spec( plot_params, lc, batpar, batmodel )
    
    red_chi2 = Utils.reduced_chisquare(lc['lcVals'], lc['lcUncs'], model, n_var)
    L_hat = np.max(mcmc_chain['flat_log_probability'])
    bic = float(n_var) * np.log( float(len(lc['bjd_tdb'])) ) - 2.*L_hat
    
    with open(os.path.join(mcmc_dir, 'fit_metrics.txt'), "w") as text_file:
        text_file.write("Reduced chi-square: " + str(red_chi2) + "\n")
        text_file.write("Maximum probability: " + str(L_hat) + "\n")
        text_file.write("BIC: " + str(bic) + "\n")
        
        if pr.ROTATION_1:
            # Calculate the radius change during transit duration
            parameter_names = list(mcmc_params['medians'].keys())
            radius_change = {}
            for k in range(len(parameter_names)):
                if parameter_names[k] == 'R0':
                    r0 = k
                elif parameter_names[k] == 'R1':
                    r1 = k
            
            # Radius change in percent per transit duration
            radius_change = mcmc_chain['flat_chain'][:,r1] / mcmc_chain['flat_chain'][:,r0] * 100. *2.9053/24.
            percentiles = np.percentile( radius_change, [16, 50, 84] )
            text_file.write("Radius change per transit duration: ({0:.2f} + {1:.2f} / - {2:.2f})% \n".format( percentiles[1], percentiles[2] - percentiles[1], percentiles[1] - percentiles[0] ) )

    