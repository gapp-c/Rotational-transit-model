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

import os
import numpy as np
import Utils, Plot, calculateBatmanModel
import parameters as pr


######## Set directories ########

fit_in_dir = os.path.join( pr.BROAD_DIR, 'eureka', pr.PRELIM_DIR )
mcmc_dir = os.path.join( pr.BROAD_DIR, 'eureka', pr.MCMC_DIR )

######## Input ########

broad_lc = Utils.pickle_read( os.path.join(pr.BROAD_DIR, 'eureka', pr.LC_FILE) )
detectors = list(broad_lc.keys())
prelim_fit = Utils.pickle_read( os.path.join( fit_in_dir, pr.FIT_FILE) )
mcmc_result = Utils.pickle_read( os.path.join( mcmc_dir, pr.CHAIN_FILE) )

######## Go ########

detectors = list(broad_lc.keys())

# Diagnostic plot
# Plot.emcee_acceptance_fraction( mcmc_result['acceptance_fraction'], mcmc_dir )

mcmc_params = Utils.emcee_distributions_into_dictionaries( mcmc_result )
if pr.SAVE_PLOTS:
    Plot.corner_plot( mcmc_result, Utils.dictionary2array(mcmc_params['max_probabilities']), os.path.join(mcmc_dir, 'white_corner') )
n_var = len(mcmc_params['max_probabilities'])
mcmc_params = Utils.add_fixed_parameters2dicts(mcmc_params, prelim_fit.params)
Plot.white_light_curves(broad_lc, mcmc_params['max_probabilities'], emcee_result=mcmc_result, opath=os.path.join(mcmc_dir, 'white-fit'))
Utils.pickle_save( mcmc_dir, pr.PARAMETERS_FILE, mcmc_params )
Utils.write_fit_dictionary_to_csv_file(mcmc_params, os.path.join( mcmc_dir, 'white-fit-parameters') )
    
######## Summary of results ########

batpar = {}
batmodel = {}
model_alldetectors = np.array([])
uncertainties_alldetectors = np.array([])
data_alldetectors = np.array([])
for nrs in detectors:
    batpar[nrs], batmodel[nrs] = calculateBatmanModel.create_Batman_arrays( broad_lc[nrs]['bjd_tdb'] )
    model = calculateBatmanModel.Broad( mcmc_params['max_probabilities'], broad_lc[nrs], batpar[nrs], batmodel[nrs], nrs )
    
    model_alldetectors = np.append(model_alldetectors, model)
    data_alldetectors = np.append(data_alldetectors, broad_lc[nrs]['lcVals'])
    uncertainties_alldetectors = np.append(uncertainties_alldetectors, broad_lc[nrs]['lcUncs'])

red_chi2 = Utils.reduced_chisquare(data_alldetectors, uncertainties_alldetectors, model_alldetectors, n_var)
L_hat = np.max(mcmc_result['flat_log_probability'])
bic = float(n_var) * np.log( float(len(data_alldetectors)) ) - 2.*L_hat

with open(os.path.join(mcmc_dir, 'fit_metrics.txt'), "w") as text_file:
    text_file.write("Reduced chi-square: " + str(red_chi2) + "\n")
    text_file.write("Maximum probability: " + str(L_hat) + "\n")
    text_file.write("BIC: " + str(bic) + "\n")
    
    # Calculate the radius change during transit duration
    if pr.ROTATION_1:
        parameter_names = list(mcmc_params['medians'].keys())
        for detector in detectors:
            
            for i in range(len(parameter_names)):
                if parameter_names[i] == 'R0_' + detector:
                    r0 = i
                elif parameter_names[i] == 'R1_' + detector:
                    r1 = i
        
            # Radius change in percent per transit duration
            radius_change = mcmc_result['flat_chain'][:,r1] / mcmc_result['flat_chain'][:,r0] * 100. *2.9053/24.
            # Plot.histogram( radius_change )
            percentiles = np.percentile( radius_change, [16, 50, 84] )
            text_file.write(detector + ": Radius change per transit duration: ({0:.2f} + {1:.2f} / - {2:.2f})\% \n".format( percentiles[1], percentiles[2] - percentiles[1], percentiles[1] - percentiles[0] ) )
        
