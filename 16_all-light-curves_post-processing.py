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
import Utils, Plot
import parameters as pr

######## Set directories ########

prelim_dir = os.path.join( pr.SPEC_DIR, 'all_' + pr.PRELIM_DIR )
mcmc_dir = os.path.join( pr.SPEC_DIR, 'all_' + pr.MCMC_DIR )
trans_dir = os.path.join( pr.SPEC_DIR, pr.TRANS_DIR, 'all_' + pr.MCMC_DIR )

######## Input ########

lc = Utils.pickle_read( os.path.join( pr.SPEC_DIR, pr.LC_EDITED_FILE ) )
channels = list( lc.keys() )
nchan = len( channels )
prelim_fit = Utils.pickle_read( os.path.join(prelim_dir, pr.FIT_FILE) )
mcmc_chain = Utils.pickle_read( os.path.join(mcmc_dir, pr.CHAIN_FILE) )

mcmc_params = Utils.emcee_distributions_into_dictionaries( mcmc_chain )
trans_spec = Utils.simultaneous_mcmc2trans_spec( mcmc_params, lc )
Plot.corner_plot( mcmc_chain, markings=Utils.dictionary2array(mcmc_params['max_probabilities']) )
n_var = len(mcmc_params['max_probabilities'])
mcmc_params = Utils.add_fixed_parameters2dicts(mcmc_params, prelim_fit.params)

######## Output ########

Plot.all_spectral_light_curves( lc, mcmc_params['max_probabilities'], emcee_result=mcmc_chain, prelim_fit=prelim_fit.params.valuesdict(), unit=1.0E-6, opath=mcmc_dir )
Utils.pickle_save( trans_dir, pr.TRANS_FILE, trans_spec)
Utils.pickle_save( mcmc_dir, pr.PARAMETERS_FILE, mcmc_params )
Utils.write_fit_dictionary_to_csv_file(mcmc_params, os.path.join( mcmc_dir, 'white-fit-parameters') )