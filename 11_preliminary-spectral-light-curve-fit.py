#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs preliminary fits to all spectral light curve channels of JWST/NIRSpec G395H transmission spectra using lmfit.

INPUT
----------
spec_lc : dictionary
    Dictionary containing the spectral light curve channels
broad_fit : lmfit parameters
    Fit parameter results of the white light curve fit

OUTPUT
-------
spec_fits : lmfit result objects
    Results of the fits, both with fixed and free limb darkening coefficients.
spec_lc : dictionary
    Dictionary containing the spectral light curve channels after sigma clipping.
trans_spec : dictionary
    Dictionary containing the results for the fit parameters of all spectral light curve channels.
    
"""

import os
import numpy as np
from lmfit import minimize
import Utils, initialParameters, calculateBatmanModel, Plot
import parameters as pr

# This is the function we want to minimize using least squares
def calc_residuals( pars, z, batpar, pmodel ):
    return ( z['lcVals'] - calculateBatmanModel.Spec( pars.valuesdict(), z, batpar, pmodel ) ) / z['lcUncs']

######## Set directories ########
fit_in_dir = os.path.join( pr.BROAD_DIR, pr.MCMC_DIR )
broad_fit = Utils.pickle_read( os.path.join(fit_in_dir, pr.PARAMETERS_FILE) )
broad_fit_params = broad_fit['max_probabilities']

# fit_in_dir = os.path.join( pr.BROAD_DIR, pr.PRELIM_DIR )
# broad_fit = Utils.pickle_read( os.path.join(fit_in_dir, pr.FIT_FILE) )
# broad_fit_params = broad_fit.params.valuesdict()

fit_out_dir = os.path.join( pr.SPEC_DIR, pr.PRELIM_DIR )
trans_out_dir = os.path.join( pr.SPEC_DIR, pr.TRANS_DIR, pr.PRELIM_DIR )

######## Input ########
spec_lc = Utils.pickle_read( os.path.join(pr.SPEC_DIR, pr.LC_FILE) )
keys = list( spec_lc.keys() )
nchan = len( keys )

# This line only serves for getting the list of fit parameters out of the parameter function for plotting
fit_parameters = initialParameters.Spec( broad_fit_params, 'NRS1', 0.1, 0.1 )
trans_spec = Utils.prepare_transmission_dictionary( fit_parameters, nchan )

# Now, run the fits of the spectral light curves

for i, k in enumerate(keys):
    print( '--------- Working on spectral channel {0:.0f} of {1:.0f} ---------'.format( i+1, nchan ) )
    
    # Set up batman model
    batpar, batmodel = calculateBatmanModel.create_Batman_arrays( spec_lc[k]['bjd_tdb'] )
    for j in range(len(batpar)):
        batpar[j] = calculateBatmanModel.fix_broad_fit_parameters(batpar[j], broad_fit_params)
    
    # Preliminary fit for sigma clipping
    p0  = initialParameters.Spec( broad_fit_params, spec_lc[k]['detector'], spec_lc[k]['u1'], spec_lc[k]['u2'] )
    prelim_fit = minimize( calc_residuals, p0, args=(spec_lc[k], batpar, batmodel) )
    prelim_fit.params.pretty_print()
    # spec_lc[k] = Utils.sigma_clipping_spec( spec_lc[k], prelim_fit.params, broad_fit_params, pr.SIGMA_CLIPPING )
    
    # if 'outliers' in spec_lc[k].keys():
    #     if spec_lc[k]['outliers'] > 0:
    #         # Re-initiate the batman model to accomodate fewer data points
    #         batpar, batmodel = calculateBatmanModel.create_Batman_object( spec_lc[k]['bjd_tdb'] )
    #         batpar = calculateBatmanModel.fix_broad_fit_parameters( batpar, broad_fit_params )
    
    # spec_fit = minimize( calc_residuals, prelim_fit.params, args=(spec_lc[k], batpar, batmodel) )
    Utils.pickle_save( fit_out_dir, k, prelim_fit )
    
    trans_spec = Utils.add_lmfit_params_2_transmission_spectrum( trans_spec, i, spec_lc[k]['wavEdgesMicr'], prelim_fit.params )
    # trans_spec['chisquare'][i] = Utils.compare_data_and_model_spec( spec_lc[k], prelim_fit.params.valuesdict(), broad_fit_params, fit_out_dir, k )
    
    model = calculateBatmanModel.Spec( prelim_fit.params.valuesdict(), spec_lc[k], batpar, batmodel )
    Plot.spectral_light_curve( spec_lc[k], model, unit=1.0E-6, opath=os.path.join(fit_out_dir, k) )

# # Output
# Utils.pickle_save( pr.SPEC_DIR, pr.LC_SPEC_FILE_SIGMA, spec_lc )
Utils.pickle_save( trans_out_dir, pr.TRANS_FILE, trans_spec )
