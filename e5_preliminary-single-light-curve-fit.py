#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs a preliminary fit to the white light curves of JWST/NIRSpec G395H using lmfit.

INPUT
----------
broad_lc : dictionary
    Dictionary containing the broad band light curves of both NRS1 & NRS2

OUTPUT
-------
broad_fits : lmfit result objects
    Results of the simulatneous fit, both with fixed and free limb darkening coefficients.
broad_lc : dictionary
    Dictionary containing the broad band light curves of both NRS1 & NRS2 after sigma clipping
    
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
fit_in_dir = os.path.join( pr.BROAD_DIR, 'eureka', pr.MCMC_DIR )
model_colors = ['blue', 'orange' ]

broad_fit = Utils.pickle_read( os.path.join(fit_in_dir, pr.PARAMETERS_FILE) )
broad_fit_params = broad_fit['max_probabilities']

######## Input ########
for i, molecule in enumerate(pr.MOLECULES):
    fit_out_dir = os.path.join( pr.RESULTS_DIR, molecule, 'eureka', pr.PRELIM_DIR )
    
    lc = {}
    lc['t'], lc['lcVals'], lc['lcUncs'] = np.loadtxt( os.path.join(pr.WORK_DIR, 'eureka', 'WASP-121b_' + molecule + '.txt'), unpack=True )
    lc['bjd_tdb'] = lc['t'] + pr.TIME_0 + pr.TRANSIT_MID_TIME
    if molecule == 'h2o':
        lc['detector'] = 'NRS1'
        lc['u1'] = 0.06585945520225178
        lc['u2'] = 0.11265248382653993
    else:
        lc['detector'] = 'NRS2'
        lc['u1'] = 0.05313422752428942
        lc['u2'] = 0.09372893834665667
    lc = Utils.remove_nans( lc )
    
    # First, initialize a batman model for every data point to change the transit radius as a function of time
    batpar, batmodel = calculateBatmanModel.create_Batman_arrays( lc['bjd_tdb'] )
    for j in range(len(batpar)):
        batpar[j] = calculateBatmanModel.fix_broad_fit_parameters(batpar[j], broad_fit_params)
    
    # GO
    p0 = initialParameters.Spec( broad_fit_params, lc['detector'], lc['u1'], lc['u2'] )
    
    print('---------', molecule, 'light curve: Rotational fit ---------')
    rotational_fit = minimize( calc_residuals, p0, args=(lc, batpar, batmodel) )
    
    # Calculate the best-fit model for plotting
    rotational_model = calculateBatmanModel.Spec( rotational_fit.params.valuesdict(), lc, batpar, batmodel )
    rotational_fit.params.pretty_print()
    Utils.pickle_save( fit_out_dir, pr.FIT_FILE, rotational_fit )
    
    # Also run the translational fit for comparison
    p0['R1'].value = 0.
    p0['R1'].vary = False
    print('---------', molecules[i], 'light curve: Translational fit ---------')
    translational_fit = minimize( calc_residuals, p0, args=(lc, batpar, batmodel) )
    translational_model = calculateBatmanModel.Spec( translational_fit.params.valuesdict(), lc, batpar, batmodel )
    translational_fit.params.pretty_print()
    
    print('Reduced chi square, rotational model:', Utils.reduced_chisquare(lc['lcVals'], lc['lcUncs'], rotational_model, 8))
    print('Reduced chi square, translational model:', Utils.reduced_chisquare(lc['lcVals'], lc['lcUncs'], translational_model, 7))
    Plot.spectral_light_curve( lc, rotational_model, unit=1.0E-6, opath=os.path.join(fit_out_dir, 'rotational_model'), label=molecules[i], model_color=model_colors[i] )
    Plot.spectral_light_curve_nmodels( lc, {'Translational model' : translational_model, 'Rotational model' : rotational_model}, unit=1.0E-6, opath=os.path.join(fit_out_dir, 'lc_models'), label=molecules[i], model_color=model_colors[i] )
