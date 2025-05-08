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
def calc_residuals( pars, lc, batpar, batmodel ):
    channels = list(lc.keys())
    
    residuals = np.asarray([])
    for ch in channels:
        model = calculateBatmanModel.all_lcs( pars.valuesdict(), lc[ch], batpar[ch], batmodel[ch], ch )
        residuals = np.append( residuals, (lc[ch]['lcVals']-model) / lc[ch]['lcUncs'] )
    
    return residuals

######## Set directories ########
fit_in_dir = os.path.join( pr.BROAD_DIR, pr.MCMC_DIR )
broad_fit = Utils.pickle_read( os.path.join(fit_in_dir, pr.PARAMETERS_FILE) )
broad_fit_params = broad_fit['max_probabilities']

# fit_in_dir = os.path.join( pr.BROAD_DIR, pr.PRELIM_DIR )
# broad_fit = Utils.pickle_read( os.path.join(fit_in_dir, pr.FIT_FILE) )
# broad_fit_params = broad_fit.params.valuesdict()

fit_out_dir = os.path.join( pr.SPEC_DIR, 'all_' + pr.PRELIM_DIR )

######## Input ########
lc = Utils.pickle_read( os.path.join(pr.SPEC_DIR, pr.LC_FILE) )
channels = list( lc.keys() )
nchan = len( channels )

# First, initialize a batman model for every data point to change the transit radius as a function of time
batpar = {}
batmodel = {}
for ch in channels:
    batpar[ch], batmodel[ch] = calculateBatmanModel.create_Batman_arrays( lc[ch]['bjd_tdb'] )

# GO
p0 = initialParameters.all_lcs( lc, broad_fit_params )

print('--------- All light curves: Running preliminary fit ---------')
prelim_fit = minimize( calc_residuals, p0, args=(lc, batpar, batmodel) )
prelim_fit.params.pretty_print()

Utils.pickle_save( fit_out_dir, pr.FIT_FILE, prelim_fit )
Plot.all_spectral_light_curves( lc, prelim_fit.params.valuesdict(), unit=1.0E-6, opath=fit_out_dir )
