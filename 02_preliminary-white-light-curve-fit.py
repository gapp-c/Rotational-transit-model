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
def calc_residuals( pars, broad_lc, batpar, batmodel ):
    detectors = list(broad_lc.keys())
    
    residuals = np.asarray([])
    for nrs in detectors:
        model = calculateBatmanModel.Broad( pars.valuesdict(), broad_lc[nrs], batpar[nrs], batmodel[nrs], nrs )
        residuals = np.append( residuals, (broad_lc[nrs]['lcVals']-model) / broad_lc[nrs]['lcUncs'] )
    
    return residuals

######## Set directories ########
fit_out_dir = os.path.join( pr.BROAD_DIR, pr.PRELIM_DIR )

######## Input ########
broad_lc = Utils.pickle_read( os.path.join( pr.BROAD_DIR, pr.LC_FILE ) )
detectors = list(broad_lc.keys())

# First, initialize a batman model for every data point to change the transit radius as a function of time
batpar = {}
batmodel = {}
for nrs in detectors:
    batpar[nrs], batmodel[nrs] = calculateBatmanModel.create_Batman_arrays( broad_lc[nrs]['bjd_tdb'] )

# GO
p0 = initialParameters.Broad( broad_lc )

print('--------- White light curve: Running preliminary fit ---------')
prelim_fit = minimize( calc_residuals, p0, args=(broad_lc, batpar, batmodel) )
prelim_fit.params.pretty_print()

# Output
Utils.pickle_save( fit_out_dir, pr.FIT_FILE, prelim_fit )
Plot.white_light_curves( broad_lc, prelim_fit.params.valuesdict(), opath=os.path.join(fit_out_dir, 'rotational_model') )
