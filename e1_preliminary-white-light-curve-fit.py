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
fit_out_dir = os.path.join( pr.BROAD_DIR, 'eureka', pr.PRELIM_DIR )

######## Input ########

detectors = ['NRS1', 'NRS2']
broad_lc = {}
for detector in detectors:
    broad_lc[detector] = {}
    broad_lc[detector]['bmjd_tdb'], broad_lc[detector]['lcVals'], broad_lc[detector]['lcUncs'] = \
        np.loadtxt( os.path.join(pr.EUREKA_DIR, 'WASP-121b_' + detector + '_WL_full.txt'), unpack=True )
    broad_lc[detector]['bjd_tdb'] = broad_lc[detector]['bmjd_tdb'] + 2400000.5

broad_lc = Utils.transit_and_eclipses_white(broad_lc)

for detector in detectors:
    broad_lc[detector]['t'] = Utils.standardized_time( broad_lc[detector]['bjd_tdb'] )
    broad_lc[detector] = Utils.remove_nans( broad_lc[detector] )
    broad_lc[detector]['lcVals'], broad_lc[detector]['lcUncs'] = Utils.normalize_lightcurve( broad_lc[detector]['lcVals'], broad_lc[detector]['lcUncs'] )

broad_lc['NRS1']['u1'] = 0.06585945520225178
broad_lc['NRS1']['u2'] = 0.11265248382653993
broad_lc['NRS1']['u1_err'] = 0.011569608991181486
broad_lc['NRS1']['u2_err'] = 0.015043755034295784

broad_lc['NRS2']['u1'] = 0.05605071953777346
broad_lc['NRS2']['u2'] = 0.09230747753839463
broad_lc['NRS2']['u1_err'] = 0.009485355297039812
broad_lc['NRS2']['u2_err'] = 0.012333638781348886

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

# Calculate the best-fit model for plotting
model = {}
for nrs in detectors:
    model[nrs] = calculateBatmanModel.Broad( prelim_fit.params.valuesdict(), broad_lc[nrs], batpar[nrs], batmodel[nrs], nrs )

# Output
Utils.pickle_save( os.path.join(pr.BROAD_DIR, 'eureka'), pr.LC_FILE, broad_lc )
Utils.pickle_save( fit_out_dir, pr.FIT_FILE, prelim_fit )
Plot.white_light_curves( broad_lc, prelim_fit.params.valuesdict(), opath=os.path.join(fit_out_dir, 'lcfit') )
