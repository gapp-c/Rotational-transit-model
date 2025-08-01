#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates the spectral light curve from 1D Spectra.
    
"""

import generateTransitLightCurves, Utils, Plot
import parameters as pr
import os
import numpy as np
import pandas as pd
import astraeus.xarrayIO as xrio
from exotic_ld import StellarLimbDarkening
if not(pr.USE_ASTRONODES):
    from tqdm import tqdm

if pr.EUREKA:
    
    detectors = ['NRS1', 'NRS2']
    spec_lc = {}
    for detector in detectors:
        spec_lc[detector] = Utils.extract_spec_lc(xrio.readXR( os.path.join( pr.EUREKA_DIR, 'WASP-121b_Eureka_PC_lightcurves_' + detector + '_firefly_bins' )), detector )
    spec_lc_edit = Utils.merge_spectral_lightcurves( spec_lc )
    spec_lc_edit = Utils.transit_and_eclipses_white( spec_lc_edit )
    
else:
    
    # Input
    spec1d = Utils.read_spec1d( os.path.join(pr.SPECTRA1D_DIR, pr.SPEC1D_NRS1), os.path.join(pr.SPECTRA1D_DIR, pr.SPEC1D_NRS2) )
    
    spec_lc = generateTransitLightCurves.spec( pr.NUMBER_OF_PIXEL_COLUMNS, spec1d )
    nint, nchan = np.shape( spec_lc['lcVals'] )
    
    # ld_parameters = {}
    # for k in ['channel', 'lower_wavelength', 'upper_wavelength', 'u1', 'u1_err', 'u2', 'u2_err']:
        #     ld_parameters[k] = np.zeros( nchan ) + np.nan
    
    print('\n Rearranging data format and computing limb darkening coefficients...')
    if not(pr.USE_ASTRONODES):
        indeces = tqdm(range(nchan))
    else:
        indeces = range(nchan)
    
    spec_lc_edit = {}
    for j in indeces:
        key = 'ch{0:.0f}of{1:.0f}'.format( j+1, nchan )
        spec_lc_edit[key] = {}
        # spec_lc_edit[key]['t'] = spec_lc['bjd_tdb'][:,j] - 2459867.642577
        for k in list(spec_lc.keys()):
            # Scalar values
            if ( k == 'detector' or  k == 'npix' ):
                spec_lc_edit[key][k] = spec_lc[k][j]
            # 2D arrays
            else:
                print(k,j)
                spec_lc_edit[key][k] =  spec_lc[k][:,j]

# Add LDCs

# JWST G395H throughput needed for LDC calculation
throughput = pd.read_csv( os.path.join( pr.CUSTOM_INPUT_DIR, pr.THROUGHPUT_FILE) ).values
throughput_x = 1.0e4 * throughput[:,0]          # Convert to Angstrom
throughput_y = throughput[:,1]

sld = StellarLimbDarkening( pr.METALLICITY, pr.EFFECTIVE_TEMPERATURE, pr.LOG_GRAVITY, '3D', pr.EXOTIC_DATA_DIR )

for key in spec_lc_edit.keys():
    values, sigmas = sld.compute_quadratic_ld_coeffs( 1.0E04*spec_lc_edit[key]['wavEdgesMicr'], 'custom', throughput_x, throughput_y, return_sigmas=True )
    
    # ld_parameters['channel'][j] = j + 1
    # ld_parameters['lower_wavelength'][j] = spec_lc['wavEdgesMicr'][0,j]
    # ld_parameters['upper_wavelength'][j] = spec_lc['wavEdgesMicr'][1,j]
    
    # ld_parameters['u1'][j] = values[0]
    # ld_parameters['u2'][j] = values[1]
    # ld_parameters['u1_err'][j] = sigmas[0]
    # ld_parameters['u2_err'][j] = sigmas[1]
    
    # # Add LDC to the light curve data dictionary
    # for k in ['u1', 'u1_err', 'u2', 'u2_err']:
    #     spec_lc_edit[key][k] = ld_parameters[k][j]
    
    spec_lc_edit[key]['u1'] = values[0]
    spec_lc_edit[key]['u2'] = values[1]
    spec_lc_edit[key]['u1_err'] = sigmas[0]
    spec_lc_edit[key]['u2_err'] = sigmas[1]
    
    # Normalize light curve
    spec_lc_edit[key]['lcVals'], spec_lc_edit[key]['lcUncs'] = Utils.normalize_lightcurve( spec_lc_edit[key]['lcVals'], spec_lc_edit[key]['lcUncs'] )
    spec_lc_edit[key]['t'] = Utils.standardized_time( spec_lc_edit[key]['bjd_tdb'] )


if pr.SHOW_PLOTS:
#     Plot.limb_darkening_coefficients( ld_parameters, os.path.join(pr.SPEC_DIR, pr.LD_FILE))#,  old_ld )
    keys = list(spec_lc_edit.keys())
    plot_lcs = { keys[0] : spec_lc_edit[keys[0]], keys[1] : spec_lc_edit[keys[1]] }

    Plot.lightcurves( plot_lcs )

Utils.pickle_save( pr.SPEC_DIR, pr.LC_FILE, spec_lc_edit )
# Utils.write_dictionary_to_csv_file( ld_parameters, os.path.join(pr.SPEC_DIR, pr.LD_FILE) )

# #%% Calculate spectral resolution

# keys = list(spec_lc_edit.keys())
# resolutions = np.zeros(len(keys))
# i = 0
# for i, key in enumerate(keys):
#     resolutions[i] = (spec_lc_edit[key]['wavEdgesMicr'][1] + spec_lc_edit[key]['wavEdgesMicr'][0]) / 2. / (spec_lc_edit[key]['wavEdgesMicr'][1] - spec_lc_edit[key]['wavEdgesMicr'][0])
#     print(key, resolutions[i], spec_lc_edit[key]['wavEdgesMicr'][0], spec_lc_edit[key]['wavEdgesMicr'][1])
    