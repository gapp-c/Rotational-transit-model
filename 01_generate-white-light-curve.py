#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates the white light curve from 1D Spectra.

"""

import os
import numpy as np
import generateTransitLightCurves, Utils, Plot
import parameters as pr
import pandas as pd
from exotic_ld import StellarLimbDarkening

# Input

if pr.EUREKA:
    detectors = ['NRS1', 'NRS2']
    broad_lc = {}
    for detector in detectors:
        broad_lc[detector] = {}
        broad_lc[detector]['bmjd_tdb'], broad_lc[detector]['lcVals'], broad_lc[detector]['lcUncs'] = \
            np.loadtxt( os.path.join(pr.EUREKA_DIR, 'WASP-121b_' + detector + '_WL_full.txt'), unpack=True )
        broad_lc[detector]['bjd_tdb'] = broad_lc[detector]['bmjd_tdb'] + 2400000.5
        
    broad_lc = Utils.transit_and_eclipses_white(broad_lc)
    
    for detector in detectors:
        broad_lc[detector] = Utils.remove_nans( broad_lc[detector] )
        broad_lc[detector]['lcVals'], broad_lc[detector]['lcUncs'] = Utils.normalize_lightcurve( broad_lc[detector]['lcVals'], broad_lc[detector]['lcUncs'] )
        
    # CORRECT THIS!!!
    broad_lc['NRS1']['wavEdgesMicr'] = [2.53724574, 3.71716004]
    broad_lc['NRS2']['wavEdgesMicr'] = [3.82349133, 5.15400743]
else:
    spec1d = Utils.read_spec1d( os.path.join(pr.SPECTRA1D_DIR, pr.SPEC1D_NRS1), os.path.join(pr.SPECTRA1D_DIR, pr.SPEC1D_NRS2) )
    broad_lc = generateTransitLightCurves.white( spec1d )

# Add standardized time variables into the data dictionaries, turn the wave edges entries into arrays and precompute stellar limb darkening parameters + uncertainties
detectors = list(broad_lc.keys())

# Input: JWST G395H throughput needed for LDC calculation
throughput = pd.read_csv( os.path.join( pr.CUSTOM_INPUT_DIR, pr.THROUGHPUT_FILE) ).values
throughput_x = 1.0e4 * throughput[:,0]          # Convert to Angstrom
throughput_y = throughput[:,1]
sld = StellarLimbDarkening( pr.METALLICITY, pr.EFFECTIVE_TEMPERATURE, pr.LOG_GRAVITY, '3D', pr.EXOTIC_DATA_DIR )

for nrs in detectors:
    broad_lc[nrs]['t'] = Utils.standardized_time( broad_lc[nrs]['bjd_tdb'] )
    broad_lc[nrs]['wavEdgesMicr'] = np.asarray( broad_lc[nrs]['wavEdgesMicr'] )
    
    values, sigmas = sld.compute_quadratic_ld_coeffs( 1.0E04*broad_lc[nrs]['wavEdgesMicr'], 'custom', throughput_x, throughput_y, return_sigmas=True )
    broad_lc[nrs]['u1'] = values[0]
    broad_lc[nrs]['u2'] = values[1]
    broad_lc[nrs]['u1_err'] = sigmas[0]
    broad_lc[nrs]['u2_err'] = sigmas[1]

Plot.lightcurves( broad_lc )
Utils.pickle_save( pr.BROAD_DIR, pr.LC_FILE, broad_lc )

# Output: Plot together with Eureka! for comparison
if not(pr.USE_ASTRONODES):
    eureka_lc = {}
    for detector in detectors:
        eureka_lc[detector] = {}
        eureka_lc[detector]['t'], eureka_lc[detector]['lcVals'], eureka_lc[detector]['lcUncs'] = np.loadtxt( os.path.join(pr.WORK_DIR, 'eureka', 'WASP-121b_' + detector + '_WL.txt'), unpack=True )
        eureka_lc[detector]['bjd_tdb'] = eureka_lc[detector]['t'] + pr.TIME_0 + pr.TRANSIT_MID_TIME
        eureka_lc[detector] = Utils.remove_nans( eureka_lc[detector] )
    if pr.SHOW_PLOTS:
        Plot.lightcurves_2versions(broad_lc, eureka_lc, opath=os.path.join(pr.WORK_DIR, 'eureka', 'reductions') )

# Utils.write_dictionary_to_csv_file({'wavelength middles [microns]' : throughput[:,0]}, 'wavelengths')

#%% Single light curves

if not(pr.EUREKA):
    for molecule in pr.SINGLE_LIGHTCURVES:
        # Input
        if molecule == 'nrs1':
            detector = 'NRS1'
            pixel_columns = [300,2042]
        else:
            detector = 'NRS2'
            if molecule == 'nrs2':
                pixel_columns = [5,2010]
            elif molecule == 'h2o':
                pixel_columns = [5,250]
            elif molecule == 'sio':
                pixel_columns = [260,717]
            elif molecule == 'co':
                pixel_columns = [718,2010]
            else:
                print('UNKNOWN MOLECULE')
                break
    
        spec1d = Utils.read_spec1d( os.path.join(pr.SPECTRA1D_DIR, pr.SPEC1D_NRS1), os.path.join(pr.SPECTRA1D_DIR, pr.SPEC1D_NRS2) )
        throughput = pd.read_csv( os.path.join( pr.CUSTOM_INPUT_DIR, pr.THROUGHPUT_FILE) ).values
        throughput_x = 1.0e4 * throughput[:,0]
        throughput_y = throughput[:,1]
        sld = StellarLimbDarkening( pr.METALLICITY, pr.EFFECTIVE_TEMPERATURE, pr.LOG_GRAVITY, '3D', pr.EXOTIC_DATA_DIR )
    
        # Integrate light curve and add meta data
        single_lc = generateTransitLightCurves.single( spec1d, detector, pixel_columns )
        lc = single_lc[detector]
        lc['t'] = Utils.standardized_time( lc['bjd_tdb'] )
    
        lc['detector'] = detector
        lc['wavEdgesMicr'] = np.asarray( lc['wavEdgesMicr'] )
        values, sigmas = sld.compute_quadratic_ld_coeffs( 1.0E04*lc['wavEdgesMicr'], 'custom', throughput_x, throughput_y, return_sigmas=True )
        lc['u1'] = values[0]
        lc['u2'] = values[1]
        lc['u1_err'] = sigmas[0]
        lc['u2_err'] = sigmas[1]
    
        if pr.SHOW_PLOTS:
            Plot.lightcurve( lc )
        Utils.pickle_save( os.path.join(pr.RESULTS_DIR, molecule), pr.LC_FILE, lc )
        print(molecule, lc['wavEdgesMicr'], ', LDC:', lc['u1'], lc['u2'])
    
