import pdb, sys, os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import xarray as xr
import Utils
import parameters as pr

"""

Routines to be run by the user:
1. white() = Generate broadband light curves, one for each detector.
2. spec() = Generate spectroscopic light curves.

TODO = Save light curves in xarray format rather than simple pickle format.

"""

DISPIXS = { 'NRS1':[300,2042], 'NRS2':[5,2010] }

##############################################################
# Top-level routines to be called by the user:

def white( spec1d ):
    """
    Generates a broadband light curve from the time series of 1D spectra.
    """
    detectors = list(spec1d.keys())
    
    print( '\nWavelength ranges:' )
    outp = {}
    for k in detectors:
        bjd = spec1d[k]['bjd']
        wavMicr = spec1d[k]['wavMicr']
        x_shift = spec1d[k]['x_shift']
        y_shift = spec1d[k]['y_shift']
        fluxVals = spec1d[k]['fluxVals']
        fluxUncs = spec1d[k]['fluxUncs']
        
        print( np.shape( wavMicr ) )
        print( np.shape( fluxVals ) )
        #plt.figure()
        #plt.plot( wavMicr, fluxVals[10,:] )
        #pdb.set_trace()
        nint, nwav = np.shape( fluxVals )
        if DISPIXS[k][1]==-1:
            DISPIXS[k][1] = nwav-1
        lcVals, lcUncs = Utils.integrateFlux( fluxVals, fluxUncs, DISPIXS[k] )
        wavEdgesMicr = [ wavMicr[DISPIXS[k][0]], wavMicr[DISPIXS[k][1]] ]
        outp[k] = { 'bjd_tdb':bjd, 'x_shift':x_shift, 'y_shift':y_shift, \
                    'lcVals':lcVals, 'lcUncs':lcUncs, 'wavEdgesMicr':wavEdgesMicr }
        
    # Cut out the transit for stand-alone analysis:
    # outp_transit = Utils.transit_cutout_white( outp )
    outp_transit = Utils.transit_and_eclipses_white( outp )
    
    # Normalize 
    for k in detectors:
        outp_transit[k]['lcVals'], outp_transit[k]['lcUncs'] = Utils.normalize_lightcurve( outp_transit[k]['lcVals'], outp_transit[k]['lcUncs'] )
    
    return outp_transit

def single( spec1d, nrs, column_range ):
    """
    Generates a broadband light curve from the time series of 1D spectra for one detector and a defined pixel bin.
    """
    
    print( '\nWavelength ranges:' )
    outp = {}
    bjd = spec1d[nrs]['bjd']
    wavMicr = spec1d[nrs]['wavMicr']
    x_shift = spec1d[nrs]['x_shift']
    y_shift = spec1d[nrs]['y_shift']
    fluxVals = spec1d[nrs]['fluxVals']
    fluxUncs = spec1d[nrs]['fluxUncs']
    
    nint, nwav = np.shape( fluxVals )
    if nrs == 'NRS2':
        lcVals, lcUncs = Utils.integrateFlux( fluxVals, fluxUncs, column_range, mask_column=293 )
    else:
        lcVals, lcUncs = Utils.integrateFlux( fluxVals, fluxUncs, column_range )
    wavEdgesMicr = [ wavMicr[column_range[0]], wavMicr[column_range[1]] ]
    outp[nrs] = { 'bjd_tdb':bjd, 'x_shift':x_shift, 'y_shift':y_shift, \
               'lcVals':lcVals, 'lcUncs':lcUncs, 'wavEdgesMicr':wavEdgesMicr }
    
    print('Single LC wavelength range:', wavEdgesMicr)
    
    # Cut out the transit for stand-alone analysis:
    # outp_transit = Utils.transit_cutout_white( outp )
    outp_transit = Utils.transit_and_eclipses_white( outp )
    
    # Normalize 
    outp_transit[nrs]['lcVals'], outp_transit[nrs]['lcUncs'] = Utils.normalize_lightcurve( outp_transit[nrs]['lcVals'], outp_transit[nrs]['lcUncs'] )
    
    return outp_transit


def spec( npix, spec1d ):
    """
    Generates a spectral light curve summed over npix pixel columns.
    """
    
    detectors = list(spec1d.keys())
    chixs = get_chixs( spec1d, npix )
    
    outp = {}
    bjd = []
    x = []
    y = []
    lcVals = []
    lcUncs = []
    wavEdges = []
    detector = []
    n_pixels = []
    for k in detectors:
        bjd_k = spec1d[k]['bjd']
        wavMicr_k = spec1d[k]['wavMicr']
        ndisp = len( wavMicr_k )
        x_shift_k = spec1d[k]['x_shift']
        y_shift_k = spec1d[k]['y_shift']
        fluxVals_k = spec1d[k]['fluxVals']
        fluxUncs_k = spec1d[k]['fluxUncs']
        nint, nwav = np.shape( fluxVals_k )
        nchan = len( chixs[k] )-1
        for i in range( nchan ):
            bjd += [ bjd_k ]
            ixL = chixs[k][i]
            # If you want to integrate the last column range to the end of the detector
            # if i == nchan-1:
            #     ixU = DISPIXS[k][1]
            # else:
            ixU = chixs[k][i+1]
            dispIxs_i = [ ixL, ixU-1 ]
            print(k, 'channel', i, 'Pixel column range:', dispIxs_i[0], 'to', dispIxs_i[1], 'Wavelength range:', wavMicr_k[ixL], 'to', wavMicr_k[ixU] )
            # print(k,i,dispIxs_i)
            if k == 'NRS2':
                lcVals_i, lcUncs_i = Utils.integrateFlux( fluxVals_k, fluxUncs_k, dispIxs_i, mask_column=293 )
            else:
                lcVals_i, lcUncs_i = Utils.integrateFlux( fluxVals_k, fluxUncs_k, dispIxs_i )
            x += [ x_shift_k ]
            y += [ y_shift_k ]
            lcVals += [ lcVals_i ]
            lcUncs += [ lcUncs_i ]
            wavEdges += [ [ wavMicr_k[ixL], wavMicr_k[ixU] ] ]
            detector += [ k ]
            n_pixels += [ npix[k] ]
            if dispIxs_i[1]-dispIxs_i[0]+1!=npix[k]:
                pdb.set_trace() # all channels should be npix-wide
    outp = {}
    outp['bjd_tdb'] = np.column_stack( bjd )
    #outp['bjd_tdb'] = np.array( bjd )
    outp['x_shift'] = np.column_stack( x )
    outp['y_shift'] = np.column_stack( y )
    outp['lcVals'] = np.column_stack( lcVals )
    outp['lcUncs'] = np.column_stack( lcUncs )
    outp['wavEdgesMicr'] = np.column_stack( wavEdges )
    outp['detector'] = np.array( detector, dtype=str )
    outp['npix'] = n_pixels
    
    # Cut out the transit for stand-alone analysis:
    outp_transit = Utils.transit_and_eclipses_spec( outp )
    
    return outp_transit


##############################################################
# Backend routines called by the top-level routines above:

def plot_white( dset='FIREFly_20221018' ):
    if dset=='FIREFly_20221018':
        ipath = pr.WHITELC_FPATH_FFLY20221018
    else:
        pdb.set_trace()
    ifile = open( ipath, 'rb' )
    white = pickle.load( ifile )
    ifile.close()
    k = 'NRS1'
    bjd = white[k]['bjd_tdb']
    flux = white[k]['lcVals']
    uncs = white[k]['lcUncs']

    thr = 24*( bjd - Tmid )
        
    plt.figure()
    plt.errorbar( thr, flux, yerr=uncs, fmt='.k' )
    plt.axvline( 0 )
    pdb.set_trace()
    return None


def get_chixs( spec1d, npix ):

    wav1 = 2.725138 # cut-on wavelength for NRS1
    wav2 = 5.172574 # cut-off wavelength for NRS2


    ixs1 = np.arange( len( spec1d['NRS1']['wavMicr'] ) )
    ix1 = ixs1[np.argmin(np.abs( spec1d['NRS1']['wavMicr']-wav1 ))]
    
    ixs2 = np.arange( len( spec1d['NRS2']['wavMicr'] ) )
    ix2 = ixs2[np.argmin(np.abs( spec1d['NRS2']['wavMicr']-wav2 ))]

    chixs = { 'NRS1':np.arange( ix1, DISPIXS['NRS1'][1], npix['NRS1'] ), 
              'NRS2':np.arange( DISPIXS['NRS2'][0], ix2+1, npix['NRS2'] ) }
    
    return chixs

