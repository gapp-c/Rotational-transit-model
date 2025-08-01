#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to calculating transit models using Batman
"""

import batman
import numpy as np
import parameters as pr
import Utils

def create_Batman_object( bjd ):
    """
    Routine for generating a batman parameters object, which is subsequently 
    used to compute transit light curves in the model fitting.
    """
    
    # Parameters fixed to literature values
    batpar = batman.TransitParams()
    batpar.per = pr.ORBITAL_PERIOD['median']
    batpar.ecc = pr.ORBITAL_ECCENTRICITY
    batpar.w = pr.ARGUMENT_OF_PERIAPSIS
    batpar.limb_dark = 'quadratic'
    
    # Parameters that will have to be changed
    batpar.t0 = pr.TRANSIT_MID_TIME
    batpar.rp = pr.PHASECURVE_RP_RS['NRS2']
    batpar.a = pr.SEMIMAJOR_AXIS_OVER_STELLAR_RADIUS['median']
    batpar.inc = pr.ORBITAL_INCLINATION['median']
    batpar.u = [ 0, 0 ]
    
    pmodel = batman.TransitModel( batpar, bjd, transittype='primary' )
    batmodel = batman.TransitModel( batpar, bjd, fac=pmodel.fac, transittype='primary' )
    
    return batpar, batmodel

def create_Batman_arrays( bjd ):
    
    batpar = []
    batmodel = []
    for i in range(len( bjd )):
        bp, bm = create_Batman_object( bjd[i:i+1] )
        batpar += [bp]
        batmodel += [bm]
    
    return batpar, batmodel

def Broad( pars, z, batpar, batmodel, nrs ):
    """
    Returns the model for one of the NRS detectors.
    """
    
    if pr.BASELINE_BROAD == 'constant':
        baseline = pars['baseline_' + nrs]
    elif pr.BASELINE_BROAD == 'linear':
        baseline = pars['c0_' + nrs] + pars['c1_' + nrs]*z['t']
    elif pr.BASELINE_BROAD == 'quadratic':
        baseline = pars['c0_' + nrs] + pars['c1_' + nrs]*z['t'] + pars['c2_' + nrs]*z['t']**2
    
    n = len(z['t'])
    
    if pr.FIT_TMID_SPEC or pr.FIT_TMID_DETECTORS:
        Tmid = pars['Tmid_' + nrs] + pr.TIME_0
    else:
        Tmid = pars['Tmid'] + pr.TIME_0
    
    if pr.FIT_ROTATION_2_SPEC:
        R2 = pars['R2_' + nrs]
    else:
        R2 = pars['R2']
    
    transit_model = np.zeros(n)
    for i in range(n):
        if pr.KIPPING_LD_PARAMETERS:
            u1, u2 = Utils.LD_coefficients_q2u( pars['q1_' + nrs], pars['q2_' + nrs] )
            batpar[i].u = [ u1, u2 ]
        else:
            batpar[i].u = [ pars['u1_' + nrs], pars['u2_' + nrs] ]
        
        batpar[i].t0 = Tmid
        batpar[i].a = pars['aRs']
        if pr.IMPACT_PARAMETER:
            batpar[i].inc = Utils.impact_parameter2inc(pars['b'], pars['aRs'])
        else:
            batpar[i].inc = pars['inc']
        rprs = pars['R0_' + nrs]
        if pr.ROTATION_1:
            rprs += pars['R1_' + nrs] * (z['bjd_tdb'][i]-Tmid)
            if pr.ROTATION_2:
                rprs += R2 * (z['bjd_tdb'][i]-Tmid)**2
        batpar[i].rp = rprs
        
        transit_model[i] = batmodel[i].light_curve( batpar[i] )
    
    partial_phasecurve = np.zeros(n)
    for i in range(n):
        if z['t'][i] > 0.2 and z['t'][i] < 1.0:
            if pr.EMISSION == 'polynomial':
                partial_phasecurve[i] = pars['p0_' + nrs] + pars['p1_' + nrs]*z['t'][i] + pars['p2_' + nrs]*z['t'][i]**2
            elif pr.EMISSION == 'cosine':
                partial_phasecurve[i] = pars['B_' + nrs] + pars['A_' + nrs] * np.cos( 2.*np.pi*z['t'][i]/pr.ORBITAL_PERIOD['median'] + pars['phi_' + nrs])
            else:
                partial_phasecurve[i] = 0.
    
    return baseline * (transit_model + partial_phasecurve)

def Spec( pars, z, batpar, batmodel ):
    
    if pr.BASELINE_SPEC == 'constant':
        baseline = pars['baseline']
    elif pr.BASELINE_SPEC == 'linear':
        baseline = pars['c0'] + pars['c1']*z['t']
    elif pr.BASELINE_SPEC == 'quadratic':
        baseline = pars['c0'] + pars['c1']*z['t'] + pars['c2']*z['t']**2
    
    n = len(z['t'])
    Tmid = pars['Tmid'] + pr.TIME_0
    transit_model = np.zeros(n)
    for i in range(n):
        batpar[i].t0 = Tmid
        
        if pr.FIT_ORBIT_SPEC:
            batpar[i].a = pars['aRs']
            if pr.IMPACT_PARAMETER:
                batpar[i].inc = Utils.impact_parameter2inc(pars['b'], pars['aRs'])
            else:
                batpar[i].inc = pars['inc']
                
        rprs = pars['R0']
        if pr.ROTATION_1:
            rprs += pars['R1'] * (z['bjd_tdb'][i]-Tmid)
            if pr.ROTATION_2:
                rprs += pars['R2'] * (z['bjd_tdb'][i]-Tmid)**2
        batpar[i].rp = rprs
        
        if pr.KIPPING_LD_PARAMETERS:
            u1, u2 = Utils.LD_coefficients_q2u( pars['q1'], pars['q2'] )
            batpar[i].u = [ u1, u2 ]
        else:
            batpar[i].u = [ pars['u1'], pars['u2'] ]
        
        transit_model[i] = batmodel[i].light_curve( batpar[i] )
    
    partial_phasecurve = np.zeros(n)
    for i in range(n):
        if z['t'][i] > 0.2 and z['t'][i] < 1.0:
            partial_phasecurve[i] = pars['p0'] + pars['p1']*z['t'][i] + pars['p2']*z['t'][i]**2
    
    return baseline * (transit_model + partial_phasecurve)

def Single_soss( pars, z, batpar, batmodel ):
    
    t = z['t']
    # print(t)
    baseline = pars['c0'] + pars['c1']*t
    
    n = len(t)
    Tmid = pars['Tmid'] + 2460244.
    transit_model = np.zeros(n)
    for i in range(n):
        batpar[i].t0 = Tmid
        batpar[i].a = pars['aRs']
        if pr.IMPACT_PARAMETER:
            batpar[i].inc = Utils.impact_parameter2inc(pars['b'], pars['aRs'])
        else:
            batpar[i].inc = pars['inc']
        
        rprs = pars['R0'] + pars['R1'] * (z['bjd_tdb'][i]-Tmid) + pars['R2'] * (z['bjd_tdb'][i]-Tmid)**2
        batpar[i].rp = rprs
        
        if pr.KIPPING_LD_PARAMETERS:
            u1, u2 = Utils.LD_coefficients_q2u( pars['q1'], pars['q2'] )
            batpar[i].u = [ u1, u2 ]
        else:
            batpar[i].u = [ pars['u1'], pars['u2'] ]
        
        transit_model[i] = batmodel[i].light_curve( batpar[i] )
    
    partial_phasecurve = np.zeros(n)
    for i in range(n):
        if t[i] > 0.6 and t[i] < 1.4:
            partial_phasecurve[i] = pars['p0'] + pars['p1']*t[i] + pars['p2']*t[i]**2
    
    return baseline * (transit_model + partial_phasecurve)

def Single( pars, z, batpar, batmodel ):
    
    baseline = pars['c0'] + pars['c1']*z['t']
    
    n = len(z['t'])
    Tmid = pars['Tmid'] + pr.TIME_0
    transit_model = np.zeros(n)
    for i in range(n):
        batpar[i].t0 = Tmid
        batpar[i].a = pars['aRs']
        if pr.IMPACT_PARAMETER:
            batpar[i].inc = Utils.impact_parameter2inc(pars['b'], pars['aRs'])
        else:
            batpar[i].inc = pars['inc']
        
        rprs = pars['R0'] + pars['R1'] * (z['bjd_tdb'][i]-Tmid) + pars['R2'] * (z['bjd_tdb'][i]-Tmid)**2
        batpar[i].rp = rprs
        
        if pr.KIPPING_LD_PARAMETERS:
            u1, u2 = Utils.LD_coefficients_q2u( pars['q1'], pars['q2'] )
            batpar[i].u = [ u1, u2 ]
        else:
            batpar[i].u = [ pars['u1'], pars['u2'] ]
        
        transit_model[i] = batmodel[i].light_curve( batpar[i] )
    
    partial_phasecurve = np.zeros(n)
    for i in range(n):
        if z['t'][i] > 0.2 and z['t'][i] < 1.0:
            partial_phasecurve[i] = pars['p0'] + pars['p1']*z['t'][i] + pars['p2']*z['t'][i]**2
    
    return baseline * (transit_model + partial_phasecurve)

def all_lcs( pars, z, batpar, batmodel, ch ):
    """
    Returns the model for one of the NRS detectors.
    """
    
    if pr.BASELINE_BROAD == 'constant':
        baseline = pars['baseline_' + ch]
    elif pr.BASELINE_BROAD == 'linear':
        baseline = pars['c0_' + ch] + pars['c1_' + ch]*z['t']
    elif pr.BASELINE_BROAD == 'quadratic':
        baseline = pars['c0_' + ch] + pars['c1_' + ch]*z['t'] + pars['c2_' + ch]*z['t']**2
    
    n = len(z['t'])
    
    if pr.FIT_TMID_SPEC:
        Tmid = pars['Tmid_' + ch] + pr.TIME_0
    elif pr.FIT_TMID_DETECTORS:
        Tmid = pars['Tmid_' + z['detector']] + pr.TIME_0
    else:
        Tmid = pars['Tmid'] + pr.TIME_0
    
    if pr.FIT_ROTATION_2_SPEC:
        R2 = pars['R2_' + ch]
    else:
        R2 = pars['R2']
    
    transit_model = np.zeros(n)
    for i in range(n):
        if pr.KIPPING_LD_PARAMETERS:
            u1, u2 = Utils.LD_coefficients_q2u( pars['q1_' + ch], pars['q2_' + ch] )
            batpar[i].u = [ u1, u2 ]
        else:
            batpar[i].u = [ pars['u1_' + ch], pars['u2_' + ch] ]
            
        batpar[i].t0 = Tmid
        batpar[i].a = pars['aRs']
        if pr.IMPACT_PARAMETER:
            batpar[i].inc = Utils.impact_parameter2inc(pars['b'], pars['aRs'])
        else:
            batpar[i].inc = pars['inc']
        rprs = pars['R0_' + ch]
        if pr.ROTATION_1:
            rprs += pars['R1_' + ch] * (z['bjd_tdb'][i]-Tmid)
            if pr.ROTATION_2:
                rprs += R2 * (z['bjd_tdb'][i]-Tmid)**2
        batpar[i].rp = rprs
        
        transit_model[i] = batmodel[i].light_curve( batpar[i] )
    
    partial_phasecurve = np.zeros(n)
    for i in range(n):
        if z['t'][i] > 0.2 and z['t'][i] < 1.0:
            partial_phasecurve[i] = pars['p0_' + ch] + pars['p1_' + ch]*z['t'][i] + pars['p2_' + ch]*z['t'][i]**2
    
    return baseline * (transit_model + partial_phasecurve)

def fix_broad_fit_parameters( batpar, broad_fit_params ):
    """
    Fixes the fit parameters from the white light curve fit that are not included in the spectroscopic light curve fits.

    Parameters
    ----------
    batpar
        batman parameters object
    broad_fit_params : dict
        White light curve fit parameters

    Returns
    -------
    batpar
        batman parameters object with fixed white light curve fit parameters
    
    """
    
    if pr.IMPACT_PARAMETER:
        batpar.inc = np.rad2deg( np.arccos( broad_fit_params['b']/broad_fit_params['aRs'] ) )
    else:
        batpar.inc = broad_fit_params['inc']
    batpar.a = broad_fit_params['aRs']
    
    return batpar

