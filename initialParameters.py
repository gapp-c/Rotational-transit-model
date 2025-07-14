#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:08:57 2023

@author: gapp
"""

import numpy as np
from lmfit import Parameters
import parameters as pr
import Utils

def Broad( broad_lc ):
    """
    Puts the initial guesses for model parameters of the broad band light curves into a dictionary.
    """
    detectors = list(broad_lc.keys())
    
    pars = Parameters()
    pars.add('aRs', value = pr.SEMIMAJOR_AXIS_OVER_STELLAR_RADIUS['median'] )
    if pr.IMPACT_PARAMETER:
        pars.add('b', value = Utils.inc2impact_parameter(pr.ORBITAL_INCLINATION['median'], pr.SEMIMAJOR_AXIS_OVER_STELLAR_RADIUS['median']), min=0., max=1. )
    else:
        pars.add('inc', value = pr.ORBITAL_INCLINATION['median'], max=90. )
    
    if not(pr.FIT_TMID_SPEC or pr.FIT_TMID_DETECTORS):
        pars.add('Tmid', value = pr.TRANSIT_MID_TIME)
    if not(pr.FIT_ROTATION_2_SPEC):
        pars.add('R2', value = 0.0, vary=pr.ROTATION_2)
        if pr.ROTATION_2_NEGATIVE:
            pars['R2'].max = 0.0
    
    # Loop for parameters that are different between both detectors
    for nrs in detectors:
        
        if pr.FIT_TMID_SPEC or pr.FIT_TMID_DETECTORS:
            pars.add('Tmid_' + nrs, value = pr.TRANSIT_MID_TIME)
        
        pars.add('R0_' + nrs, value = pr.PHASECURVE_RP_RS[nrs] )
        pars.add('R1_' + nrs, value = 0.0, vary=pr.ROTATION_1 )
        if pr.FIT_ROTATION_2_SPEC:
            pars.add('R2_' + nrs, value = 0.0, vary=pr.ROTATION_2)
            if pr.ROTATION_2_NEGATIVE:
                pars['R2_' + nrs].max = 0.0
        
        if pr.BASELINE_BROAD == 'constant':
            pars.add('baseline_' + nrs, value=pr.BASELINE_C0 )
        elif pr.BASELINE_BROAD == 'linear':
            pars.add('c0_' + nrs, value=pr.BASELINE_C0 )
            pars.add('c1_' + nrs, value=pr.BASELINE_C1 )
        elif pr.BASELINE_BROAD == 'quadratic':
            pars.add('c0_' + nrs, value=pr.BASELINE_C0 )
            pars.add('c1_' + nrs, value=pr.BASELINE_C1 )
            pars.add('c2_' + nrs, value=pr.BASELINE_C2 )
        
        # Limb darkening coefficients
        if pr.KIPPING_LD_PARAMETERS:
            q1, q2 = Utils.LD_coefficients_u2q(broad_lc[nrs]['u1'], broad_lc[nrs]['u2'])
            pars.add('q1_' + nrs, value=q1, min = 0., max = 1., vary = pr.FIT_1ST_LD_PARAMETER )
            pars.add('q2_' + nrs, value=q2, min = 0., max = 1., vary = pr.FIT_2ND_LD_PARAMETER )
        else:
            pars.add('u1_' + nrs, value=broad_lc[nrs]['u1'], vary = pr.FIT_1ST_LD_PARAMETER )
            pars.add('u2_' + nrs, value=broad_lc[nrs]['u2'], vary = pr.FIT_2ND_LD_PARAMETER )
        
        # Partial phasecurve
        pars.add('p0_' + nrs, value=0.1 )
        pars.add('p1_' + nrs, value=0.1 )
        pars.add('p2_' + nrs, value=0.1 )
        
    return pars

def set_uninformative_priors( params ):
    
    parameters = list(params.keys())
    
    priors = {}
    for parameter in parameters:
        if params[parameter].vary:
            priors[parameter] = {}
            priors[parameter]['prior'] = 'uninformative'
            
    return priors

def lmfitparams2priors( params ):
    
    parameters = list(params.keys())
    
    priors = {}
    for parameter in parameters:
        if params[parameter].vary:
            priors[parameter] = {}
            if np.isfinite(params[parameter].min) or np.isfinite(params[parameter].max):
                priors[parameter]['prior'] = 'boxcar'
                priors[parameter]['lower_limit'] = params[parameter].min
                priors[parameter]['upper_limit'] = params[parameter].max
            else:
                priors[parameter]['prior'] = 'uninformative'
            
    return priors
    
def dynesty_uniform_priors( p0 ):
    
    priors = {}
    for par in p0:
        if p0[par].vary:
            priors[par] = { 'prior' : 'uniform'}            
            if par[:2] == 'R0':
                priors[par]['lower_limit'] = 0.05
                priors[par]['upper_limit'] = 0.2
            elif par[:2] == 'R1':
                priors[par]['lower_limit'] = -1.
                priors[par]['upper_limit'] = 1.
            elif par[:2] == 'R2':
                priors[par]['lower_limit'] = -10.
                priors[par]['upper_limit'] = 10.
                if pr.ROTATION_2_NEGATIVE:
                    priors[par]['upper_limit'] = 0.
            elif par[:4] == 'Tmid':
                priors[par]['lower_limit'] = 0.4
                priors[par]['upper_limit'] = 0.8
            elif par[:2] == 'c0':
                priors[par]['lower_limit'] = 0.
                priors[par]['upper_limit'] = 2.
            elif par[:2] == 'c1':
                priors[par]['lower_limit'] = -1.
                priors[par]['upper_limit'] = 1.
            elif par[:2] == 'c2':
                priors[par]['lower_limit'] = -2.
                priors[par]['upper_limit'] = 2.
            elif par[:2] == 'p0':
                priors[par]['lower_limit'] = -1.
                priors[par]['upper_limit'] = 1.
            elif par[:2] == 'p1':
                priors[par]['lower_limit'] = -1.
                priors[par]['upper_limit'] = 1.
            elif par[:2] == 'p2':
                priors[par]['lower_limit'] = -1.
                priors[par]['upper_limit'] = 1.
            elif par[:2] == 'u1':
                priors[par]['lower_limit'] = -3.
                priors[par]['upper_limit'] = 3.
            elif par[:2] == 'u2':
                priors[par]['lower_limit'] = -3.
                priors[par]['upper_limit'] = 3.
            elif par[:9] == 'sigma_sys':
                priors[par]['lower_limit'] = 0.
                priors[par]['upper_limit'] = 1000.
            elif par == 'aRs':
                priors[par]['lower_limit'] = 2.
                priors[par]['upper_limit'] = 5.
            elif par == 'inc':
                priors[par]['lower_limit'] = 50.
                priors[par]['upper_limit'] = 90.
            elif par == 'b':
                priors[par]['lower_limit'] = 0.
                priors[par]['upper_limit'] = 1.
    
    return priors

def Spec( broad_fit_params, detector, u1, u2 ):
    """
    Puts the initial guesses for model parameters of a spectral light curve into a dictionary.
    Requires the wavelength range of the light curve for setting initial values for the LD coefficients.
    """
    
    pars = Parameters()
    
    if pr.FIT_ORBIT_SPEC:
        pars.add('aRs', value = broad_fit_params['aRs'] )
        if pr.IMPACT_PARAMETER:
            pars.add('b', value = broad_fit_params['b'], min=0., max=1. )
        else:
            pars.add('inc', value = broad_fit_params['inc'], max=90. )
    
    pars.add('R0', value = broad_fit_params['R0_' + detector])
    pars.add('R1', value = broad_fit_params['R1_' + detector], vary=pr.ROTATION_1)
    pars.add('R2', value = broad_fit_params['R2_' + detector], vary=pr.ROTATION_2)
    if pr.ROTATION_2_NEGATIVE:
        pars['R2'].max = 0.0
    
    if pr.FIT_TMID_SPEC:
        pars.add('Tmid', value = broad_fit_params['Tmid_' + detector], vary = True)
    elif pr.FIT_TMID_DETECTORS:
        pars.add('Tmid', value = broad_fit_params['Tmid_' + detector], vary = False)
    else:
        pars.add('Tmid', value = broad_fit_params['Tmid'], vary = False)
    
    if pr.BASELINE_SPEC == 'constant':
        pars.add('baseline', value = broad_fit_params['baseline_' + detector])
    elif pr.BASELINE_SPEC == 'linear':
        if pr.BASELINE_BROAD == 'linear':
            pars.add('c0', value = broad_fit_params['c0_' + detector])
            pars.add('c1', value = broad_fit_params['c1_' + detector])
        else:
            pars.add('c0', value = broad_fit_params['baseline_' + detector])
            pars.add('c1', value = 0.)
    elif pr.BASELINE_SPEC == 'quadratic':
        pars.add('c0', value = broad_fit_params['c0_' + detector])
        pars.add('c1', value = broad_fit_params['c1_' + detector])
        pars.add('c2', value = broad_fit_params['c2_' + detector])
    
    pars.add('p0', value = broad_fit_params['p0_' + detector])
    pars.add('p1', value = broad_fit_params['p1_' + detector])
    pars.add('p2', value = broad_fit_params['p2_' + detector])
    
    if pr.KIPPING_LD_PARAMETERS:
        q1, q2 = Utils.LD_coefficients_u2q(u1, u2)
        pars.add('q1', value = q1, min = 0., max = 1., vary = pr.FIT_1ST_LD_PARAMETER)
        pars.add('q2', value = q2, min = 0., max = 1., vary = pr.FIT_2ND_LD_PARAMETER)
    else:
        pars.add('u1', value = u1, vary = pr.FIT_1ST_LD_PARAMETER )
        pars.add('u2', value = u2, vary = pr.FIT_1ST_LD_PARAMETER )
    
    return pars

def all_lcs( lc, broad_fit_params ):
    """
    Puts the initial guesses for model parameters of a spectral light curve into a dictionary.
    Requires the wavelength range of the light curve for setting initial values for the LD coefficients.
    """
    
    channels = list(lc.keys())
    pars = Parameters()
    
    pars.add('aRs', value = broad_fit_params['aRs'] )
    if pr.IMPACT_PARAMETER:
        pars.add('b', value = broad_fit_params['b'], min=0., max=1. )
    else:
        pars.add('inc', value = broad_fit_params['inc'], max=90. )
    
    if not(pr.FIT_TMID_SPEC):
        pars.add('Tmid', value = pr.TRANSIT_MID_TIME)
    elif pr.FIT_TMID_DETECTORS:
        for nrs in ['NRS1', 'NRS2']:
            pars.add('Tmid_' + nrs, value = pr.TRANSIT_MID_TIME)
    if not(pr.FIT_ROTATION_2_SPEC):
        pars.add('R2', value=0)
    
    # Loop for parameters that are different between both detectors
    for ch in channels:
        detector = lc[ch]['detector']
        
        if pr.FIT_TMID_SPEC:
            pars.add('Tmid_' + ch, value = pr.TRANSIT_MID_TIME)
        
        pars.add('R0_' + ch, value = broad_fit_params['R0_' + detector] )
        pars.add('R1_' + ch, value = broad_fit_params['R1_' + detector], vary=pr.ROTATION_1 )
        if pr.FIT_ROTATION_2_SPEC:
            pars.add('R2_' + ch, value = broad_fit_params['R2_' + detector], vary=pr.ROTATION_2)#, max=0. )
        
        if pr.BASELINE_BROAD == 'constant':
            pars.add('baseline_' + ch, value=pr.BASELINE_C0 )
        elif pr.BASELINE_BROAD == 'linear':
            pars.add('c0_' + ch, value=broad_fit_params['c0_' + detector] )
            pars.add('c1_' + ch, value=broad_fit_params['c1_' + detector] )
        elif pr.BASELINE_BROAD == 'quadratic':
            pars.add('c0_' + ch, value=broad_fit_params['c0_' + detector] )
            pars.add('c1_' + ch, value=broad_fit_params['c1_' + detector] )
            pars.add('c2_' + ch, value=broad_fit_params['c2_' + detector] )
        
        pars.add('p0_' + ch, value = broad_fit_params['p0_' + detector])
        pars.add('p1_' + ch, value = broad_fit_params['p1_' + detector])
        pars.add('p2_' + ch, value = broad_fit_params['p2_' + detector])
        
        # Limb darkening coefficients
        if pr.KIPPING_LD_PARAMETERS:
            q1, q2 = Utils.LD_coefficients_u2q(lc[ch]['u1'], lc[ch]['u2'])
            pars.add('q1_' + ch, value=q1, min = 0., max = 1., vary = pr.FIT_1ST_LD_PARAMETER )
            pars.add('q2_' + ch, value=q2, min = 0., max = 1., vary = pr.FIT_2ND_LD_PARAMETER )
        else:
            pars.add('u1_' + ch, value=lc[ch]['u1'], vary = pr.FIT_1ST_LD_PARAMETER )
            pars.add('u2_' + ch, value=lc[ch]['u2'], vary = pr.FIT_2ND_LD_PARAMETER )
    
    return pars

def Spec_mcmc ( prelim_fit_params ):
    
    parameter_dict = prelim_fit_params.valuesdict()
    
    p0 = Parameters()
    for parameter in parameter_dict:
        if prelim_fit_params[parameter].vary:
            p0.add( parameter, value=prelim_fit_params[parameter].value, min=prelim_fit_params[parameter].min, max=prelim_fit_params[parameter].max)
            p0[parameter].stderr = prelim_fit_params[parameter].stderr
    
    if pr.ADD_SYSTEMATIC_UNCERTAINTY:
        p0.add('sigma_sys', value = 0., min = 0. )
    
    return p0