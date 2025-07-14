#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains various plotting routines important for the data analysis
"""

import os, pylab, seaborn
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mplc
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from corner import corner
import Utils, calculateBatmanModel
import parameters as pr

########## BACKEND ROUTINES CALLED BY OTHER ROUTINES ###########

def errorbars_colorscheme(x, y, yerr, colorvalue, colorscheme, colormap, lower_end, upper_end, errorbar_size, errorbar_alpha, z_order, ax=None ):
    """
    Workaround to have the same colors on the errorbars

    Parameters
    ----------
    colorvalues : float
        float that identifies all errorbars' color
    colorscheme : plt object
        colorscheme returned by plt.scatter
    
    """
    
    clb = plt.colorbar(colorscheme)
    norm = mplc.Normalize(vmin=lower_end, vmax=upper_end, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=colormap)
    c = mapper.to_rgba(colorvalue)
    
    try:
        ax.errorbar(x, y, yerr, color=c, elinewidth=errorbar_size, alpha=errorbar_alpha, ls='none', zorder=z_order)
    except Exception:
        plt.errorbar(x, y, yerr, color=c, elinewidth=errorbar_size, alpha=errorbar_alpha, ls='none', zorder=z_order)
    clb.remove()

def lineplot_colorscheme(x, y, colorvalue, colorscheme, colormap, lower_end, upper_end, line_alpha, ax=None ):
    """
    Workaround to have the same colors on the errorbars

    Parameters
    ----------
    colorvalues : float
        float that identifies all errorbars' color
    colorscheme : plt object
        colorscheme returned by plt.scatter
    
    """
    
    clb = plt.colorbar(colorscheme)
    norm = mplc.Normalize(vmin=lower_end, vmax=upper_end, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=colormap)
    c = mapper.to_rgba(colorvalue)
    
    try:
        ax.plot(x, y, color=c, alpha=line_alpha)
    except Exception:
        plt.plot(x, y, color=c, alpha=line_alpha )
    clb.remove()


def errorbars_colorscheme_individual(x, y, yerr, colorvalues, colorscheme, colormap, lower_end, upper_end, errorbar_size, errorbar_alpha, z_order, ax=None ):
    """
    Workaround to have the same colors on the errorbars

    Parameters
    ----------
    colorvalues : array
        array of floats that identifies each errorbar's color
    colorscheme : plt object
        colorscheme returned by plt.scatter

    """
    
    clb = plt.colorbar(colorscheme)
    norm = mplc.Normalize(vmin=lower_end, vmax=upper_end, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=colormap)
    color = np.array([(mapper.to_rgba(v)) for v in colorvalues])
    
    #loop over each data point to plot
    for x, y, e, color in zip(x, y, yerr, color):
        try:
            ax.errorbar(x, y, e, color=color, elinewidth=errorbar_size, alpha=errorbar_alpha, ls='none', zorder=z_order)
        except Exception:
            plt.errorbar(x, y, e, color=color, elinewidth=errorbar_size, alpha=errorbar_alpha, ls='none', zorder=z_order)
    clb.remove()


############# FRONTEND ROUTINES TO BE CALLED BY THE USER ################

def xy( x, y ):
    
    plt.clf()
    plt.plot(x,y)
    plt.show()
    plt.close()

def model_spectra( trans_spectra, opath = '' ):
    
    plt.clf()
    
    keys = list(trans_spectra.keys())
    for run in keys:
        plt.plot( trans_spectra[run]['wavelength']['value'], np.sqrt(trans_spectra[run]['transit_depth']['value']), label=run )
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('R$_p$/R$_s$')
    plt.xscale('log')
    plt.legend(fontsize='small')
    if len(opath) > 0:
        plt.savefig( opath + '.pdf', bbox_inches='tight' )
    plt.show()

def limb_darkening_coefficients( ld_coefficients, opath='', ld2 = None ):
    
    point_size = 1.
    
    errorbar_size = point_size/3.
    alpha_errorbars = 0.6
    
    wavelength = (ld_coefficients['lower_wavelength'] + ld_coefficients['upper_wavelength']) / 2.
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    
    fig, ax = plt.subplots( 2, 1, sharex=True )
    
    i = 0
    for ldc in ['u1', 'u2']:        
        ax[i].scatter( wavelength, ld_coefficients[ldc], c='blue', s=point_size)
        ax[i].errorbar( wavelength, ld_coefficients[ldc], yerr=ld_coefficients[ldc + '_err'], elinewidth=errorbar_size, alpha=alpha_errorbars, c='blue', ls='none')
        # ax[i].plot( wavelength, ld_coefficients[ldc], color='blue' )
        # ax[i].plot( wavelength, ld_coefficients[ldc] + ld_coefficients[ldc + '_err'], color='blue' )
        # ax[i].plot( wavelength, ld_coefficients[ldc] - ld_coefficients[ldc + '_err'], color='blue' )
        if type(ld2)==dict:
            ax[i].scatter( wavelength, ld2[ldc], c='red', s=point_size)
        ax[i].set_ylabel(ldc)
        i += 1
    
    ax[1].set_xlabel('Wavelength [microns]')
    
    if len(opath) > 0:
        fig.savefig( opath + '.pdf', bbox_inches='tight' )
        print( '\nSaved:\n{0}'.format( opath ) )
    if pr.SHOW_PLOTS:
        plt.show()
    

def lightcurve( lc, opath = '' ):
    
    plt.clf()
    fig, ax = plt.subplots( 1, 1, figsize=[6,4], sharex=True )
    ax.scatter( lc['bjd_tdb'], lc['lcVals'], c='blue', s=10. )
    ax.errorbar( lc['bjd_tdb'], lc['lcVals'], yerr=lc['lcUncs'], ecolor='blue', ls='none')
    ax.set_xlabel('t')
    ax.set_ylabel('Relative flux')
    fig.tight_layout()
    if len(opath) > 0:
        fig.savefig( opath + '.png', bbox_inches='tight', dpi=300 )
    else:
        plt.show()
    plt.close()

def lightcurves( lc, opath = '' ):
    
    plt.clf()
    fig, ax = plt.subplots( len(lc), 1, figsize=[6,6], sharex=True )
    
    i = 0
    for key in lc.keys():
        ax[i].scatter( lc[key]['t'], lc[key]['lcVals'], c='blue', s=10. )
        ax[i].errorbar( lc[key]['t'], lc[key]['lcVals'], yerr=lc[key]['lcUncs'], ecolor='blue', ls='none')
        ax[i].set_ylabel(key)
        # ax[i].scatter( lc[key]['t'][:150], lc[key]['lcVals'][:150], c='red', s=10. )
        ax[i].grid()
        i += 1
    
    ax[-1].set_xlabel('t')
    fig.tight_layout()
    if len(opath) > 0:
        fig.savefig( opath + '.png', bbox_inches='tight', dpi=300 )
    else:
        plt.show()
    plt.close()

def lightcurves_2versions( lc, lc2, opath = '' ):
    pointsize = 1.
    plt.clf()
    fig, ax = plt.subplots( len(lc), 2, figsize=[6,4], sharex='col', sharey='row' )
    
    i = 0
    for key in lc.keys():
        ax[i,0].scatter( lc[key]['t'], lc[key]['lcVals'], c='blue', s=pointsize )
        ax[i,0].errorbar( lc[key]['t'], lc[key]['lcVals'], yerr=lc[key]['lcUncs'], ecolor='blue', ls='none')
        ax[i,0].set_xlim( np.min(lc[key]['t']), np.max(lc[key]['t']) )
        
        ax[i,1].scatter( lc2[key]['t'], lc2[key]['lcVals'], c='blue', s=pointsize )
        ax[i,1].errorbar( lc2[key]['t'], lc2[key]['lcVals'], yerr=lc2[key]['lcUncs'], ecolor='blue', ls='none')
        ax[i,1].set_xlim( np.min(lc2[key]['t']), np.max(lc2[key]['t']) )
        
        ax[i,0].set_ylabel(key)
        i += 1
    ax[0,0].set_title('Firefly')
    ax[0,1].set_title('Eureka!')
    ax[-1,0].set_xlabel('t')
    ax[-1,1].set_xlabel('t')
    fig.tight_layout()
    if len(opath) > 0:
        fig.savefig( opath + '.pdf', bbox_inches='tight', dpi=300 )
    else:
        plt.show()
    plt.close()

def broad_light_curves_phases( z, best_params, emcee_result = None, opath = '' ):
    """
    Plots light curves for both the NRS1 and the NRS2 detectors. If you pass it an emcee result dictionary, it will also plot a collection
    of models for both detectors.
    """
    
    point_size = 6.
    errorbar_size = 0.6
    shade_alpha = 0.2
    shade_color = 'blue'
    line_width = 1.2
    
    detectors = list(z.keys())
    data_phases = {}
    normalized_flux = {}
    for nrs in detectors:
        data_phases[nrs] = (z[nrs]['bjd_tdb'] - pr.TIME_0 - best_params['Tmid_' + nrs].value) / pr.ORBITAL_PERIOD['median']
    
    # First, initiate batman for computing models and calculate the best fit model
    batpar = {}
    batmodel = {}
    best_model = {}
    
    for nrs in z.keys():
        batpar[nrs], batmodel[nrs] = calculateBatmanModel.create_Batman_object( z[nrs]['bjd_tdb'] )
        
        best_model[nrs] = calculateBatmanModel.Broad( best_params, z[nrs], batpar[nrs], batmodel[nrs], nrs )
    
    # Plot the best-fit light curve model against the data for both detectors:
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    # Plot the best-fit light curve model against the data for both detectors:
    # fig = plt.figure( figsize=[10,6] )      # figsize=[10,6]
    fig, ax = plt.subplots( 2, 1, sharex=True, figsize=[8,4] )
    
    # Background: Data
    i = 0
    for nrs in z.keys():
        # Plot data and best fit model
        label = nrs + ', {:.2f} - {:.2f} microns'.format(z[nrs]['wavEdgesMicr'][0], z[nrs]['wavEdgesMicr'][1])
        ax[i].scatter( data_phases[nrs], z[nrs]['lcVals'], c='white', edgecolor='black', s=point_size, label=label, zorder=3 )
        ax[i].errorbar( data_phases[nrs], z[nrs]['lcVals'], yerr=z[nrs]['lcUncs'], ecolor='black', ls='none', zorder=2)
        i += 1
    
    try:
        # Plot random draws from the fit params
        random_indices = np.random.randint( len( emcee_result['flat_chain'] ), size=100)
        i = 0
        for nrs in z.keys():
            for index in random_indices:
                parameters = emcee_result['flat_chain'][index]
                
                pars_dict = Utils.arrays2dict( parameters, emcee_result['parameters'] )
                pars_dict = Utils.merge_dictionaries(pars_dict, best_params)
                
                random_model = calculateBatmanModel.Broad( pars_dict, z[nrs], batpar[nrs], batmodel[nrs], nrs )
                ax[i].plot( data_phases[nrs], random_model, 'yellow', alpha=0.1, zorder=3.5)
            i += 1
    except TypeError:
        pass
    
    i = 0
    for nrs in z.keys():
        # Plot best fit model
        ax[i].plot( data_phases[nrs], best_model[nrs], 'orange', label='Batman', zorder=4 )
        ax[i].set_ylabel('Relative Flux')
        ax[i].legend()
        i += 1
    
    ax[0].set_xlim( -0.04, 0.04)
    ax[0].set_ylim( 0.9813, 0.984 )
    ax[1].set_ylim( 0.98, 0.982 )
    ax[1].set_xlabel( 'Phase' )
    fig.tight_layout()
    
    if len(opath) > 0:
        fig.savefig( opath + '.png', bbox_inches='tight', dpi=300 )
        print( '\nSaved:\n{0}'.format( opath + '.png' ) )
    else:
        plt.show()
    
    plt.close()

def broad_light_curve_corrections( z, fp, fs, ofile ):
    """
    Plots light curves for both the NRS1 and the NRS2 detectors. If you pass it an emcee result dictionary, it will also plot a collection
    of models for both detectors.
    """
    
    detectors = list(z.keys())
    
    # Plot the best-fit light curve model against the data for both detectors:
    plt.clf()
    fig, ax = plt.subplots( 1, 2, figsize=[10,6], sharex=True, sharey=True )
    
    i = 0
    for nrs in detectors:
        # Plot data and fp + B_plot
        ax[i].scatter( z[nrs]['bjd_tdb'], z[nrs]['lcVals'], c='black', s=10. )
        ax[i].errorbar( z[nrs]['bjd_tdb'], z[nrs]['lcVals'], yerr=z[nrs]['lcUncs'], ecolor='black', ls='none')
        ax[i].plot( z[nrs]['bjd_tdb'], fp[nrs] + fs[nrs], color='blue')
        ax[i].set_title( nrs + ', {:.2f} - {:.2f} microns'.format(z[nrs]['wavEdgesMicr'][0], z[nrs]['wavEdgesMicr'][1]) )
        ax[i].set_xlabel('BJD_TDB')
        i += 1
    
    ax[0].set_ylabel('Relative Flux')
    fig.tight_layout()
    
    if len(ofile) > 0:
        opath = ofile + '.pdf'
        fig.savefig( opath, bbox_inches='tight' )
        print( '\nSaved:\n{0}'.format( opath ) )
    if pr.SHOW_PLOTS:
        plt.show()
    
    plt.close()

def white_light_curves( data, best_params, time_limits='None', residual_limits=[-600., 600.], opath = '', emcee_result = None ):
    """
    Plots light curves for both the NRS1 and the NRS2 detectors. Adds panels with the residuals.
    """
    
    point_size = 6.
    errorbar_size = 0.6
    shade_alpha = 0.2
    shade_color = 'blue'
    line_width = 1.2
    
    detectors = list(data.keys())
    
    # Calculate times of transit ingress and egress
    
    T14 = {}
    T23 = {}
    try:
        for nrs in detectors:
            # Using the equations from Winn et al. (2010)
            T14[nrs] = Utils.T14_Winn(pr.ORBITAL_PERIOD['median'], best_params['aRs'], best_params['RpRs_' + nrs], best_params['inc'])
            T23[nrs] = Utils.T23_Winn(pr.ORBITAL_PERIOD['median'], best_params['aRs'], best_params['RpRs_' + nrs], best_params['inc'])
    except KeyError:
        for nrs in detectors:
            # Using the equations from Winn et al. (2010)
            if pr.IMPACT_PARAMETER:
                inc = Utils.impact_parameter2inc( best_params['b'], best_params['aRs'] )
            else:
                inc = best_params['inc']
            T14[nrs] = Utils.T14_Winn(pr.ORBITAL_PERIOD['median'], best_params['aRs'], best_params['R0_' + nrs], inc)
            T23[nrs] = Utils.T23_Winn(pr.ORBITAL_PERIOD['median'], best_params['aRs'], best_params['R0_' + nrs], inc)
            print(nrs, T23[nrs]*24.)
    
    T1 = {}
    T2 = {}
    T3 = {}
    T4 = {}
    
    for nrs in detectors:
        if pr.FIT_TMID_SPEC or pr.FIT_TMID_DETECTORS:
            T1[nrs] = best_params['Tmid_' + nrs] - T14[nrs]/2.
            T2[nrs] = best_params['Tmid_' + nrs] - T23[nrs]/2.
            T3[nrs] = best_params['Tmid_' + nrs] + T23[nrs]/2.
            T4[nrs] = best_params['Tmid_' + nrs] + T14[nrs]/2.
        else:
            T1[nrs] = best_params['Tmid'] - T14[nrs]/2.
            T2[nrs] = best_params['Tmid'] - T23[nrs]/2.
            T3[nrs] = best_params['Tmid'] + T23[nrs]/2.
            T4[nrs] = best_params['Tmid'] + T14[nrs]/2.
    
    # First, initialize a batman model for every data point to change the transit radius as a function of time
    batpar = {}
    batmodel = {}
    best_model = {}
    for nrs in detectors:
        batpar[nrs], batmodel[nrs] = calculateBatmanModel.create_Batman_arrays( data[nrs]['bjd_tdb'] )
        best_model[nrs] = calculateBatmanModel.Broad( best_params, data[nrs], batpar[nrs], batmodel[nrs], nrs )
        
    # Plot the best-fit light curve model against the data for both detectors:
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    # Plot the best-fit light curve model against the data for both detectors:
    fig, ax = plt.subplots( 2, 2, sharex=True, sharey='row', figsize=[9.1,5] )
    
    # ax = [ax1,ax2,ax3,ax4]
    i = 0
    j = 0
    
    # Data
    for nrs in detectors:
        ax[i,j].scatter( data[nrs]['bjd_tdb'] - pr.TIME_0, data[nrs]['lcVals'], c='white', edgecolor='black', s=point_size, label='Data', zorder=2.0E03 )
        ax[i,j].errorbar( data[nrs]['bjd_tdb'] - pr.TIME_0, data[nrs]['lcVals'], yerr=data[nrs]['lcUncs'], ecolor='black', elinewidth=errorbar_size, ls='none', zorder=1.0E03 )
        ax[i,j].axvspan(T1[nrs], T2[nrs], alpha=shade_alpha, color=shade_color)
        ax[i,j].axvspan(T3[nrs], T4[nrs], alpha=shade_alpha, color=shade_color)
        
        j += 1
    
    # Models
    try:
        # Plot random draws from the fit params
        random_indices = np.random.randint( len( emcee_result['flat_chain'] ), size=100)
        j = 0
        for nrs in data.keys():
            for index in random_indices:
                parameters = emcee_result['flat_chain'][index]
                pars_dict = Utils.arrays2dict( parameters, emcee_result['parameters'] )
                pars_dict = Utils.merge_dictionaries(pars_dict, best_params)
                random_model = calculateBatmanModel.Broad( pars_dict, data[nrs], batpar[nrs], batmodel[nrs], nrs )
                ax[i,j].plot( data[nrs]['bjd_tdb'] - pr.TIME_0, random_model, 'yellow', linewidth=line_width, alpha=0.1, zorder=1.0E04)
            j += 1
    
    except TypeError:
        pass
    
    j = 0
    for nrs in data.keys():
        ax[i,j].plot( data[nrs]['bjd_tdb'] - pr.TIME_0, best_model[nrs], 'orange', linewidth=line_width, label='ML model', zorder=2.0E04 )
        # if pr.QUADRATIC_BASELINE_BROAD:
        #     ax[i,j].plot( data[nrs]['bjd_tdb'] - pr.TIME_0, np.zeros(len(data[nrs]['bjd_tdb'])) + best_params['c0_' + nrs] - best_params['c1_' + nrs]**2/(4.*best_params['c2_' + nrs]), 'magenta', label='Baseline' )
        #     print(nrs, best_params['c0_' + nrs] - best_params['c1_' + nrs]**2/(4.*best_params['c2_' + nrs]))
        try:
            ax[i,j].set_title( nrs + ', {:.2f} - {:.2f} microns'.format(data[nrs]['wavEdgesMicr'][0], data[nrs]['wavEdgesMicr'][1]) )
        except KeyError:
            ax[i,j].set_title( nrs )
        j += 1
    
    i = 1
    j = 0
    # Residuals
    unit = 1.0E06       # ppm
    for nrs in data.keys():
        residuals = (data[nrs]['lcVals'] - best_model[nrs])/data[nrs]['lcVals'] * unit#/data[nrs]['lcUncs']
        
        # Perform binning
        if time_limits == 'None':
            bin_edges = np.linspace( min(data[nrs]['bjd_tdb']), max(data[nrs]['bjd_tdb']), 100 )
        else:
            bin_edges = np.linspace( min(data[nrs]['bjd_tdb']), max(data[nrs]['bjd_tdb']), 200 )
        residuals_binned, applied_bin_edges, indices = binned_statistic( data[nrs]['bjd_tdb'], residuals, bins=bin_edges )
        bin_middles = Utils.bin_edges_into_middles( bin_edges )
        
        ax[i,j].scatter( data[nrs]['bjd_tdb'] - pr.TIME_0, residuals, c='white', edgecolor='black', s=point_size, zorder=2.0E03 )
        ax[i,j].errorbar( data[nrs]['bjd_tdb'] - pr.TIME_0, residuals, yerr=data[nrs]['lcUncs']/data[nrs]['lcVals']*unit, ecolor='black', elinewidth=errorbar_size, ls='none', zorder=1.0E03 )
        ax[i,j].plot( bin_middles - pr.TIME_0, residuals_binned, linewidth=line_width, color='red', zorder=1.0E04 )
        ax[i,j].axvspan(T1[nrs], T2[nrs], alpha=shade_alpha, color=shade_color)
        ax[i,j].axvspan(T3[nrs], T4[nrs], alpha=shade_alpha, color=shade_color)
            
        ax[i,j].set_xlabel('Time [BJD_TDB - {:.0f}]'.format(pr.TIME_0))
        
        j += 1
    
    # Cosmetics
    ax[0,0].set_ylabel('Relative Flux')
    ax[0,0].legend( loc='lower left' )
    ax[1,0].set_ylabel('Relative residuals [ppm]')
    try:
        ax[0,0].set_xlim( time_limits )
    except ValueError:
        ax[0,0].set_xlim( min(bin_middles) - pr.TIME_0, max(bin_middles) - pr.TIME_0)
    ax[1,0].set_ylim( residual_limits )
    fig.tight_layout()
    
    if pr.SAVE_PLOTS and len(opath) > 0:
        fig.savefig( opath + '.pdf', dpi=300, bbox_inches='tight' )
        print( '\nSaved:',opath + '.pdf' )
    if pr.SHOW_PLOTS:
        plt.show()
    
    plt.close()

def broad_light_curves( data, model, params, opath = '', emcee_result = None, prelim_fit_params = None ):
    """
    Plots light curves for both the NRS1 and the NRS2 detectors. Adds panels with the residuals.
    """
    
    point_size = 6.
    errorbar_size = 0.6
    shade_alpha = 0.2
    shade_color = 'blue'
    line_width = 1.2
    
    detectors = list(data.keys())
    
    # Count the number of varied parameters for the reduced chisquare calculation
    n_params = Utils.count_varied_parameters_lmfit( params )
    data_alldetectors = np.array([])
    uncertainties_alldetectors = np.array([])
    model_alldetectors = np.array([])
    
    print('Detector, Reduced Chisquares, RMS [ppm]')
    for nrs in detectors:
        # print(nrs, compare_residuals_to_poisson( z[nrs]['lcVals'], z[nrs]['lcUncs'], model) )
        chisquare = Utils.reduced_chisquare( data[nrs]['lcVals'], data[nrs]['lcUncs'], model[nrs], n_params)
        root_mean_square = Utils.rms(data[nrs]['lcVals'], model[nrs]) / 1.0E-06
        print(nrs, chisquare, root_mean_square)
        
        model_alldetectors = np.append(model_alldetectors, model[nrs])
        data_alldetectors = np.append(data_alldetectors, data[nrs]['lcVals'])
        uncertainties_alldetectors = np.append(uncertainties_alldetectors, data[nrs]['lcUncs'])
    print('Reduced chi squared:', Utils.reduced_chisquare(data_alldetectors, uncertainties_alldetectors, model_alldetectors, n_params) )
    
    # Calculate times of transit ingress and egress
    
    T14 = {}
    T23 = {}
    for nrs in detectors:
        # Using the equations from Winn et al. (2010)
        T14[nrs] = Utils.T14_Winn(pr.ORBITAL_PERIOD['median'], params['aRs'], params['R0_' + nrs], params['inc'])
        T23[nrs] = Utils.T23_Winn(pr.ORBITAL_PERIOD['median'], params['aRs'], params['R0_' + nrs], params['inc'])
    
    T1 = {}
    T2 = {}
    T3 = {}
    T4 = {}
    
    for nrs in detectors:
        if pr.FIT_TMID_SPEC:
            T1[nrs] = params['Tmid_' + nrs] - T14[nrs]/2.
            T2[nrs] = params['Tmid_' + nrs] - T23[nrs]/2.
            T3[nrs] = params['Tmid_' + nrs] + T23[nrs]/2.
            T4[nrs] = params['Tmid_' + nrs] + T14[nrs]/2.
        else:
            T1[nrs] = params['Tmid'] - T14[nrs]/2.
            T2[nrs] = params['Tmid'] - T23[nrs]/2.
            T3[nrs] = params['Tmid'] + T23[nrs]/2.
            T4[nrs] = params['Tmid'] + T14[nrs]/2.
    
    # Plot the best-fit light curve model against the data for both detectors:
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    # Plot the best-fit light curve model against the data for both detectors:
    fig, ax = plt.subplots( 2, 2, sharex=True, sharey='row', figsize=[9.1,5] )
    
    # ax = [ax1,ax2,ax3,ax4]
    i = 0
    j = 0
    
    # Data
    for nrs in detectors:
        ax[i,j].scatter( data[nrs]['bjd_tdb'] - pr.TIME_0, data[nrs]['lcVals'], c='white', edgecolor='black', s=point_size, label='Data', zorder=2.0E03 )
        ax[i,j].errorbar( data[nrs]['bjd_tdb'] - pr.TIME_0, data[nrs]['lcVals'], yerr=data[nrs]['lcUncs'], ecolor='black', elinewidth=errorbar_size, ls='none', zorder=1.0E03 )
        ax[i,j].axvspan(T1[nrs], T2[nrs], alpha=shade_alpha, color=shade_color)
        ax[i,j].axvspan(T3[nrs], T4[nrs], alpha=shade_alpha, color=shade_color)
        
        j += 1
    
    # # Models
    # try:
    #     # Plot random draws from the fit params
    #     random_indices = np.random.randint( len( emcee_result['flat_chain'] ), size=100)
    #     j = 0
    #     for nrs in data.keys():
    #         for index in random_indices:
    #             parameters = emcee_result['flat_chain'][index]
    #             pars_dict = Utils.arrays2dict( parameters, emcee_result['parameters'] )
    #             pars_dict = Utils.merge_dictionaries(pars_dict, best_params)
    #             random_model = calculateBatmanModel.Broad( pars_dict, z[nrs], batpar[nrs], batmodel[nrs], nrs )
    #             ax[i,j].plot( z[nrs]['bjd_tdb'] - pr.TIME_0, random_model, 'yellow', linewidth=line_width, alpha=0.1, zorder=1.0E04)
    #         j += 1
    
    # except TypeError:
    #     pass
    
    j = 0
    for nrs in data.keys():
        ax[i,j].plot( data[nrs]['bjd_tdb'] - pr.TIME_0, model[nrs], 'yellow', linewidth=line_width, label='ML model', zorder=2.0E04 )
        # if pr.QUADRATIC_BASELINE_BROAD:
        #     ax[i,j].plot( z[nrs]['bjd_tdb'] - pr.TIME_0, np.zeros(len(z[nrs]['bjd_tdb'])) + best_params['c0_' + nrs] - best_params['c1_' + nrs]**2/(4.*best_params['c2_' + nrs]), 'magenta', label='Baseline' )
        #     print(nrs, best_params['c0_' + nrs] - best_params['c1_' + nrs]**2/(4.*best_params['c2_' + nrs]))
        
        ax[i,j].set_title( nrs + ', {:.2f} - {:.2f} microns'.format(data[nrs]['wavEdgesMicr'][0], data[nrs]['wavEdgesMicr'][1]) )
        j += 1
    
    i = 1
    j = 0
    # Residuals
    unit = 1.0E06       # ppm
    for nrs in data.keys():
        residuals = (data[nrs]['lcVals'] - model[nrs])/data[nrs]['lcVals'] * unit#/z[nrs]['lcUncs']
        
        # Perform binning
        bin_edges = np.linspace( min(data[nrs]['bjd_tdb']), max(data[nrs]['bjd_tdb']), 100 )
        residuals_binned, applied_bin_edges, indices = binned_statistic( data[nrs]['bjd_tdb'], residuals, bins=bin_edges )
        bin_middles = Utils.bin_edges_into_middles( bin_edges )
        
        ax[i,j].scatter( data[nrs]['bjd_tdb'] - pr.TIME_0, residuals, c='white', edgecolor='black', s=point_size, zorder=2.0E03 )
        ax[i,j].errorbar( data[nrs]['bjd_tdb'] - pr.TIME_0, residuals, yerr=data[nrs]['lcUncs']/data[nrs]['lcVals']*unit, ecolor='black', elinewidth=errorbar_size, ls='none', zorder=1.0E03 )
        ax[i,j].plot( bin_middles - pr.TIME_0, residuals_binned, linewidth=line_width, color='red', zorder=1.0E04 )
        ax[i,j].axvspan(T1[nrs], T2[nrs], alpha=shade_alpha, color=shade_color)
        ax[i,j].axvspan(T3[nrs], T4[nrs], alpha=shade_alpha, color=shade_color)
        
        ax[i,j].set_xlabel('Time [BJD_TDB - {:.0f}]'.format(pr.TIME_0))
        
        j += 1
    
    # Cosmetics
    ax[0,0].set_ylabel('Relative Flux')
    ax[0,0].legend( loc='lower left' )
    ax[1,0].set_ylabel('Relative residuals [ppm]')
    ax[0,0].set_xlim( min(bin_middles) - pr.TIME_0, max(bin_middles) - pr.TIME_0)
    ax[1,0].set_ylim(-600.,600.)
    fig.tight_layout()
    
    if pr.SAVE_PLOTS and len(opath) > 0:
        fig.savefig( opath + '.pdf', dpi=300, bbox_inches='tight' )
        print( '\nSaved:',opath + '.pdf' )
    if pr.SHOW_PLOTS:
        plt.show()
    
    plt.close()


def broad_light_curves_emission( z, best_params, filename, emcee_result = None, prelim_fit_params = None ):
    """
    Plots light curves for both the NRS1 and the NRS2 detectors. If you pass it an emcee result dictionary, it will also plot a collection
    of models for both detectors.
    """
    
    # First, initiate batman for computing models and calculate the best fit model
    batpar = {}
    batmodel = {}
    best_model = {}
    no_emission = {}
    
    for nrs in z.keys():
        batpar[nrs], batmodel[nrs] = calculateBatmanModel.create_Batman_object( z[nrs]['bjd_tdb'] )

        if pr.VARY_LD_PARAMETERS:
            # Setting the limb darkening parameters before calculating the batman model is necessary when LD is not fitted for
            if pr.KIPPING_LD_PARAMETERS:
                u1, u2 = Utils.LD_coefficients_q2u( best_params['q1_' + nrs], best_params['q2_' + nrs] )
                batpar[nrs].u = [ u1, u2 ]
            else:
                batpar[nrs].u = [best_params['u1_' + nrs], best_params['u2_' + nrs]]
        else:
            # Setting the limb darkening parameters before calculating the batman model is necessary when LD is not fitted for
            if pr.KIPPING_LD_PARAMETERS:
                u1, u2 = Utils.LD_coefficients_q2u( prelim_fit_params['q1_' + nrs], prelim_fit_params['q2_' + nrs] )
                batpar[nrs].u = [ u1, u2 ]
            else:
                batpar[nrs].u = [prelim_fit_params['u1_' + nrs], prelim_fit_params['u2_' + nrs]]
        
        best_model[nrs] = calculateBatmanModel.Broad( best_params, z[nrs], batpar[nrs], batmodel[nrs], nrs )
        no_emission[nrs] = calculateBatmanModel.Broad( best_params, z[nrs], batpar[nrs], batmodel[nrs], nrs, switch=False )
        
    # Plot the best-fit light curve model against the data for both detectors:
    plt.clf()
    fig, ax = plt.subplots( 1, 2, figsize=[10,6], sharex=True, sharey=True )
    
    # Background: Data
    i = 0
    for nrs in z.keys():
        # Plot data and best fit model
        ax[i].scatter( z[nrs]['bjd_tdb'], z[nrs]['lcVals'], c='black', s=10. )
        ax[i].errorbar( z[nrs]['bjd_tdb'], z[nrs]['lcVals'], yerr=z[nrs]['lcUncs'], ecolor='black', ls='none')
        i += 1
    
    try:
        # Plot random draws from the fit params
        random_indices = np.random.randint( len( emcee_result['flat_chain'] ), size=100)
        i = 0
        for nrs in z.keys():
            for index in random_indices:
                parameters = emcee_result['flat_chain'][index]
                pars_dict = Utils.arrays2dict( parameters, emcee_result['parameters'] )
                random_model = calculateBatmanModel.Broad( pars_dict, z[nrs], batpar[nrs], batmodel[nrs], nrs )
                ax[i].plot( z[nrs]['bjd_tdb'], random_model, 'cyan', alpha=0.1)
            i += 1
    except TypeError:
        pass
    
    i = 0
    for nrs in z.keys():
        # Plot best fit model
        ax[i].plot( z[nrs]['bjd_tdb'], best_model[nrs], 'blue' )
        ax[i].plot( z[nrs]['bjd_tdb'], no_emission[nrs], 'magenta' )
        ax[i].set_title( nrs + ', {:.2f} - {:.2f} microns'.format(z[nrs]['wavEdgesMicr'][0], z[nrs]['wavEdgesMicr'][1]) )
        ax[i].set_xlabel('BJD_TDB')
        i += 1
    
    ax[0].set_ylabel('Relative Flux')
    fig.tight_layout()
    
    if len(filename) > 0:
        opath = filename + '.pdf'
        fig.savefig( opath, bbox_inches='tight' )
        print( '\nSaved:\n{0}'.format( opath ) )
    plt.show()
    
    plt.close()

def broad_light_curves_partial_phasecurve( z, best_params, emcee_result = None, ofile = '' ):
    """
    Plots light curves for both the NRS1 and the NRS2 detectors. If you pass it an emcee result dictionary, it will also plot a collection
    of models for both detectors.
    """
    
    # First, initiate batman for computing models and calculate the best fit model
    batpar = {}
    batmodel = {}
    best_model = {}
    no_emission = {}
    
    for nrs in z.keys():
        batpar[nrs], batmodel[nrs] = calculateBatmanModel.create_Batman_object( z[nrs]['bjd_tdb'] )
        
        best_model[nrs] = calculateBatmanModel.Broad( best_params, z[nrs], batpar[nrs], batmodel[nrs], nrs )
    
    
    # Set A=0 to show the light curve model without the planetary emission
    
    for nrs in z.keys():
        params_no_emission = Utils.switch_off_emission(best_params)
        
        no_emission[nrs] = calculateBatmanModel.Broad( params_no_emission, z[nrs], batpar[nrs], batmodel[nrs], nrs )
        
    # Plot the best-fit light curve model against the data for both detectors:
    plt.clf()
    fig, ax = plt.subplots( 1, 2, figsize=[10,6], sharex=True, sharey=True )
    
    # Background: Data
    i = 0
    for nrs in z.keys():
        # Plot data and best fit model
        ax[i].scatter( z[nrs]['bjd_tdb'], z[nrs]['lcVals'], c='black', s=10. )
        ax[i].errorbar( z[nrs]['bjd_tdb'], z[nrs]['lcVals'], yerr=z[nrs]['lcUncs'], ecolor='black', ls='none')
        i += 1
    
    try:
        # Plot random draws from the fit params
        random_indices = np.random.randint( len( emcee_result['flat_chain'] ), size=100)
        for i, nrs in enumerate(list(z.keys())):
            for index in random_indices:
                parameters = emcee_result['flat_chain'][index]
                
                pars_dict = Utils.arrays2dict( parameters, emcee_result['parameters'] )
                pars_dict = Utils.merge_dictionaries(pars_dict, best_params)
                
                random_model = calculateBatmanModel.Broad( pars_dict, z[nrs], batpar[nrs], batmodel[nrs], nrs )
                ax[i].plot( z[nrs]['bjd_tdb'], random_model, 'magenta', alpha=0.1)
                
                pars_no_emission = Utils.switch_off_emission(pars_dict)
                random_no_emission = calculateBatmanModel.Broad( pars_no_emission, z[nrs], batpar[nrs], batmodel[nrs], nrs )
                ax[i].plot( z[nrs]['bjd_tdb'], random_no_emission, 'cyan', alpha=0.1)

    except TypeError:
        pass
    
    i = 0
    for nrs in z.keys():
        # Plot best fit model
        ax[i].plot( z[nrs]['bjd_tdb'], best_model[nrs], 'red', label='Full model' )
        ax[i].plot( z[nrs]['bjd_tdb'], no_emission[nrs], 'blue', label='Baseline' )
        ax[i].set_title( nrs + ', {:.2f} - {:.2f} microns'.format(z[nrs]['wavEdgesMicr'][0], z[nrs]['wavEdgesMicr'][1]) )
        ax[i].set_xlabel('BJD_TDB')
        ax[i].set_xlim(np.min(z[nrs]['bjd_tdb']), np.max(z[nrs]['bjd_tdb']) )
        i += 1
    
    ax[0].set_ylabel('Relative Flux')
    ax[0].legend()
    fig.tight_layout()
    
    if len(ofile) > 0:
        opath = ofile + '.png'
        fig.savefig( opath, bbox_inches='tight', dpi=300 )
        print( '\nSaved:\n{0}'.format( opath ) )
    else:
        plt.show()
    
    plt.close()

def broad_phasecurve_only( z, best_params, ofile = '', emcee_result = None ):
    """
    Plots light curves for both the NRS1 and the NRS2 detectors. If you pass it an emcee result dictionary, it will also plot a collection
    of models for both detectors.
    """
    
    # First, initiate batman for computing models and calculate the best fit model
    batpar = {}
    batmodel = {}
    model = {}
    
    for nrs in z.keys():
        batpar[nrs], batmodel[nrs] = calculateBatmanModel.create_Batman_object( z[nrs]['bjd_tdb'] )
        
        if pr.FIT_PHASE_SPEC:
            phi = best_params['phi_' + nrs]
        else:
            phi = best_params['phi']
            
        if pr.FIT_AMPLITUDE_SPEC:
            A = best_params['A_' + nrs]
        else:
            A = best_params['A']
        
        if pr.BASELINE_BROAD == 'linear':
            model[nrs] = (A * (best_params['BoverA_' + nrs] + np.cos(2.*np.pi*z[nrs]['t']/pr.ORBITAL_PERIOD['median'] + phi))) * (best_params['c0_' + nrs] + best_params['c1_' + nrs]*z[nrs]['t'])
        elif pr.BASELINE_BROAD == 'constant':
            model[nrs] = (A * (best_params['BoverA_' + nrs] + np.cos(2.*np.pi*z[nrs]['t']/pr.ORBITAL_PERIOD['median'] + phi))) * (best_params['baseline_' + nrs])
        
        # Unit
        model[nrs] *= 1.0E6
        
        print(nrs, np.min(model[nrs]))
        
    # Plot the best-fit light curve model against the data for both detectors:
    plt.clf()
    fig, ax = plt.subplots( 1, 2, figsize=[10,6], sharex=True, sharey=True )
    
    i = 0
    for nrs in z.keys():
        # Plot best fit model
        ax[i].plot( z[nrs]['bjd_tdb'], model[nrs], 'magenta' )
        ax[i].set_title( nrs + ', {:.2f} - {:.2f} microns'.format(z[nrs]['wavEdgesMicr'][0], z[nrs]['wavEdgesMicr'][1]) )
        ax[i].set_xlabel('BJD_TDB')
        i += 1
    
    ax[0].set_ylabel('Relative Flux [ppm]')
    fig.tight_layout()
    
    if len(ofile) > 0:
        opath = ofile + '.pdf'
        fig.savefig( opath, bbox_inches='tight' )
        print( '\nSaved:\n{0}'.format( opath ) )
    else:
        plt.show()
    
    plt.close()


def n_lightcurves_m_models_zoom( data, best_params, emcee_result = None, lc_limits=[0.983, 1.0005], zoom_limits=[0.9837, 0.985], residual_limits=[-500., 500.], scale_errorbars = False, opath = '' ):
    """
    Plots light curves for both the NRS1 and the NRS2 detectors. Adds panels with the residuals.
    """
    
    point_size = 5.
    errorbar_size = 0.4
    model_colors = ['blue', 'darkorange']
    mcmc_colors = ['aqua', 'yellow']
    shade_alpha = 0.2
    shade_color = 'gray'
    line_width = 1.
    model_labels = {'translational' : '$\mathcal{M}_{tra}$', 'linear' : '$\mathcal{M}_{rot}$', 'quadratic' : '$\mathcal{M}_{rot}$'}
    reductions = list(data.keys())
    
    # Calculate contact points
    T1 = {}
    T2 = {}
    T3 = {}
    T4 = {}
    for reduction in reductions:
        models = list(best_params[reduction].keys())
        in_egress_model = models[0]
        
        if pr.IMPACT_PARAMETER:
            inc = Utils.impact_parameter2inc( best_params[reduction][in_egress_model]['b'], best_params[reduction][in_egress_model]['aRs'])
        else:
            inc = best_params[reduction][in_egress_model]['inc']
        
        # Using the equations from Winn et al. (2010)
        T14 = Utils.T14_Winn(pr.ORBITAL_PERIOD['median'], best_params[reduction][in_egress_model]['aRs'], best_params[reduction][in_egress_model]['R0'], inc)
        T23 = Utils.T23_Winn(pr.ORBITAL_PERIOD['median'], best_params[reduction][in_egress_model]['aRs'], best_params[reduction][in_egress_model]['R0'], inc)
        
        T1[reduction] = best_params[reduction][in_egress_model]['Tmid'] - T14/2.
        T2[reduction] = best_params[reduction][in_egress_model]['Tmid'] - T23/2.
        T3[reduction] = best_params[reduction][in_egress_model]['Tmid'] + T23/2.
        T4[reduction] = best_params[reduction][in_egress_model]['Tmid'] + T14/2.
    
    T1['JWST/NIRISS'] += 2460244. - pr.TIME_0
    T2['JWST/NIRISS'] += 2460244. - pr.TIME_0
    T3['JWST/NIRISS'] += 2460244. - pr.TIME_0
    T4['JWST/NIRISS'] += 2460244. - pr.TIME_0
    
    # Plot the best-fit light curve model against the data for both detectors:
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    # Plot the best-fit light curve model against the data for both detectors:
    fig, ax = plt.subplots( 4, len(reductions), sharex='col', figsize=[7.5,8.7] )
    
    # Data
    for j, reduction in enumerate(reductions):
        models = list(best_params[reduction].keys())
        rescale_model = models[1]
        
        if scale_errorbars:
            errorbars = np.sqrt( data[reduction]['lcUncs']**2 + (best_params[reduction][rescale_model]['sigma_sys']*1.0E-6)**2 )
        else:
            errorbars = data[reduction]['lcUncs']
        
        for k in range(2):
            ax[k,j].scatter( data[reduction]['bjd_tdb'] - pr.TIME_0, data[reduction]['lcVals'], c='white', edgecolor='black', s=point_size, label='Data', zorder=2.0E03 )
            ax[k,j].errorbar( data[reduction]['bjd_tdb'] - pr.TIME_0, data[reduction]['lcVals'], yerr=errorbars, ecolor='black', elinewidth=errorbar_size, ls='none', zorder=1.0E03 )
            ax[k,j].axvspan(T1[reduction], T2[reduction], alpha=shade_alpha, color=shade_color)
            ax[k,j].axvspan(T3[reduction], T4[reduction], alpha=shade_alpha, color=shade_color)
            ax[k,j].yaxis.set_minor_locator(AutoMinorLocator())
        ax[-1,j].set_xlabel( 'Time [BJD$_{TDB}$' + ' - {:.0f}]'.format(pr.TIME_0) )
        
        # Models
        for i, model in enumerate(models):
            
            # First, initialize a batman model for every data point to change the transit radius as a function of time
            batpar, batmodel = calculateBatmanModel.create_Batman_arrays( data[reduction]['bjd_tdb'] )
            if reduction == 'JWST/NIRISS':
                best_model = calculateBatmanModel.Single_soss( best_params[reduction][model], data[reduction], batpar, batmodel )
            else:
                best_model = calculateBatmanModel.Single( best_params[reduction][model], data[reduction], batpar, batmodel )
            
            # Plot random draws from the fit params
            try:
                random_indices = np.random.randint( len( emcee_result[reduction][model]['flat_chain'] ), size=100)
                for index in random_indices:
                    parameters = emcee_result[reduction][model]['flat_chain'][index]
                    pars_dict = Utils.arrays2dict( parameters, emcee_result[reduction][model]['parameters'] )
                    pars_dict = Utils.merge_dictionaries(pars_dict, best_params[reduction][model])
                    if reduction == 'JWST/NIRISS':
                        random_model = calculateBatmanModel.Single_soss( pars_dict, data[reduction], batpar, batmodel )
                    else:
                        random_model = calculateBatmanModel.Single( pars_dict, data[reduction], batpar, batmodel )
                    for k in range(2):
                        ax[k,j].plot( data[reduction]['bjd_tdb'] - pr.TIME_0, random_model, mcmc_colors[i], linewidth=line_width, alpha=0.1, zorder=1.0E04)
            except TypeError:
                pass
            
            for k in range(2):
                ax[k,j].plot( data[reduction]['bjd_tdb'] - pr.TIME_0, best_model, model_colors[i], linewidth=line_width, label=model_labels[model], zorder=2.0E04 )
            try:
                ax[0,j].set_title( reduction + ', {0:.2f} - {1:.2f}$\,\mu$m'.format(data[reduction]['wavEdgesMicr'][0], data[reduction]['wavEdgesMicr'][1]) )
            except KeyError:
                ax[0,j].set_title( None )
            
            # Residuals
            unit = 1.0E06       # ppm
            residuals = (data[reduction]['lcVals'] - best_model)/data[reduction]['lcVals'] * unit
            
            # Perform binning
            bin_edges = np.linspace( min(data[reduction]['bjd_tdb']), max(data[reduction]['bjd_tdb']), 400 )
            residuals_binned, applied_bin_edges, indices = binned_statistic( data[reduction]['bjd_tdb'], residuals, bins=bin_edges )
            bin_middles = Utils.bin_edges_into_middles( bin_edges )
            
            ax[i+2,j].scatter( data[reduction]['bjd_tdb'] - pr.TIME_0, residuals, c='white', edgecolor=model_colors[i], s=point_size, zorder=2.0E03, label='Unbinned' )
            ax[i+2,j].errorbar( data[reduction]['bjd_tdb'] - pr.TIME_0, residuals, yerr=errorbars/data[reduction]['lcVals']*unit, ecolor=model_colors[i], elinewidth=errorbar_size, ls='none', zorder=1.0E03 )
            ax[i+2,j].plot( bin_middles - pr.TIME_0, residuals_binned, linewidth=line_width, color=mcmc_colors[i], zorder=1.0E04, label='Binned' )
            ax[i+2,j].axvspan(T1[reduction], T2[reduction], alpha=shade_alpha, color=shade_color)
            ax[i+2,j].axvspan(T3[reduction], T4[reduction], alpha=shade_alpha, color=shade_color)
            ax[i+2,j].yaxis.set_minor_locator(AutoMinorLocator())
            
            ax[i+2,1].legend( fontsize='small' )
            ax[i+2,0].set_ylabel(model_labels[model] + ' residuals [ppm]')
            ax[i+2,0].set_ylim( residual_limits )
    
    # Cosmetics
    for k in range(2):
        ax[k,0].set_ylabel('Normalized Flux')
    ax[0,0].xaxis.set_minor_locator(AutoMinorLocator())
    ax[0,1].legend( )
    
    ax[0,0].set_xlim( 377.945, 378.1 )
    ax[0,1].set_xlim( 0.5546, 0.7306)
    ax[1,0].set_ylim( 0.9835, 0.986 )
    ax[1,1].set_ylim( 0.9845, 0.986 )
    
    residual_limits = [-700., 700.]
    ax[2,0].set_ylim( residual_limits )
    ax[3,0].set_ylim( residual_limits )
    ax[2,1].set_ylim( residual_limits )
    ax[3,1].set_ylim( residual_limits )
    
    fig.tight_layout()
    
    if pr.SAVE_PLOTS and len(opath) > 0:
        fig.savefig( opath + '.pdf', dpi=400, bbox_inches='tight' )
        print( '\nSaved:',opath + '.pdf' )
    if pr.SHOW_PLOTS:
        plt.show()
    
    plt.close()


def spectral_light_curve( data, model, unit = 1.0E-2, opath = '', model_color='magenta' ):
    """
    Plots light curve data together with a model. The plot also shows the detector that
    measured the light curve as well as the wavelength range of the observation.
    """
    
    point_size = 6.
    errorbar_size = 0.6
    line_width = 1.2
    alpha_errorbars = 0.6
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    fig, ax = plt.subplots( 2, 1, sharex=True )
    
    ax[0].scatter( data['bjd_tdb'] - pr.TIME_0, data['lcVals'], edgecolor='black', c='white', s=point_size, label= 'Data', zorder=2 )
    ax[0].errorbar( data['bjd_tdb'] - pr.TIME_0, data['lcVals'], yerr=data['lcUncs'], ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none', zorder=1)
    
    ax[0].plot( data['bjd_tdb'] - pr.TIME_0, model, model_color, label='Model', linewidth=line_width, zorder=3 )
    ax[0].set_ylabel( 'Flux' )
    try:
        ax[0].set_title(data['detector'] + ', {0:.3f} - {1:.3f} microns'.format(data['wavEdgesMicr'][0], data['wavEdgesMicr'][1]))
    except KeyError:
        ax[0].set_title(data['detector'])
    ax[0].legend(fontsize='small')
    
    residuals = (data['lcVals'] - model)/data['lcVals'] / unit
    
    # Perform binning
    bin_edges = np.linspace( min(data['bjd_tdb']), max(data['bjd_tdb']), 100 )
    residuals_binned, applied_bin_edges, indices = binned_statistic( data['bjd_tdb'], residuals, bins=bin_edges )
    bin_middles = Utils.bin_edges_into_middles( bin_edges )
    
    ax[1].scatter ( data['bjd_tdb'] - pr.TIME_0, residuals, edgecolor='black', c='white', s=point_size, zorder=2, label='Unbinned' )
    ax[1].errorbar( data['bjd_tdb'] - pr.TIME_0, residuals, yerr=data['lcUncs']/data['lcVals'] / unit, ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none', zorder=1)
    ax[1].plot( bin_middles - pr.TIME_0, residuals_binned, color=model_color, linewidth=line_width, zorder=3, label='Binned' )
    ax[1].legend( loc='lower left' )
    ax[1].set_xlabel( 'Time [BJD_TDB - {:.0f}]'.format(pr.TIME_0) )
    if unit == 1.0E-2:
        ax[1].set_ylabel( 'Relative difference [\%]' )
    elif unit == 1.0E-6:
        ax[1].set_ylabel( 'Relative difference [ppm]' )
    else:
        ax[1].set_ylabel( 'Relative difference' )
    
    ax[0].set_xlim( np.min(data['t']), np.max(data['t']))
    if len(opath) > 0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
        print( '\nSaved:\n' + opath + '.pdf' )
    # if pr.SHOW_PLOTS:
    #     plt.show()
    
    plt.close()

def spectral_light_curve_nmodels( data, models, unit=1.0E-2, opath = '', emcee_result = None, label=None, model_color='magenta' ):
    """
    Plots light curve data together with a model. The plot also shows the detector that
    measured the light curve as well as the wavelength range of the observation.
    """
    
    point_size = 5.
    errorbar_size = 0.5
    line_width = 1.2
    alpha_errorbars = 0.6
    keys = list(models.keys())
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    fig, ax = plt.subplots( 2, len(keys), sharex=True, figsize=[8,4] )
    
    for j, key in enumerate(keys):
        ax[0,j].scatter( data['bjd_tdb'] - pr.TIME_0, data['lcVals'], edgecolor='black', c='white', s=point_size, label= 'Data', zorder=2 )
        ax[0,j].errorbar( data['bjd_tdb'] - pr.TIME_0, data['lcVals'], yerr=data['lcUncs'], ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none', zorder=1)
        ax[0,j].plot( data['bjd_tdb'] - pr.TIME_0, models[key], model_color, label='Model', linewidth=line_width, zorder=3 )
        ax[0,j].set_title( key )
        ax[0,j].legend(fontsize='small')
        
        residuals = (data['lcVals']-models[key])/data['lcVals'] / unit
        bin_edges = np.linspace( min(data['bjd_tdb']), max(data['bjd_tdb']), 500 )
        residuals_binned, applied_bin_edges, indices = binned_statistic( data['bjd_tdb'], residuals, bins=bin_edges )
        bin_middles = Utils.bin_edges_into_middles( bin_edges )
        ax[1,j].scatter ( data['bjd_tdb'] - pr.TIME_0, residuals, edgecolor='black', c='white', s=point_size, zorder=2, label='Unbinned' )
        ax[1,j].errorbar( data['bjd_tdb'] - pr.TIME_0, residuals, yerr=data['lcUncs']/data['lcVals'] / unit, ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none', zorder=1)
        ax[1,j].plot( bin_middles - pr.TIME_0, residuals_binned, color=model_color, linewidth=line_width, zorder=3, label='Binned' )
        ax[1,j].legend(fontsize='small', loc = 'lower left')
        
        ax[1,j].set_xlabel( 'Time [BJD_TDB - {:.0f}]'.format(pr.TIME_0) )
        
    # ax[0,0].set_xlim( np.min(data['t']), np.max(data['t']))
    ax[0,0].set_xlim( 0.5426, 0.7426)
    ax[0,0].set_ylabel( 'Flux' )
    if unit == 1.0E-2:
        ax[1,0].set_ylabel( 'Relative difference [\%]' )
    elif unit == 1.0E-6:
        ax[1,0].set_ylabel( 'Relative difference [ppm]' )
    else:
        ax[1,0].set_ylabel( 'Relative difference' )
    if len(opath) > 0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
        print( '\nSaved:\n' + opath + '.pdf' )
    if pr.SHOW_PLOTS:
        plt.show()
    
    plt.close()

def spectral_light_curve_mcmc( data, emcee_result, broad_fit_params, unit = 1.0E-2, opath = '', label=None, model_color='magenta' ):
    """
    Plots light curve data together with a model. The plot also shows the detector that
    measured the light curve as well as the wavelength range of the observation.
    """
    
    point_size = 6.
    errorbar_size = 0.6
    line_width = 1.2
    alpha_errorbars = 0.6
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    fig, ax = plt.subplots( 2, 1, sharex=True )
    
    # Data
    ax[0].scatter( data['bjd_tdb'] - pr.TIME_0, data['lcVals'], edgecolor='black', c='white', s=point_size, label= label + ' light curve', zorder=2 )
    ax[0].errorbar( data['bjd_tdb'] - pr.TIME_0, data['lcVals'], yerr=data['lcUncs'], ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none', zorder=1)
    
    # Plot random draws from the fit params
    batpar, batmodel = calculateBatmanModel.create_Batman_arrays( data['bjd_tdb'] )
    for j in range(len(batpar)):
        batpar[j] = calculateBatmanModel.fix_broad_fit_parameters(batpar[j], broad_fit_params)
    random_indices = np.random.randint( len( emcee_result['flat_chain'] ), size=100)
    i = 0
    for index in random_indices:
        parameters = emcee_result['flat_chain'][index]
        pars_dict = Utils.arrays2dict( parameters, emcee_result['parameters'] )
        random_model = calculateBatmanModel.Spec( pars_dict, data, batpar, batmodel )
        ax[0].plot( data['bjd_tdb'] - pr.TIME_0, random_model, 'yellow', linewidth=line_width, zorder=3, alpha=0.1)
        i += 1
    index = np.argmax( emcee_result['flat_log_probability'] )
    parameters = emcee_result['flat_chain'][index]
    pars_dict = Utils.arrays2dict( parameters, emcee_result['parameters'] )
    model = calculateBatmanModel.Spec( pars_dict, data, batpar, batmodel )
    
    ax[0].plot( data['bjd_tdb'] - pr.TIME_0, model, model_color, label='Model', linewidth=line_width, zorder=3 )
    ax[0].set_ylabel( 'Flux' )
    ax[0].set_title(data['detector'] + ', {0:.3f} - {1:.3f} microns'.format(data['wavEdgesMicr'][0], data['wavEdgesMicr'][1]))
    ax[0].legend(fontsize='small')
    
    residuals = (data['lcVals'] - model)/data['lcVals'] / unit
    
    # Perform binning
    bin_edges = np.linspace( min(data['bjd_tdb']), max(data['bjd_tdb']), 100 )
    residuals_binned, applied_bin_edges, indices = binned_statistic( data['bjd_tdb'], residuals, bins=bin_edges )
    bin_middles = Utils.bin_edges_into_middles( bin_edges )
    
    ax[1].scatter ( data['bjd_tdb'] - pr.TIME_0, residuals, edgecolor='black', c='white', s=point_size, zorder=2, label='Unbinned' )
    ax[1].errorbar( data['bjd_tdb'] - pr.TIME_0, residuals, yerr=data['lcUncs']/data['lcVals'] / unit, ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none', zorder=1)
    ax[1].plot( bin_middles - pr.TIME_0, residuals_binned, color=model_color, linewidth=line_width, zorder=3, label='Binned' )
    ax[1].legend( loc='lower left' )
    ax[1].set_xlabel( 'Time [BJD_TDB - {:.0f}]'.format(pr.TIME_0) )
    if unit == 1.0E-2:
        ax[1].set_ylabel( 'Relative difference [\%]' )
    elif unit == 1.0E-6:
        ax[1].set_ylabel( 'Relative difference [ppm]' )
    else:
        ax[1].set_ylabel( 'Relative difference' )
    
    ax[0].set_xlim( np.min(data['t']), np.max(data['t']))
    if len(opath) > 0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
        print( '\nSaved:\n' + opath + '.pdf' )
    plt.show()
    
    plt.close()

def spectral_light_curve_jason( data, best_params, broad_fit_params, emcee_result=None, unit = 1.0E-2, opath = '', label=None, model_color='magenta' ):
    """
    Plots light curve data together with a model. The plot also shows the detector that
    measured the light curve as well as the wavelength range of the observation.
    """
    
    point_size = 6.
    errorbar_size = 0.6
    line_width = 1.2
    alpha_errorbars = 0.6
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    fig, ax = plt.subplots( 2, 1, sharex=True )
    
    # Data
    ax[0].scatter( data['bjd_tdb'] - pr.TIME_0, data['lcVals'], edgecolor='black', c='white', s=point_size, label= label + ' light curve', zorder=2 )
    ax[0].errorbar( data['bjd_tdb'] - pr.TIME_0, data['lcVals'], yerr=data['lcUncs'], ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none', zorder=1)
    
    # Initiate batman for model evaluations
    batpar, batmodel = calculateBatmanModel.create_Batman_arrays( data['bjd_tdb'] )
    for j in range(len(batpar)):
        batpar[j] = calculateBatmanModel.fix_broad_fit_parameters(batpar[j], broad_fit_params)
    
    # Plot random draws from the fit params
    try:
        random_indices = np.random.randint( len( emcee_result['flat_chain'] ), size=100)
        for index in random_indices:
            parameters = emcee_result['flat_chain'][index]
            pars_dict = Utils.arrays2dict( parameters, emcee_result['parameters'] )
            random_model = calculateBatmanModel.Spec( pars_dict, data, batpar, batmodel )
            ax[0].plot( data['bjd_tdb'] - pr.TIME_0, random_model, 'yellow', linewidth=line_width, zorder=3, alpha=0.1)
    except TypeError:
        pass
    
    # Plot best fit
    model = calculateBatmanModel.Spec( best_params, data, batpar, batmodel )
    
    ax[0].plot( data['bjd_tdb'] - pr.TIME_0, model, model_color, label='ML Model', linewidth=line_width, zorder=3 )
    ax[0].set_ylabel( 'Flux' )
    try:
        ax[0].set_title(data['detector'] + ', {0:.3f} - {1:.3f} microns'.format(data['wavEdgesMicr'][0], data['wavEdgesMicr'][1]))
    except KeyError:
        ax[0].set_title(data['detector'])
    ax[0].legend(fontsize='small')
    
    residuals = (data['lcVals'] - model)/data['lcVals'] / unit
    
    # Perform binning
    bin_edges = np.linspace( min(data['bjd_tdb']), max(data['bjd_tdb']), 100 )
    residuals_binned, applied_bin_edges, indices = binned_statistic( data['bjd_tdb'], residuals, bins=bin_edges )
    bin_middles = Utils.bin_edges_into_middles( bin_edges )
    
    ax[1].scatter ( data['bjd_tdb'] - pr.TIME_0, residuals, edgecolor='black', c='white', s=point_size, zorder=2, label='Unbinned' )
    ax[1].errorbar( data['bjd_tdb'] - pr.TIME_0, residuals, yerr=data['lcUncs']/data['lcVals'] / unit, ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none', zorder=1)
    ax[1].plot( bin_middles - pr.TIME_0, residuals_binned, color=model_color, linewidth=line_width, zorder=3, label='Binned' )
    ax[1].legend( loc='lower left' )
    ax[1].set_xlabel( 'Time [BJD_TDB - {:.0f}]'.format(pr.TIME_0) )
    if unit == 1.0E-2:
        ax[1].set_ylabel( 'Relative difference [\%]' )
    elif unit == 1.0E-6:
        ax[1].set_ylabel( 'Relative difference [ppm]' )
    else:
        ax[1].set_ylabel( 'Relative difference' )
    
    ax[0].set_xlim( np.min(data['t']), np.max(data['t']))
    if len(opath) > 0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
        print( '\nSaved:\n' + opath + '.pdf' )
    if pr.SHOW_PLOTS:
        plt.show()
    
    plt.close()

def all_spectral_light_curves( lc, best_params, emcee_result=None, prelim_fit=None, unit = 1.0E-6, opath = '' ):
    
    point_size = 6.
    errorbar_size = 0.6
    line_width = 1.2
    alpha_errorbars = 0.6
    channels = list(lc.keys())
    model_color = 'orange'
    
    for ch in channels:
        
        batpar, batmodel = calculateBatmanModel.create_Batman_arrays( lc[ch]['bjd_tdb'] )
        
        plt.clf()
        plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
        
        fig, ax = plt.subplots( 2, 1, sharex=True )
        
        # Data
        ax[0].scatter( lc[ch]['bjd_tdb'] - pr.TIME_0, lc[ch]['lcVals'], edgecolor='black', c='white', s=point_size, label= 'Data', zorder=2 )
        ax[0].errorbar( lc[ch]['bjd_tdb'] - pr.TIME_0, lc[ch]['lcVals'], yerr=lc[ch]['lcUncs'], ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none', zorder=1)
        
        # Plot random draws from the fit params
        try:
            random_indices = np.random.randint( len( emcee_result['flat_chain'] ), size=100)
            for index in random_indices:
                parameters = emcee_result['flat_chain'][index]
                pars_dict = Utils.arrays2dict( parameters, emcee_result['parameters'] )
                pars_dict = Utils.merge_dictionaries( pars_dict, prelim_fit )
                random_model = calculateBatmanModel.all_lcs( pars_dict, lc[ch], batpar, batmodel, ch )
                ax[0].plot( lc[ch]['bjd_tdb'] - pr.TIME_0, random_model, 'yellow', linewidth=line_width, zorder=3, alpha=0.1)
        except TypeError:
            pass
        
        # Plot best fit
        model = calculateBatmanModel.all_lcs( best_params, lc[ch], batpar, batmodel, ch )
        
        ax[0].plot( lc[ch]['bjd_tdb'] - pr.TIME_0, model, model_color, label='ML Model', linewidth=line_width, zorder=3 )
        ax[0].set_ylabel( 'Flux' )
        try:
            ax[0].set_title(lc[ch]['detector'] + ', {0:.3f} - {1:.3f} microns'.format(lc[ch]['wavEdgesMicr'][0], lc[ch]['wavEdgesMicr'][1]))
        except KeyError:
            ax[0].set_title(lc[ch]['detector'])
        ax[0].legend(fontsize='small')
        
        residuals = (lc[ch]['lcVals'] - model)/lc[ch]['lcVals'] / unit
        
        # Perform binning
        bin_edges = np.linspace( min(lc[ch]['bjd_tdb']), max(lc[ch]['bjd_tdb']), 100 )
        residuals_binned, applied_bin_edges, indices = binned_statistic( lc[ch]['bjd_tdb'], residuals, bins=bin_edges )
        bin_middles = Utils.bin_edges_into_middles( bin_edges )
    
        ax[1].scatter ( lc[ch]['bjd_tdb'] - pr.TIME_0, residuals, edgecolor='black', c='white', s=point_size, zorder=2, label='Unbinned' )
        ax[1].errorbar( lc[ch]['bjd_tdb'] - pr.TIME_0, residuals, yerr=lc[ch]['lcUncs']/lc[ch]['lcVals'] / unit, ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none', zorder=1)
        ax[1].plot( bin_middles - pr.TIME_0, residuals_binned, color=model_color, linewidth=line_width, zorder=3, label='Binned' )
        ax[1].legend( loc='lower left' )
        ax[1].set_xlabel( 'Time [BJD_TDB - {:.0f}]'.format(pr.TIME_0) )
        if unit == 1.0E-2:
            ax[1].set_ylabel( 'Relative difference [\%]' )
        elif unit == 1.0E-6:
            ax[1].set_ylabel( 'Relative difference [ppm]' )
        else:
            ax[1].set_ylabel( 'Relative difference' )
        
        ax[0].set_xlim( np.min( lc[ch]['t']), np.max(lc[ch]['t']))
        if len(opath) > 0:
            pylab.savefig( opath + ch + '.pdf', bbox_inches='tight' )
            print( '\nSaved:\n' + opath + ch + '.pdf' )
        if pr.SHOW_PLOTS:
            plt.show()
        plt.close()

def n_lightcurves_m_models( data, best_params, prelim_params, white_fit_params, emcee_result=None, unit = 1.0E-2, opath = '' ):
    """
    Plots light curve data together with a model. The plot also shows the detector that
    measured the light curve as well as the wavelength range of the observation.
    """
    
    point_size = 6.
    errorbar_size = 0.6
    line_width = 1.2
    alpha_errorbars = 1.0
    model_colors = ['blue', 'darkorange']
    mcmc_colors = ['aqua', 'yellow']
    shade_alpha = 0.2
    shade_color = 'gray'
    
    molecules = list(data.keys())
    n_models = len(best_params[molecules[0]].keys())
    
    # Calculate times of transit ingress and egress
    in_egress_model = 'Translational model'
    T14 = {}
    T23 = {}
    for molecule in molecules:
        # Using the equations from Winn et al. (2010)
        T14[molecule] = Utils.T14_Winn(pr.ORBITAL_PERIOD['median'], white_fit_params[in_egress_model]['aRs'], best_params[molecule][in_egress_model]['R0'], white_fit_params[in_egress_model]['inc'])
        T23[molecule] = Utils.T23_Winn(pr.ORBITAL_PERIOD['median'], white_fit_params[in_egress_model]['aRs'], best_params[molecule][in_egress_model]['R0'], white_fit_params[in_egress_model]['inc'])
    
    T1 = {}
    T2 = {}
    T3 = {}
    T4 = {}
    
    for molecule in molecules:
        T1[molecule] = best_params[molecule][in_egress_model]['Tmid'] - T14[molecule]/2.
        T2[molecule] = best_params[molecule][in_egress_model]['Tmid'] - T23[molecule]/2.
        T3[molecule] = best_params[molecule][in_egress_model]['Tmid'] + T23[molecule]/2.
        T4[molecule] = best_params[molecule][in_egress_model]['Tmid'] + T14[molecule]/2.
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    fig, ax = plt.subplots( 2 + n_models, len(molecules), figsize=[8.2,9], sharex=True )
    
    for j, molecule in enumerate(molecules):
        # Initiate batman for model evaluations
        batpar, batmodel = calculateBatmanModel.create_Batman_arrays( data[molecule]['bjd_tdb'] )
        
        # Data
        for i in range(2):
            ax[i,j].scatter( data[molecule]['bjd_tdb'] - pr.TIME_0, data[molecule]['lcVals'], edgecolor='black', c='white', s=point_size, label= 'Data', zorder=2 )
            ax[i,j].errorbar( data[molecule]['bjd_tdb'] - pr.TIME_0, data[molecule]['lcVals'], yerr=data[molecule]['lcUncs'], ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none', zorder=1)
            ax[i,j].yaxis.set_minor_locator(AutoMinorLocator())
            ax[i,j].axvspan(T1[molecule], T2[molecule], alpha=shade_alpha, color=shade_color)
            ax[i,j].axvspan(T3[molecule], T4[molecule], alpha=shade_alpha, color=shade_color)
        
        # Models
        models = list(best_params[molecule].keys())
        for k, model in enumerate(models):
            for i in range(len(batpar)):
                batpar[i] = calculateBatmanModel.fix_broad_fit_parameters(batpar[i], white_fit_params[model])
            best_params_plot = Utils.merge_dictionaries( best_params[molecule][model], prelim_params[molecule][model] )
            best_model = calculateBatmanModel.Spec( best_params_plot, data[molecule], batpar, batmodel )
            for i in range(2):
                ax[i,j].plot( data[molecule]['bjd_tdb'] - pr.TIME_0, best_model, model_colors[k], label=model, linewidth=line_width, zorder=4 )
            
            residuals = (data[molecule]['lcVals'] - best_model)/data[molecule]['lcVals'] / unit
            # Utils.write_dictionary_to_csv_file( {'Time from mid-transit [s]' : (data[molecule]['bjd_tdb']-pr.TIME_0-best_params_plot['Tmid'])*24.*3600., 'Residuals [ppm]' : residuals}, 'residuals_' + molecule + '_' + model )
            
            # Perform binning
            bin_edges = np.linspace( min(data[molecule]['bjd_tdb']), max(data[molecule]['bjd_tdb']), 100 )
            residuals_binned, applied_bin_edges, indices = binned_statistic( data[molecule]['bjd_tdb'], residuals, bins=bin_edges )
            bin_middles = Utils.bin_edges_into_middles( bin_edges )
            
            ax[k+2,j].scatter ( data[molecule]['bjd_tdb'] - pr.TIME_0, residuals, edgecolor=model_colors[k], c='white', s=point_size, zorder=2, label='Unbinned' )
            ax[k+2,j].errorbar( data[molecule]['bjd_tdb'] - pr.TIME_0, residuals, yerr=data[molecule]['lcUncs']/data[molecule]['lcVals'] / unit, ecolor=model_colors[k], elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none', zorder=1)
            ax[k+2,j].plot( bin_middles - pr.TIME_0, residuals_binned, color=mcmc_colors[k], linewidth=line_width, zorder=3, label='Binned' )
            ax[k+2,j].axvspan(T1[molecule], T2[molecule], alpha=shade_alpha, color=shade_color)
            ax[k+2,j].axvspan(T3[molecule], T4[molecule], alpha=shade_alpha, color=shade_color)
            
            if unit == 1.0E-2:
                label = model + ' residuals [\%]'
            elif unit == 1.0E-6:
                label = model + ' residuals [ppm]'
            else:
                label = model + ' residuals [ppm]'
            ax[k+2,0].set_ylabel( label )
            
            # Plot random draws from the fit params
            try:
                random_indices = np.random.randint( len( emcee_result[molecule][model]['flat_chain'] ), size=100)
                for index in random_indices:
                    parameters = emcee_result[molecule][model]['flat_chain'][index]
                    pars_dict = Utils.arrays2dict( parameters, emcee_result[molecule][model]['parameters'] )
                    pars_dict = Utils.merge_dictionaries(pars_dict, prelim_params[molecule][model] )
                    random_model = calculateBatmanModel.Spec( pars_dict, data[molecule], batpar, batmodel )
                    for i in range(2):
                        ax[i,j].plot( data[molecule]['bjd_tdb'] - pr.TIME_0, random_model, mcmc_colors[k], linewidth=line_width, zorder=3, alpha=0.1)
            except IndexError:
                pass
            
            ax[k+2,j].yaxis.set_minor_locator(AutoMinorLocator())
        
        ax[-1,j].set_xlabel( 'Time [BJD_TDB - {:.0f}]'.format(pr.TIME_0) )
        ax[0,j].set_title( molecule + ' band, {0:.2f} - {1:.2f}$\,\mu$m'.format(data[molecule]['wavEdgesMicr'][0], data[molecule]['wavEdgesMicr'][1]))
    
    ax[0,0].legend()
    # ax[1,0].legend( fontsize='x-small' )
    ax[2,0].legend( loc='lower left' )
    ax[3,0].legend( loc='lower left' )
    ax[0,0].set_ylabel( 'Normalized flux' )
    ax[1,0].set_ylabel( 'Normalized flux' )
    ax[0,0].xaxis.set_minor_locator(AutoMinorLocator())
    
    ax[0,0].set_xlim( 0.5546, 0.7306)
    ax[1,0].set_ylim( 0.98375, 0.985 )
    ax[1,1].set_ylim( 0.9832, 0.9847 )
    
    lc_limits = [0.9827, 1.001]
    ax[0,0].set_ylim( lc_limits )
    ax[0,1].set_ylim( lc_limits )
    
    residual_limits = [-750., 750.]
    ax[2,0].set_ylim( residual_limits )
    ax[3,0].set_ylim( residual_limits )
    ax[2,1].set_ylim( residual_limits )
    ax[3,1].set_ylim( residual_limits )
    
    fig.tight_layout()
    if len(opath) > 0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
        print( '\nSaved:\n' + opath + '.pdf' )
    plt.show()
    
    plt.close()

def white_lightcurves_n_models( data, best_params, lc_limits=[0.983, 1.0005], residual_limits=[-500., 500.], opath = '', emcee_result = None ):
    """
    Plots light curves for both the NRS1 and the NRS2 detectors. Adds panels with the residuals.
    """
    
    point_size = 6.
    errorbar_size = 0.6
    model_colors = ['blue', 'darkorange']
    mcmc_colors = ['aqua', 'yellow']
    shade_alpha = 0.2
    shade_color = 'gray'
    line_width = 1.
    
    detectors = list(data.keys())
    models = list(best_params.keys())
    
    # Calculate times of transit ingress and egress
    in_egress_model = 'Translational'
    T14 = {}
    T23 = {}
    try:
        for nrs in detectors:
            # Using the equations from Winn et al. (2010)
            T14[nrs] = Utils.T14_Winn(pr.ORBITAL_PERIOD['median'], best_params[in_egress_model]['aRs'], best_params[in_egress_model]['R0_' + nrs], best_params[in_egress_model]['inc'])
            T23[nrs] = Utils.T23_Winn(pr.ORBITAL_PERIOD['median'], best_params[in_egress_model]['aRs'], best_params[in_egress_model]['R0_' + nrs], best_params[in_egress_model]['inc'])
    except KeyError:
        for nrs in detectors:
            # Using the equations from Winn et al. (2010)
            T14[nrs] = Utils.T14_Winn(pr.ORBITAL_PERIOD['median'], best_params[in_egress_model]['aRs'], best_params[in_egress_model]['R0_' + nrs], best_params[in_egress_model]['inc'])
            T23[nrs] = Utils.T23_Winn(pr.ORBITAL_PERIOD['median'], best_params[in_egress_model]['aRs'], best_params[in_egress_model]['R0_' + nrs], best_params[in_egress_model]['inc'])
    
    T1 = {}
    T2 = {}
    T3 = {}
    T4 = {}
    for nrs in detectors:
        if pr.FIT_TMID_SPEC:
            T1[nrs] = best_params[in_egress_model]['Tmid_' + nrs] - T14[nrs]/2.
            T2[nrs] = best_params[in_egress_model]['Tmid_' + nrs] - T23[nrs]/2.
            T3[nrs] = best_params[in_egress_model]['Tmid_' + nrs] + T23[nrs]/2.
            T4[nrs] = best_params[in_egress_model]['Tmid_' + nrs] + T14[nrs]/2.
        else:
            T1[nrs] = best_params[in_egress_model]['Tmid'] - T14[nrs]/2.
            T2[nrs] = best_params[in_egress_model]['Tmid'] - T23[nrs]/2.
            T3[nrs] = best_params[in_egress_model]['Tmid'] + T23[nrs]/2.
            T4[nrs] = best_params[in_egress_model]['Tmid'] + T14[nrs]/2.
        
    # Plot the best-fit light curve model against the data for both detectors:
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    # Plot the best-fit light curve model against the data for both detectors:
    fig, ax = plt.subplots( len(models) + 1, len(detectors), sharex=True, sharey='row', figsize=[7.5,6.75] )
    
    # Data
    for j, nrs in enumerate(detectors):
        ax[0,j].scatter( data[nrs]['bjd_tdb'] - pr.TIME_0, data[nrs]['lcVals'], c='white', edgecolor='black', s=point_size, label='Data', zorder=2.0E03 )
        ax[0,j].errorbar( data[nrs]['bjd_tdb'] - pr.TIME_0, data[nrs]['lcVals'], yerr=data[nrs]['lcUncs'], ecolor='black', elinewidth=errorbar_size, ls='none', zorder=1.0E03 )
        ax[0,j].axvspan(T1[nrs], T2[nrs], alpha=shade_alpha, color=shade_color)
        ax[0,j].axvspan(T3[nrs], T4[nrs], alpha=shade_alpha, color=shade_color)
        ax[-1,j].set_xlabel( 'Time [BJD$_{TDB}$' + ' - {:.0f}]'.format(pr.TIME_0) )
        ax[0,j].yaxis.set_minor_locator(AutoMinorLocator())
    
    # Models
    for i, model in enumerate(models):
        
        # First, initialize a batman model for every data point to change the transit radius as a function of time
        batpar = {}
        batmodel = {}
        best_model = {}
        for nrs in detectors:
            batpar[nrs], batmodel[nrs] = calculateBatmanModel.create_Batman_arrays( data[nrs]['bjd_tdb'] )
            best_model[nrs] = calculateBatmanModel.Broad( best_params[model], data[nrs], batpar[nrs], batmodel[nrs], nrs )
        
        # Plot random draws from the fit params
        try:
            random_indices = np.random.randint( len( emcee_result[model]['flat_chain'] ), size=100)
            for j, nrs in enumerate(detectors):
                for index in random_indices:
                    parameters = emcee_result[model]['flat_chain'][index]
                    pars_dict = Utils.arrays2dict( parameters, emcee_result[model]['parameters'] )
                    pars_dict = Utils.merge_dictionaries(pars_dict, best_params[model])
                    random_model = calculateBatmanModel.Broad( pars_dict, data[nrs], batpar[nrs], batmodel[nrs], nrs )
                    ax[0,j].plot( data[nrs]['bjd_tdb'] - pr.TIME_0, random_model, mcmc_colors[i], linewidth=line_width, alpha=0.1, zorder=1.0E04)
        except TypeError:
            pass
        
        for j, nrs in enumerate(detectors):
            ax[0,j].plot( data[nrs]['bjd_tdb'] - pr.TIME_0, best_model[nrs], model_colors[i], linewidth=line_width, label=model + ' model', zorder=2.0E04 )
            try:
                ax[0,j].set_title( nrs + ', {:.2f} - {:.2f} microns'.format(data[nrs]['wavEdgesMicr'][0], data[nrs]['wavEdgesMicr'][1]) )
            except KeyError:
                ax[0,j].set_title( nrs )
    
        # Residuals
        unit = 1.0E06       # ppm
        for j, nrs in enumerate(detectors):
            residuals = (data[nrs]['lcVals'] - best_model[nrs])/data[nrs]['lcVals'] * unit#/data[nrs]['lcUncs']
            
            # Perform binning
            bin_edges = np.linspace( min(data[nrs]['bjd_tdb']), max(data[nrs]['bjd_tdb']), 100 )
            residuals_binned, applied_bin_edges, indices = binned_statistic( data[nrs]['bjd_tdb'], residuals, bins=bin_edges )
            bin_middles = Utils.bin_edges_into_middles( bin_edges )
            
            ax[i+1,j].scatter( data[nrs]['bjd_tdb'] - pr.TIME_0, residuals, c='white', edgecolor=model_colors[i], s=point_size, zorder=2.0E03, label='Unbinned' )
            ax[i+1,j].errorbar( data[nrs]['bjd_tdb'] - pr.TIME_0, residuals, yerr=data[nrs]['lcUncs']/data[nrs]['lcVals']*unit, ecolor=model_colors[i], elinewidth=errorbar_size, ls='none', zorder=1.0E03 )
            ax[i+1,j].plot( bin_middles - pr.TIME_0, residuals_binned, linewidth=line_width, color=mcmc_colors[i], zorder=1.0E04, label='Binned' )
            ax[i+1,j].axvspan(T1[nrs], T2[nrs], alpha=shade_alpha, color=shade_color)
            ax[i+1,j].axvspan(T3[nrs], T4[nrs], alpha=shade_alpha, color=shade_color)
            ax[i+1,j].yaxis.set_minor_locator(AutoMinorLocator())
            
        ax[i+1,0].legend( )
        ax[i+1,0].set_ylabel(model + ' model residuals [ppm]')
        ax[i+1,0].set_ylim( residual_limits )
    
    # Cosmetics
    ax[0,0].xaxis.set_minor_locator(AutoMinorLocator())
    ax[0,0].set_ylabel('Normalized Flux')
    ax[0,0].legend( )
    ax[0,0].set_xlim( 0.5546, 0.7306)
    ax[0,0].set_ylim( lc_limits )
    fig.tight_layout()
    
    if pr.SAVE_PLOTS and len(opath) > 0:
        fig.savefig( opath + '.pdf', dpi=300, bbox_inches='tight' )
        print( '\nSaved:',opath + '.png' )
    if pr.SHOW_PLOTS:
        plt.show()
    
    plt.close()

def white_lightcurves_n_models_zoom( data, best_params, emcee_result = None, lc_limits=[0.983, 1.0005], zoom_limits=[0.9837, 0.985], residual_limits=[-500., 500.], scale_errorbars=False, opath = '' ):
    """
    Plots light curves for both the NRS1 and the NRS2 detectors. Adds panels with the residuals.
    """
    
    point_size = 5.
    errorbar_size = 0.4
    model_colors = ['blue', 'darkorange']
    mcmc_colors = ['aqua', 'yellow']
    shade_alpha = 0.2
    shade_color = 'gray'
    line_width = 1.
    
    detectors = list(data.keys())
    models = list(best_params.keys())
    
    # Calculate times of transit ingress and egress
    in_egress_model = models[0]
    rescale_model = models[1]
    if pr.IMPACT_PARAMETER:
        inc = Utils.impact_parameter2inc( best_params[in_egress_model]['b'], best_params[in_egress_model]['aRs'])
    else:
        inc = best_params[in_egress_model]['inc']
        
    T14 = {}
    T23 = {}
    try:
        for nrs in detectors:
            # Using the equations from Winn et al. (2010)
            T14[nrs] = Utils.T14_Winn(pr.ORBITAL_PERIOD['median'], best_params[in_egress_model]['aRs'], best_params[in_egress_model]['R0_' + nrs], inc)
            T23[nrs] = Utils.T23_Winn(pr.ORBITAL_PERIOD['median'], best_params[in_egress_model]['aRs'], best_params[in_egress_model]['R0_' + nrs], inc)
    except KeyError:
        for nrs in detectors:
            # Using the equations from Winn et al. (2010)
            T14[nrs] = Utils.T14_Winn(pr.ORBITAL_PERIOD['median'], best_params[in_egress_model]['aRs'], best_params[in_egress_model]['R0_' + nrs], inc)
            T23[nrs] = Utils.T23_Winn(pr.ORBITAL_PERIOD['median'], best_params[in_egress_model]['aRs'], best_params[in_egress_model]['R0_' + nrs], inc)
    
    T1 = {}
    T2 = {}
    T3 = {}
    T4 = {}
    for nrs in detectors:
        if pr.FIT_TMID_SPEC:
            T1[nrs] = best_params[in_egress_model]['Tmid_' + nrs] - T14[nrs]/2.
            T2[nrs] = best_params[in_egress_model]['Tmid_' + nrs] - T23[nrs]/2.
            T3[nrs] = best_params[in_egress_model]['Tmid_' + nrs] + T23[nrs]/2.
            T4[nrs] = best_params[in_egress_model]['Tmid_' + nrs] + T14[nrs]/2.
        else:
            T1[nrs] = best_params[in_egress_model]['Tmid'] - T14[nrs]/2.
            T2[nrs] = best_params[in_egress_model]['Tmid'] - T23[nrs]/2.
            T3[nrs] = best_params[in_egress_model]['Tmid'] + T23[nrs]/2.
            T4[nrs] = best_params[in_egress_model]['Tmid'] + T14[nrs]/2.
    
    # Plot the best-fit light curve model against the data for both detectors:
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    # Plot the best-fit light curve model against the data for both detectors:
    fig, ax = plt.subplots( len(models) + 2, len(detectors), sharex=True, figsize=[7.5,8.7] )
    
    # Data
    for j, nrs in enumerate(detectors):
        if scale_errorbars:
            errorbars = np.sqrt( data[nrs]['lcUncs']**2 + (best_params[rescale_model]['sigma_sys_' + nrs]*1.0E-6)**2 )
        else:
            errorbars = data[nrs]['lcUncs']
            
        for k in range(2):
            ax[k,j].scatter( data[nrs]['bjd_tdb'] - pr.TIME_0, data[nrs]['lcVals'], c='white', edgecolor='black', s=point_size, label='Data', zorder=2.0E03 )
            ax[k,j].errorbar( data[nrs]['bjd_tdb'] - pr.TIME_0, data[nrs]['lcVals'], yerr=errorbars, ecolor='black', elinewidth=errorbar_size, ls='none', zorder=1.0E03 )
            ax[k,j].axvspan(T1[nrs], T2[nrs], alpha=shade_alpha, color=shade_color)
            ax[k,j].axvspan(T3[nrs], T4[nrs], alpha=shade_alpha, color=shade_color)
            ax[k,j].yaxis.set_minor_locator(AutoMinorLocator())
        ax[-1,j].set_xlabel( 'Time [BJD$_{TDB}$' + ' - {:.0f}]'.format(pr.TIME_0) )
    
    # Models
    for i, model in enumerate(models):
        
        # First, initialize a batman model for every data point to change the transit radius as a function of time
        batpar = {}
        batmodel = {}
        best_model = {}
        for nrs in detectors:
            batpar[nrs], batmodel[nrs] = calculateBatmanModel.create_Batman_arrays( data[nrs]['bjd_tdb'] )
            best_model[nrs] = calculateBatmanModel.Broad( best_params[model], data[nrs], batpar[nrs], batmodel[nrs], nrs )
        
        # Plot random draws from the fit params
        try:
            random_indices = np.random.randint( len( emcee_result[model]['flat_chain'] ), size=100)
            for j, nrs in enumerate(detectors):
                for index in random_indices:
                    parameters = emcee_result[model]['flat_chain'][index]
                    pars_dict = Utils.arrays2dict( parameters, emcee_result[model]['parameters'] )
                    pars_dict = Utils.merge_dictionaries(pars_dict, best_params[model])
                    random_model = calculateBatmanModel.Broad( pars_dict, data[nrs], batpar[nrs], batmodel[nrs], nrs )
                    for k in range(2):
                        ax[k,j].plot( data[nrs]['bjd_tdb'] - pr.TIME_0, random_model, mcmc_colors[i], linewidth=line_width, alpha=0.1, zorder=1.0E04)
        except TypeError:
            pass
        
        for j, nrs in enumerate(detectors):
            for k in range(2):
                ax[k,j].plot( data[nrs]['bjd_tdb'] - pr.TIME_0, best_model[nrs], model_colors[i], linewidth=line_width, label=model, zorder=2.0E04 )
            try:
                ax[0,j].set_title( nrs + ', {:.2f} - {:.2f} microns'.format(data[nrs]['wavEdgesMicr'][0], data[nrs]['wavEdgesMicr'][1]) )
            except KeyError:
                ax[0,j].set_title( nrs + ', Eureka! reduction' )
            
        # Residuals
        unit = 1.0E06       # ppm
        for j, nrs in enumerate(detectors):
            if scale_errorbars:
                errorbars = np.sqrt( data[nrs]['lcUncs']**2 + (best_params[rescale_model]['sigma_sys_' + nrs]*1.0E-6)**2 )
            else:
                errorbars = data[nrs]['lcUncs']
            
            residuals = (data[nrs]['lcVals'] - best_model[nrs])/data[nrs]['lcVals'] * unit#/data[nrs]['lcUncs']
            
            # Perform binning
            bin_edges = np.linspace( min(data[nrs]['bjd_tdb']), max(data[nrs]['bjd_tdb']), 400 )
            residuals_binned, applied_bin_edges, indices = binned_statistic( data[nrs]['bjd_tdb'], residuals, bins=bin_edges )
            bin_middles = Utils.bin_edges_into_middles( bin_edges )
            
            ax[i+2,j].scatter( data[nrs]['bjd_tdb'] - pr.TIME_0, residuals, c='white', edgecolor=model_colors[i], s=point_size, zorder=2.0E03, label='Unbinned' )
            ax[i+2,j].errorbar( data[nrs]['bjd_tdb'] - pr.TIME_0, residuals, yerr=errorbars/data[nrs]['lcVals']*unit, ecolor=model_colors[i], elinewidth=errorbar_size, ls='none', zorder=1.0E03 )
            ax[i+2,j].plot( bin_middles - pr.TIME_0, residuals_binned, linewidth=line_width, color=mcmc_colors[i], zorder=1.0E04, label='Binned' )
            ax[i+2,j].axvspan(T1[nrs], T2[nrs], alpha=shade_alpha, color=shade_color)
            ax[i+2,j].axvspan(T3[nrs], T4[nrs], alpha=shade_alpha, color=shade_color)
            ax[i+2,j].yaxis.set_minor_locator(AutoMinorLocator())
            
        ax[i+2,0].legend( fontsize='small' )
        ax[i+2,0].set_ylabel(model + ' residuals [ppm]')
        ax[i+2,0].set_ylim( residual_limits )
    
    # Cosmetics
    for k in range(2):
        ax[k,0].set_ylabel('Normalized Flux')
    ax[0,0].xaxis.set_minor_locator(AutoMinorLocator())
    ax[0,0].legend( )
    
    ax[0,0].set_xlim( 0.5546, 0.7306)
    ax[0,0].set_ylim( [0.983, 1.0005] )
    ax[0,1].set_ylim( [0.984, 1.0015] )
    
    # Zoom
    ax[1,0].set_ylim( 0.98382, 0.985 )
    ax[1,1].set_ylim( 0.9848, 0.986 )
    
    residual_limits = [-500., 500.]
    ax[2,0].set_ylim( residual_limits )
    ax[3,0].set_ylim( residual_limits )
    ax[2,1].set_ylim( residual_limits )
    ax[3,1].set_ylim( residual_limits )
    
    fig.tight_layout()
    
    if pr.SAVE_PLOTS and len(opath) > 0:
        fig.savefig( opath + '.pdf', dpi=400, bbox_inches='tight' )
        print( '\nSaved:',opath + '.pdf' )
    if pr.SHOW_PLOTS:
        plt.show()
    
    plt.close()


def spectral_light_curve_residuals( z, best_params, broad_fit_params, opath = '', emcee_result = None ):
    """
    Plots light curve data together with a model. The plot also shows the detector that
    measured the light curve as well as the wavelength range of the observation.
    """
    
    point_size = 2.
    errorbar_size = point_size/3.
    unit = 1.0E02       # percent
    alpha_errorbars = 0.6
    
    # Calculate the best fit model
    batpar, batmodel = calculateBatmanModel.create_Batman_object( z['bjd_tdb'] )
    batpar = calculateBatmanModel.fix_broad_fit_parameters( batpar, broad_fit_params )
    best_model = calculateBatmanModel.Spec( best_params, z, batpar, batmodel )
    
    residuals = (z['lcVals']-best_model)/z['lcVals'] * unit
    
    # Perform binning
    bin_edges = np.linspace( min(z['bjd_tdb']), max(z['bjd_tdb']), 100 )
    residuals_binned, applied_bin_edges, indices = binned_statistic( z['bjd_tdb'], residuals, bins=bin_edges )
    bin_middles = Utils.bin_edges_into_middles( bin_edges )
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    
    fig, ax = plt.subplots( 2, 1, sharex=True )
    
    ax[0].scatter( z['bjd_tdb'], z['lcVals'], c='black', s=point_size, label='Data' )
    ax[0].errorbar( z['bjd_tdb'], z['lcVals'], yerr=z['lcUncs'], ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none')
    
    try:
        # Plot random draws from the fit params
        random_indices = np.random.randint( len( emcee_result['flat_chain'] ), size=100)
        i = 0
        for index in random_indices:
            parameters = emcee_result['flat_chain'][index]
            pars_dict = Utils.arrays2dict( parameters, emcee_result['parameters'] )
            pars_dict = Utils.merge_dictionaries(pars_dict, best_params)
            random_model = calculateBatmanModel.Spec( pars_dict, z, batpar, batmodel )
            ax[0].plot( z['bjd_tdb'], random_model, 'cyan', alpha=0.1)
            i += 1
    except TypeError:
        pass
    
    ax[0].plot( z['bjd_tdb'], best_model, 'blue', label='Batman model' )
    # if pr.QUADRATIC_BASELINE_SPEC:
    #     ax[0].plot( z['bjd_tdb'], np.zeros(len(z['bjd_tdb'])) + best_params['c0'] - best_params['c1']**2/4./best_params['c2'], 'magenta', label='Baseline' )
    ax[0].set_ylabel( 'Flux' )
    ax[0].set_title(z['detector'] + ', {:.3f} microns'.format(np.sum(z['wavEdgesMicr'])/2.))
    ax[0].legend(fontsize='small')
    
    ax[1].scatter ( z['bjd_tdb'], residuals, c='black', s=point_size )
    ax[1].errorbar( z['bjd_tdb'], residuals, yerr=z['lcUncs']/z['lcVals'] * unit, ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none')
    ax[1].plot( bin_middles, residuals_binned, color='red' )
    ax[1].set_xlabel( 'BJD_TDB' )
    ax[1].set_ylabel( 'Relative difference [%]' )
    
    if len(opath) > 0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
        print( '\nSaved:\n' + opath + '.pdf' )
    else:
        plt.show()
    
    plt.close()
    
def spectral_light_curve_partial_phasecurve( z, best_params, broad_fit_params, opath = '', emcee_result = None ):
    """
    Plots light curve data together with a model. The plot also shows the detector that
    measured the light curve as well as the wavelength range of the observation.
    """
    
    point_size = 2.
    errorbar_size = point_size/3.
    unit = 1.0E02       # percent
    alpha_errorbars = 0.6
    
    # Calculate the best fit model
    batpar, batmodel = calculateBatmanModel.create_Batman_object( z['bjd_tdb'] )
    batpar = calculateBatmanModel.fix_broad_fit_parameters( batpar, broad_fit_params )
    best_model = calculateBatmanModel.Spec( best_params, z, batpar, batmodel )
    residuals = (z['lcVals']-best_model)/z['lcVals'] * unit
    
    # Set the planetary emission to zero
    best_params['A'] = 0.
    no_emission = calculateBatmanModel.Spec( best_params, z, batpar, batmodel )
    
    # Perform binning
    bin_edges = np.linspace( min(z['bjd_tdb']), max(z['bjd_tdb']), 100 )
    residuals_binned, applied_bin_edges, indices = binned_statistic( z['bjd_tdb'], residuals, bins=bin_edges )
    bin_middles = Utils.bin_edges_into_middles( bin_edges )
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    
    fig, ax = plt.subplots( 2, 1, sharex=True )
    
    ax[0].scatter( z['bjd_tdb'], z['lcVals'], c='black', s=point_size, label='Data' )
    ax[0].errorbar( z['bjd_tdb'], z['lcVals'], yerr=z['lcUncs'], ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none')
    
    try:
        # Plot random draws from the fit params
        random_indices = np.random.randint( len( emcee_result['flat_chain'] ), size=100)
        i = 0
        for index in random_indices:
            parameters = emcee_result['flat_chain'][index]
            pars_dict = Utils.arrays2dict( parameters, emcee_result['parameters'] )
            pars_dict = Utils.merge_dictionaries(pars_dict, best_params)
            random_model = calculateBatmanModel.Spec( pars_dict, z, batpar, batmodel )
            ax[0].plot( z['bjd_tdb'], random_model, 'cyan', alpha=0.1)
            
            pars_dict['A'] = 0.
            random_no_emission = calculateBatmanModel.Spec( pars_dict, z, batpar, batmodel )
            ax[0].plot( z['bjd_tdb'], random_no_emission, 'magenta', alpha=0.1)
            i += 1
    except TypeError:
        pass
    
    ax[0].plot( z['bjd_tdb'], best_model, 'blue', label='Full model' )
    ax[0].plot(z['bjd_tdb'], no_emission, 'red', label='Transit only')
    # if pr.QUADRATIC_BASELINE_SPEC:
    #     ax[0].plot( z['bjd_tdb'], np.zeros(len(z['bjd_tdb'])) + best_params['c0'] - best_params['c1']**2/4./best_params['c2'], 'magenta', label='Baseline' )
    ax[0].set_ylabel( 'Flux' )
    ax[0].set_title(z['detector'] + ', {:.3f} microns'.format(np.sum(z['wavEdgesMicr'])/2.))
    ax[0].legend(fontsize='small')
    
    ax[1].scatter ( z['bjd_tdb'], residuals, c='black', s=point_size )
    ax[1].errorbar( z['bjd_tdb'], residuals, yerr=z['lcUncs']/z['lcVals'] * unit, ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none')
    ax[1].plot( bin_middles, residuals_binned, color='red' )
    ax[1].set_xlabel( 'BJD_TDB' )
    ax[1].set_ylabel( 'Relative difference [%]' )
    
    if len(opath) > 0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
        print( '\nSaved:\n' + opath + '.pdf' )
    else:
        plt.show()
    
    plt.close()

def spectral_light_curve_emission( z, best_params, broad_fit_params, opath, emcee_result = None, prelim_fit_params = None ):
    """
    Plots light curve data together with a model. The plot also shows the detector that
    measured the light curve as well as the wavelength range of the observation.
    """
    
    point_size = 2.
    errorbar_size = point_size/3.
    unit = 1.0E02       # percent
    alpha_errorbars = 0.6
    
    # Calculate the best fit model
    batpar, batmodel = calculateBatmanModel.create_Batman_object( z['bjd_tdb'] )
    batpar = calculateBatmanModel.fix_broad_fit_parameters( batpar, broad_fit_params )
    
    if pr.VARY_LD_PARAMETERS:
        if pr.KIPPING_LD_PARAMETERS:
            u1, u2 = Utils.LD_coefficients_q2u( best_params['q1'], best_params['q2'] )
            batpar.u = [ u1, u2 ]
        else:
            batpar.u = [best_params['u1'], best_params['u2']]
    else:
        # Setting the limb darkening parameters before calculating the batman model is necessary when LD is not fitted for
        if pr.KIPPING_LD_PARAMETERS:
            u1, u2 = Utils.LD_coefficients_q2u( prelim_fit_params['q1'], prelim_fit_params['q2'] )
            batpar.u = [ u1, u2 ]
        else:
            batpar.u = [prelim_fit_params['u1'], prelim_fit_params['u2']]
        
    best_model, baseline = calculateBatmanModel.Spec( best_params, z, batpar, batmodel )
    no_emission, baseline = calculateBatmanModel.Spec( best_params, z, batpar, batmodel, switch=False )
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    
    plt.scatter( z['bjd_tdb'], z['lcVals'], c='black', s=point_size, label='Data' )
    plt.errorbar( z['bjd_tdb'], z['lcVals'], yerr=z['lcUncs'], ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none')
    
    try:
        # Plot random draws from the fit params
        random_indices = np.random.randint( len( emcee_result['flat_chain'] ), size=100)
        i = 0
        for index in random_indices:
            parameters = emcee_result['flat_chain'][index]
            pars_dict = Utils.arrays2dict( parameters, emcee_result['parameters'] )
            random_model, random_baseline = calculateBatmanModel.Spec( pars_dict, z, batpar, batmodel )
            plt.plot( z['bjd_tdb'], random_model, 'cyan', alpha=0.1)
            i += 1
    except TypeError:
        pass
    
    plt.plot( z['bjd_tdb'], best_model, 'blue', label='Full model' )
    plt.plot( z['bjd_tdb'], no_emission, 'magenta', label='No emission' )
    # plt.plot( z['bjd_tdb'], baseline, 'red', label='Baseline' )
    plt.ylabel( 'Flux' )
    plt.title(z['detector'] + ', {:.3f} microns'.format(np.sum(z['wavEdgesMicr'])/2.))
    plt.legend(fontsize='small')
    
    if len(opath) > 0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
        print( '\nSaved:\n' + opath + '.pdf' )
    plt.show()
        
    plt.close()
    
    # plt.clf()
    # plt.scatter( z['bjd_tdb'], z['lcVals'], c='black', s=point_size, label='Data' )
    # plt.errorbar( z['bjd_tdb'], z['lcVals'], yerr=z['lcUncs'], ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none')
    # plt.plot( z['bjd_tdb'], best_model, 'blue', label='Model' )
    # plt.plot( z['bjd_tdb'], baseline, 'red', label='Baseline' )
    # plt.ylabel( 'Flux' )
    # plt.title(z['detector'] + ', {:.3f} microns'.format(np.sum(z['wavEdgesMicr'])/2.))
    # plt.legend(fontsize='small')
    # plt.xlabel( 'BJD_TDB' )
    # pylab.savefig( opath + '_nodiff.pdf', bbox_inches='tight' )
    # plt.close()
    

def spec_light_curve_correction( z, fp, fs, baseline, ofile ):
    """
    Plots light curves for both the NRS1 and the NRS2 detectors. If you pass it an emcee result dictionary, it will also plot a collection
    of models for both detectors.
    """
    
    point_size = 2.
    errorbar_size = point_size/3.
    alpha_errorbars = 0.6
    
    # Plot the best-fit light curve model against the data for both detectors:
    plt.clf()
    # Plot data and fp + B_plot
    plt.scatter( z['bjd_tdb'], z['lcVals'], c='black', s=point_size, label='Data' )
    plt.errorbar( z['bjd_tdb'], z['lcVals'], yerr=z['lcUncs'], ecolor='black', elinewidth=errorbar_size, alpha=alpha_errorbars, ls='none')
    plt.plot( z['bjd_tdb'], fp + fs, color='blue', label='Correction' )
    plt.plot(z['bjd_tdb'], baseline, color='red', label='Baseline' )
    plt.title(z['detector'] + ', {:.3f} microns'.format(np.sum(z['wavEdgesMicr'])/2.))
    plt.xlabel('BJD_TDB')
    plt.ylabel('Relative Flux')
    plt.legend()
    
    if len(ofile) > 0:
        opath = ofile + '.pdf'
        plt.savefig( opath, bbox_inches='tight' )
        print( '\nSaved:\n{0}'.format( opath ) )
    else:
        plt.show()
    
    plt.close()
    
def Spectra( wavelength, RpRs, uncertainties, model, opath ):
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    plt.scatter( wavelength, RpRs, c='black', s=4. )
    plt.errorbar( wavelength, RpRs, yerr=uncertainties, elinewidth=0.75, ecolor='black', ls='none')
    plt.plot( wavelength, model )
    plt.ylabel( 'Transit depth (R$_p$/R$_s$)$^2$' )
    plt.xlabel( 'Wavelength (micron)' )
    if (len(opath) > 0):
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    # plt.show()
    plt.close()

def Transmission_Spectrum_bin( trans_spec, n_bins=80, opath='' ):
    
    color_unbinned = 'blue'
    color_binned = 'black'
    alpha_unbinned = 0.8
    
    # Perform binning
    bin_edges = np.linspace( min(trans_spec['wavelength']['value']), max(trans_spec['wavelength']['value']), n_bins )
    RpRs, applied_bin_edges, indices = binned_statistic( trans_spec['wavelength']['value'], trans_spec['RpRs']['value'], bins=bin_edges )
    wavelength = Utils.bin_edges_into_middles(bin_edges)
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    plt.scatter( trans_spec['wavelength']['value'], trans_spec['RpRs']['value']**2, c=color_unbinned, s=4., alpha=alpha_unbinned )
    # plt.scatter( wavelength, RpRs**2, c=color_binned, zorder=9.9E9, s=4. )
    
    if 'upperr' in list(trans_spec['RpRs'].keys()):
        # Check, if the transmission spectrum has asymmetric errorbars stored
        percentile16 = trans_spec['RpRs']['value'] - trans_spec['RpRs']['lowerr']
        percentile84 = trans_spec['RpRs']['value'] + trans_spec['RpRs']['upperr']
        upper_error = percentile84**2 - trans_spec['RpRs']['value']**2
        lower_error = trans_spec['RpRs']['value']**2 -percentile16**2
        # upper_error = 2.*trans_spec['RpRs']['value']*trans_spec['RpRs']['upperr']
        # lower_error = 2.*trans_spec['RpRs']['value']*trans_spec['RpRs']['lowerr']
        asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
        plt.errorbar( trans_spec['wavelength']['value'], trans_spec['RpRs']['value']**2, yerr=asymmetric_error, elinewidth=0.75, ecolor=color_unbinned, alpha=alpha_unbinned, ls='none')
        
        # # Bin the errorbars
        # upper_error_binned = Utils.calculate_binned_errorbars( upper_error, indices, len(RpRs) )
        # lower_error_binned = Utils.calculate_binned_errorbars( lower_error, indices, len(RpRs) )
        # asymmetric_error = np.array(list(zip(lower_error_binned, upper_error_binned))).T
        # plt.errorbar( wavelength, RpRs**2, yerr=asymmetric_error, elinewidth=0.75, zorder=9.0E9, ecolor=color_binned, ls='none')
        
    else:
        # If not, just use symmetric errorbars
        # plt.errorbar( trans_spec['wavelength']['value'], trans_spec['RpRs']['value']**2, yerr=2.*trans_spec['RpRs']['value']*trans_spec['RpRs']['stderr'], elinewidth=0.75, ecolor=color_unbinned, alpha=alpha_unbinned, ls='none')
        
        # Bin the errorbars
        error_binned = Utils.calculate_binned_errorbars( 2.*trans_spec['RpRs']['value']*trans_spec['RpRs']['stderr'], indices, len(RpRs) )
        plt.errorbar( wavelength, RpRs**2, yerr=error_binned, elinewidth=0.75, ecolor=color_binned, ls='none')
        
    plt.xlabel( 'Wavelength (micron)' )
    plt.ylabel( 'Transit depth (R$_p$/R$_s$)$^2$' )
    if len(opath) > 0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' ) # dpi=300
    else:
        plt.show()
        
    plt.close()

def fit_variable( tr_spec, var, filename, n_bins=80 ):
    
    alpha_unbinned = 0.4
    color_unbinned = 'grey'
    color_binned = 'black'
    
    # Perform binning
    bin_edges = np.linspace( min(tr_spec['wavelength']['value']), max(tr_spec['wavelength']['value']), n_bins )
    binned_data, applied_bin_edges, indices = binned_statistic( tr_spec['wavelength']['value'], tr_spec[var]['value'], bins=bin_edges )
    wavelength = Utils.bin_edges_into_middles(bin_edges)
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    plt.scatter( tr_spec['wavelength']['value'], tr_spec[var]['value'], c=color_unbinned, alpha=alpha_unbinned, s=4. )
    plt.scatter( wavelength, binned_data, c=color_binned, s=4., zorder=9.9E9 )
    
    if 'upperr' in list(tr_spec[var].keys()):
        # Check, if the transmission spectrum has asymmetric errorbars stored
        asymmetric_error = np.array(list(zip(tr_spec[var]['lowerr'], tr_spec[var]['upperr']))).T
        plt.errorbar( tr_spec['wavelength']['value'], tr_spec[var]['value'], yerr=asymmetric_error, elinewidth=0.75, ecolor=color_unbinned, alpha=alpha_unbinned, ls='none')
        
        # Bin the errorbars
        upper_error_binned = Utils.calculate_binned_errorbars( tr_spec[var]['upperr'], indices, len(binned_data) )
        lower_error_binned = Utils.calculate_binned_errorbars( tr_spec[var]['lowerr'], indices, len(binned_data) )
        asymmetric_error = np.array(list(zip(lower_error_binned, upper_error_binned))).T
        plt.errorbar( wavelength, binned_data, yerr=asymmetric_error, elinewidth=0.75, ecolor=color_binned, zorder=9.0E9, ls='none')
        
    else:
        # If not, just use symmetric errorbars
        plt.errorbar( tr_spec['wavelength']['value'], tr_spec[var]['value'], yerr=tr_spec[var]['stderr'], elinewidth=0.75, ecolor=color_unbinned, alpha=alpha_unbinned, ls='none')
        
        error_binned = Utils.calculate_binned_errorbars(tr_spec[var]['stderr'], indices, len(binned_data))
        plt.errorbar( wavelength, binned_data, yerr=error_binned, elinewidth=0.75, ecolor=color_binned, zorder=9.0E9, ls='none')
        
    if var=='Tmid':
        plt.ylabel( 'Transit mid time [BJD]' )
    else:
        plt.ylabel( var )
    
    if var=='q2':
        plt.ylim([-0.1, 1.1])
    
    plt.xlabel( 'Wavelength (micron)' )
    pylab.savefig( filename + '_' + var + '.pdf', bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()

def Transmission_Spectrum_withmodel_bin( tr_spec, model, filename, n_bins=250 ):
    """
    Plots a transmission spectrum together with a model into shared bins with n_bins bins.
    """
    
    bin_edges = np.linspace( min(model['SiO']['wavelength']), max(model['SiO']['wavelength']), n_bins )
    
    # Bin the JWST data
    RpRs, applied_bin_edges, indices = binned_statistic( tr_spec['wavelength']['value'], tr_spec['RpRs']['value'], bins=bin_edges )
    RpRs_unc, applied_bin_edges, indices = binned_statistic(tr_spec['wavelength']['value'], tr_spec['RpRs']['stderr'], bins=bin_edges)
    
    wavelength = Utils.bin_edges_into_middles(bin_edges)
    
    # Define plot colors
    colors = {}
    colors['SiO'] = 'red'
    colors['noSiO'] = 'blue'
    
    # Clear plot memory
    plt.clf()
    plt.scatter( wavelength, RpRs**2, c='black', s=6., label='JWST' )
    plt.errorbar( wavelength, RpRs**2, yerr=2.*RpRs*RpRs_unc, elinewidth=1., ecolor='black', ls='none')
    for k in model.keys():
        model[k]['RpRs'], applied_bin_edges, indices = binned_statistic( model[k]['wavelength'], model[k]['flux'], bins=bin_edges )
        plt.plot( wavelength, model[k]['RpRs'], color=colors[k], label=k )
    plt.xlim( 2.65, 5.25 )
    plt.ylabel( 'Transit depth (R$_p$/R$_s$)$^2$' )
    plt.xlabel( 'Wavelength (micron)' )
    plt.legend()
    pylab.savefig( filename + '_model.pdf', bbox_inches='tight' ) # dpi=300
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()

def Transmission_Spectrum_chi( tr_spec, filepath ):
    
    point_size = 2.
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    
    fig, ax = plt.subplots( 2, 1, sharex=True )
    
    ax[0].scatter( tr_spec['wavelength']['value'], tr_spec['RpRs']['value']**2, c='black', s=point_size )
    ax[0].errorbar( tr_spec['wavelength']['value'], tr_spec['RpRs']['value']**2, yerr=2.*tr_spec['RpRs']['value']*tr_spec['RpRs']['stderr'], elinewidth = point_size/3., ecolor='black', ls='none')
    ax[0].set_ylabel( '(R$_p$/R$_s$)$^2$' )
    
    ax[1].scatter( tr_spec['wavelength']['value'], np.sqrt(tr_spec['chisquare']['value']), c='black', s=point_size )
    ax[1].set_xlabel( 'Wavelength (micron)' )
    #ax[1].set_ylabel( '$\chi^2$' )
    ax[1].set_ylabel( '$\chi$' )
    
    pylab.savefig( filepath + '_chi.pdf', bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()
    
def all_fit_parameters( tr_spec, parameters=[], x_limits=[2.7,5.2], fig_size=[9,10.3], outliers=None, opath='', LDcoefficients = None, white_fit = None, white_chisquare=None ):
    
    data_color = 'black'
    point_size = 5.
    errorbar_size = 0.5
    alpha_errorbars = 1.0
    alpha_shading = 1
    parameters = list(tr_spec.keys())
    prior_color = 'red'
    white_color = 'blue'
    
    n_lines = 5     #3
    n_columns = 2   #2
    
    # Prepare transit mid-time calculation
    try:
        i1 = np.where( white_fit['Parameter'] == 'Tmid_NRS1' )[0][0]
        i2 = np.where( white_fit['Parameter'] == 'Tmid_NRS2' )[0][0]
        T_mean = (np.float(white_fit['Median'][i1]) + np.float(white_fit['Median'][i2])) / 2.
    except TypeError:
        T_mean = np.median(tr_spec['Tmid']['value'])
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    # print(parameters)
    fig, ax = plt.subplots( len(parameters)//2, 2, figsize=fig_size, sharex=True )
    # fig, ax = plt.subplots( n_lines, n_columns, figsize=fig_size, sharex=True )
    
    i = 0
    if len(parameters) == 0:
        parameters = list(tr_spec.keys())
        
    for var in parameters:
        i_plot = int(i/n_columns)
        j_plot = i%n_columns
        if ( not( var in ['wavelength', 'wavelength_halfwidth', 'chisquare', 'bic'] ) ):
            # Plot theoretical values for LDCs
            if (var in ['u1', 'u2']):
                try:
                    ax[i_plot,j_plot].plot( tr_spec['wavelength'], LDcoefficients[var], color=prior_color, zorder=2. )
                    ax[i_plot,j_plot].fill_between( tr_spec['wavelength'], LDcoefficients[var] - LDcoefficients[var + '_err'], LDcoefficients[var] + LDcoefficients[var + '_err'], color=prior_color, alpha=alpha_shading )
                except Exception:
                    pass
            
            # Posteriors
            try:
                # Check, if the transmission spectrum has asymmetric errorbars stored
                asymmetric_error = np.array(list(zip(tr_spec[var]['lowerr'], tr_spec[var]['upperr']))).T
                # ax[i].errorbar( tr_spec['wavelength'], tr_spec[var]['value'], yerr=asymmetric_error, elinewidth=errorbar_size, ecolor='blue', alpha=alpha_errorbars, ls='none')
                # ax[i].errorbar( tr_spec['wavelength']['value'], tr_spec[var]['value'], yerr=asymmetric_error, elinewidth=errorbar_size, ecolor=data_color, alpha=alpha_errorbars, ls='none', zorder=3.)
                if var == 'Tmid':
                    ax[i_plot,j_plot].errorbar( tr_spec['wavelength'], (tr_spec[var]['value'] - T_mean) * 24. * 3600., yerr=asymmetric_error * 24. * 3600., elinewidth=errorbar_size, ecolor=data_color, alpha=alpha_errorbars, ls='none', zorder=3.)
                else:
                    ax[i_plot,j_plot].errorbar( tr_spec['wavelength'], tr_spec[var]['value'], yerr=asymmetric_error, elinewidth=errorbar_size, ecolor=data_color, alpha=alpha_errorbars, ls='none', zorder=3.)
                # ax[i].set_ylabel( Utils.pretty_labels(var) )
                ax[i_plot,j_plot].set_ylabel( Utils.pretty_labels(var) )
            except KeyError:
                # If not, just use symmetric errorbars
                try:
                    # ax[i].errorbar( tr_spec['wavelength'], tr_spec[var]['value'], yerr=tr_spec[var]['stderr'], elinewidth=errorbar_size, alpha=alpha_errorbars, ecolor='blue', ls='none')
                    # ax[i].errorbar( tr_spec['wavelength']['value'], tr_spec[var]['value'], yerr=tr_spec[var]['stderr'], elinewidth=errorbar_size, alpha=alpha_errorbars, ecolor=data_color, ls='none', zorder=3.)
                    if var == 'Tmid':
                        ax[i_plot,j_plot].errorbar( tr_spec['wavelength'], (tr_spec[var]['value'] - T_mean) * 24. * 3600., yerr=tr_spec[var]['stderr'] * 24. * 3600., elinewidth=errorbar_size, alpha=alpha_errorbars, ecolor=data_color, ls='none', zorder=3.)
                    else:
                        ax[i_plot,j_plot].errorbar( tr_spec['wavelength'], tr_spec[var]['value'], yerr=tr_spec[var]['stderr'], elinewidth=errorbar_size, alpha=alpha_errorbars, ecolor=data_color, ls='none', zorder=3.)
                    if ( max( tr_spec[var]['stderr'] > 0.) or var=='outliers' ):
                        # ax[i].set_ylabel( var )
                        ax[i_plot,j_plot].set_ylabel( var )
                        if ( var == 'q2' ):
                            ax[i].set_ylim(-0.1, 1.1)
                            ax[i_plot,j_plot].set_ylim(-0.1, 1.1)
                        # if ( var == 'outliers' ):
                        #     ax[i].set_ylim(-1., 10.)
                    else:
                        # ax[i].set_ylabel( var + ' (fixed)' )
                        ax[i_plot,j_plot].set_ylabel( var + ' (fixed)' )
                except KeyError:
                    # ax[i].set_ylabel( Utils.pretty_labels(var) )
                    ax[i_plot,j_plot].set_ylabel( Utils.pretty_labels(var) )
            
            if var == 'Tmid':
                ax[i_plot,j_plot].scatter( tr_spec['wavelength'], (tr_spec[var]['value'] - T_mean) * 24. * 3600., c='white', edgecolor=data_color, s=point_size, zorder=4., label='Posterior medians' )
            else:
                ax[i_plot,j_plot].scatter( tr_spec['wavelength'], tr_spec[var]['value'], c='white', edgecolor=data_color, s=point_size, zorder=4., label='Posterior medians' )
            
            # Plot the white light curve results
            # if var in ['RpRs', 'Tmid', 'u1', 'u2', 'c0', 'c1', 'c2']:
            try:
                for nrs in ['NRS1', 'NRS2']:
                    i_white = np.where( white_fit['Parameter'] == var + '_' + nrs )[0][0]
                    median = np.float(white_fit['Median'][i_white])
                    low = median - np.float(white_fit['1 sigma lower error '][i_white])
                    up = median + np.float(white_fit['1 sigma upper error'][i_white])
                    if var == 'Tmid':
                        median = (median - T_mean) * 24. * 3600.
                        low = (low - T_mean) * 24. * 3600.
                        up = (up - T_mean) * 24. * 3600.
                    elif var == 'sigma_sys':
                        median *= 1.0E6
                        low *= 1.0E6
                        up *= 1.0E6
                    # ax[i].fill_between( [lower_wavelengths[nrs], upper_wavelengths[nrs]], [low, low], [up, up], color=white_color, alpha=alpha_shading )
                    # ax[i].plot( [lower_wavelengths[nrs], upper_wavelengths[nrs]], [median, median], '--', color=white_color, zorder=3. )
                    ax[i_plot,j_plot].fill_between( [lower_wavelengths[nrs], upper_wavelengths[nrs]], [low, low], [up, up], color=white_color, alpha=alpha_shading )
                    ax[i_plot,j_plot].plot( [lower_wavelengths[nrs], upper_wavelengths[nrs]], [median, median], '--', color=white_color, zorder=3. )
                    
                ax[i_plot,j_plot].plot( [lower_wavelengths['NRS1'], lower_wavelengths['NRS1']], [tr_spec[var]['value'][0], tr_spec[var]['value'][0]], '--', color=white_color, zorder=1., label='White light curve posterior medians' )
            except TypeError:
                pass
            
            # ax[i].yaxis.set_minor_locator(AutoMinorLocator())
            ax[i_plot,j_plot].yaxis.set_minor_locator(AutoMinorLocator())
            i += 1
    
    i_plot = int(i/n_columns)
    j_plot = i%n_columns
    try:
        ax[i_plot,j_plot].scatter( tr_spec['wavelength'], tr_spec['chisquare'], c='white', edgecolor=data_color, s=point_size, zorder=4., label='Posterior medians' )
        ax[-1,0].set_ylabel( '$\chi_\\nu^2$' )
    except Exception:
        pass
    # ax[-1,0].plot( [lower_wavelengths[nrs], upper_wavelengths[nrs]], [white_chisquare, white_chisquare], '--', color=white_color, zorder=3. )
    
    try:
        ax[-1,-1].bar( tr_spec['wavelength'], outliers, width = tr_spec['wavelength_halfwidth'], color='white', edgecolor=data_color )
        ax[-1,-1].yaxis.set_major_locator(MultipleLocator(1))
        ax[-1,-1].set_ylabel( 'Number of sigma clipped outliers' )
    except TypeError:
        pass
    
    # ax[-2,-1].plot( [lower_wavelengths['NRS1'], lower_wavelengths['NRS1']], [tr_spec[var]['value'][0], tr_spec[var]['value'][0]], color=prior_color, zorder=1., label='Prior medians' )
    ax[-2,-1].legend()
    
    ax[0,0].set_xlim(x_limits)
    ax[0,0].xaxis.set_minor_locator(AutoMinorLocator())
    ax[-1,0].set_xlabel('Wavelength [$\mu$m]')
    ax[-1,1].set_xlabel('Wavelength [$\mu$m]')
    fig.tight_layout()
    if len(opath)>0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    # else:
    plt.show()
    plt.close()
    
def fixLD_freeLD_Transmission_Spectra( tr_spec_fix, tr_spec_free, filepath ):
    
    point_size = 2.
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    
    fig, ax = plt.subplots( 2, 1, sharex=True )
    
    ax[0].scatter( tr_spec_fix['wavelength']['value'], tr_spec_fix['RpRs']['value']**2, c='black', s=point_size, label='Fixed LD' )
    ax[0].errorbar( tr_spec_fix['wavelength']['value'], tr_spec_fix['RpRs']['value']**2, yerr=2.*tr_spec_fix['RpRs']['value']*tr_spec_fix['RpRs']['stderr'], elinewidth = point_size/3., ecolor='black', ls='none')
    ax[0].scatter( tr_spec_free['wavelength']['value'], tr_spec_free['RpRs']['value']**2, c='blue', s=point_size, label='Free LD' )
    ax[0].errorbar( tr_spec_free['wavelength']['value'], tr_spec_free['RpRs']['value']**2, yerr=2.*tr_spec_free['RpRs']['value']*tr_spec_free['RpRs']['stderr'], elinewidth = point_size/3., ecolor='blue', ls='none')
    ax[0].set_ylabel( '(R$_p$/R$_s$)$^2$' )
    ax[0].legend()
    
    ax[1].scatter( tr_spec_free['wavelength']['value'], 1.0E06 * (tr_spec_fix['RpRs']['value']**2 - tr_spec_free['RpRs']['value']**2), c='black', s=point_size )
    ax[1].errorbar( tr_spec_free['wavelength']['value'], 1.0E06 * (tr_spec_fix['RpRs']['value']**2 - tr_spec_free['RpRs']['value']**2), yerr=1.0E06 * 2. *tr_spec_free['RpRs']['value']*tr_spec_free['RpRs']['stderr'], elinewidth = point_size/3., ecolor='black', ls='none')
    ax[1].set_xlabel( 'Wavelength (micron)' )
    ax[1].set_ylabel( 'Absolute difference [ppm]' )
    
    pylab.savefig( filepath + '_fixVSfree.pdf', bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()


def Transmission_Spectrum_nruns( tr_spec, transit_depth_unit=1., limits=None, title=None, opath='' ):
    
    point_size = 8.
    alpha = 0.8
    colors = ['blue', 'red', 'black']
    markers = ['d', 'o', 's']
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    keys = list(tr_spec.keys())
    
    for i, run in enumerate(keys):
        if 'transit_depth' in list(tr_spec[run].keys()):
            if 'upperr' in list(tr_spec[run]['transit_depth'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                asymmetric_error = np.array(list(zip(tr_spec[run]['transit_depth']['lowerr'], tr_spec[run]['transit_depth']['upperr']))).T
                plt.errorbar( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
            else:
                # If not, just use symmetric errorbars
                plt.errorbar( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, yerr=tr_spec[run]['transit_depth']['stderr'] / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
            plt.scatter( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, c='white', edgecolor=colors[i], marker=markers[i], s=point_size, alpha=alpha, label=run, zorder=2 )
        else:
            if 'upperr' in list(tr_spec[run]['RpRs'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                percentile16 = tr_spec[run]['RpRs']['value'] - tr_spec[run]['RpRs']['lowerr']
                percentile84 = tr_spec[run]['RpRs']['value'] + tr_spec[run]['RpRs']['upperr']
                upper_error = percentile84**2 - tr_spec[run]['RpRs']['value']**2
                lower_error = tr_spec[run]['RpRs']['value']**2 -percentile16**2
                asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
                plt.errorbar( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
            else:
                # If not, just use symmetric errorbars
                plt.errorbar( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, yerr=2.*tr_spec[run]['RpRs']['value']*tr_spec[run]['RpRs']['stderr'] / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
            plt.scatter( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, c='white', edgecolor=colors[i], marker=markers[i], s=point_size, alpha=alpha, label=run, zorder=2 )
    
    if transit_depth_unit == 1.0E-6:
        plt.ylabel( 'Transit depth [ppm]' )
    elif transit_depth_unit == 1.0E-2:
        plt.ylabel( 'Transit depth [\%]' )
    else:
        plt.ylabel( 'Transit depth' )
    plt.ylim( limits )
    plt.xlabel( 'Wavelength [micron]' )
    plt.title( title )
    plt.legend()
    if ( len(opath) > 0):
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    else:
        plt.show()
    plt.close()

def Transmission_Spectrum_nruns_meansubtracted( tr_spec, transit_depth_unit=1., limits=None, title=None, opath='' ):
    
    point_size = 8.
    alpha = 0.8
    colors = ['blue', 'red', 'black']
    markers = ['d', 'o', 's']
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    keys = list(tr_spec.keys())
    
    for i, run in enumerate(keys):
        if 'transit_depth' in list(tr_spec[run].keys()):
            if 'upperr' in list(tr_spec[run]['transit_depth'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                asymmetric_error = np.array(list(zip(tr_spec[run]['transit_depth']['lowerr'], tr_spec[run]['transit_depth']['upperr']))).T
                plt.errorbar( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
            else:
                # If not, just use symmetric errorbars
                plt.errorbar( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, yerr=tr_spec[run]['transit_depth']['stderr'] / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
            plt.scatter( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, c='white', edgecolor=colors[i], marker=markers[i], s=point_size, alpha=alpha, label=run, zorder=2 )
        else:
            mean = np.mean( tr_spec[run]['RpRs']['value'][:7]**2 )
            if 'upperr' in list(tr_spec[run]['RpRs'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                percentile16 = tr_spec[run]['RpRs']['value'] - tr_spec[run]['RpRs']['lowerr']
                percentile84 = tr_spec[run]['RpRs']['value'] + tr_spec[run]['RpRs']['upperr']
                upper_error = percentile84**2 - tr_spec[run]['RpRs']['value']**2
                lower_error = tr_spec[run]['RpRs']['value']**2 -percentile16**2
                asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
                plt.errorbar( tr_spec[run]['wavelength'], (tr_spec[run]['RpRs']['value']**2 - mean) / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
            else:
                # If not, just use symmetric errorbars
                plt.errorbar( tr_spec[run]['wavelength'], (tr_spec[run]['RpRs']['value']**2 - mean) / transit_depth_unit, xerr=tr_spec[run]['wavelength_halfwidth'], yerr=2.*tr_spec[run]['RpRs']['value']*tr_spec[run]['RpRs']['stderr'] / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
            plt.scatter( tr_spec[run]['wavelength'], (tr_spec[run]['RpRs']['value']**2 - mean) / transit_depth_unit, c='white', edgecolor=colors[i], marker=markers[i], s=point_size, alpha=alpha, label=run, zorder=2 )
        
    if transit_depth_unit == 1.0E-6:
        plt.ylabel( 'Transit depth - offset [ppm]' )
    elif transit_depth_unit == 1.0E-2:
        plt.ylabel( 'Transit depth - offset [\%]' )
    else:
        plt.ylabel( 'Transit depth - offset' )
    plt.xlim( [2.7,5.1] )
    plt.ylim( limits )
    plt.xlabel( 'Wavelength [micron]' )
    plt.title( title )
    plt.legend()
    if ( len(opath) > 0):
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    else:
        plt.show()
    plt.close()

def Transmission_Spectrum_nruns_meansubtracted_npanels( tr_spec, transit_depth_unit=1., limits=None, title=None, opath='' ):
    
    keys = list(tr_spec.keys())
    point_size = 8.
    alpha = 0.8
    colors = ['black', 'black', 'black']
    markers = ['o', 'o', 'o']
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    fig, ax = plt.subplots( 1, len(keys), figsize=[5,2], sharex=True, sharey=True )
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    for i, run in enumerate(keys):
        if 'transit_depth' in list(tr_spec[run].keys()):
            if 'upperr' in list(tr_spec[run]['transit_depth'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                asymmetric_error = np.array(list(zip(tr_spec[run]['transit_depth']['lowerr'], tr_spec[run]['transit_depth']['upperr']))).T
                ax[i].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
            else:
                # If not, just use symmetric errorbars
                ax[i].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, yerr=tr_spec[run]['transit_depth']['stderr'] / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
            ax[i].scatter( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, c='white', edgecolor=colors[i], marker=markers[i], s=point_size, alpha=alpha, label=run, zorder=2 )
        else:
            # mean = np.mean( tr_spec[run]['RpRs']['value'][:7]**2 )
            mean = tr_spec[run]['RpRs']['value'][6]**2
            if 'upperr' in list(tr_spec[run]['RpRs'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                percentile16 = tr_spec[run]['RpRs']['value'] - tr_spec[run]['RpRs']['lowerr']
                percentile84 = tr_spec[run]['RpRs']['value'] + tr_spec[run]['RpRs']['upperr']
                upper_error = percentile84**2 - tr_spec[run]['RpRs']['value']**2
                lower_error = tr_spec[run]['RpRs']['value']**2 -percentile16**2
                asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
                ax[i].errorbar( tr_spec[run]['wavelength'], (tr_spec[run]['RpRs']['value']**2 - mean) / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
            else:
                # If not, just use symmetric errorbars
                ax[i].errorbar( tr_spec[run]['wavelength'], (tr_spec[run]['RpRs']['value']**2 - mean) / transit_depth_unit, xerr=tr_spec[run]['wavelength_halfwidth'], yerr=2.*tr_spec[run]['RpRs']['value']*tr_spec[run]['RpRs']['stderr'] / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
            ax[i].scatter( tr_spec[run]['wavelength'], (tr_spec[run]['RpRs']['value']**2 - mean) / transit_depth_unit, c='white', edgecolor=colors[i], marker=markers[i], s=point_size, label='Data', zorder=2 )
        
        ax[i].fill_between( tr_spec[run]['wavelength'][:7], (tr_spec[run]['RpRs']['value']**2 - mean)[6] / transit_depth_unit, (tr_spec[run]['RpRs']['value']**2 - mean)[:7] / transit_depth_unit, color='blue', alpha=0.3, label='H$_2$O')
        ax[i].fill_between( tr_spec[run]['wavelength'][6:9], (tr_spec[run]['RpRs']['value']**2 - mean)[6] / transit_depth_unit, (tr_spec[run]['RpRs']['value']**2 - mean)[6:9] / transit_depth_unit, color='green', alpha=0.3, label='SiO')
        ax[i].fill_between( tr_spec[run]['wavelength'][8:], (tr_spec[run]['RpRs']['value']**2 - mean)[6] / transit_depth_unit, (tr_spec[run]['RpRs']['value']**2 - mean)[8:] / transit_depth_unit, color='orange', alpha=0.3, label='CO')
        ax[i].set_xlabel( 'Wavelength [$\mu$m]' )
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
        # ax[i].set_title( run )
    if transit_depth_unit == 1.0E-6:
        ax[0].set_ylabel( 'Transit depth - offset [ppm]' )
    elif transit_depth_unit == 1.0E-2:
        ax[0].set_ylabel( 'Transit depth - offset [\%]' )
    else:
        ax[0].set_ylabel( 'Transit depth - offset' )
    ax[0].legend( loc = 'upper left', fontsize='small' )
    ax[0].set_xlim( [2.7,5.1] )
    ax[0].set_ylim( limits )
    # fig.tight_layout()
    if ( len(opath) > 0):
        pylab.savefig( opath + '.png', dpi=600, bbox_inches='tight' )
    plt.show()
    plt.close()

def Transmission_Spectrum_nruns_errorbars( tr_spec, transit_depth_unit=1., transit_depth_limits=None, uncertainty_limits=None, title=None, opath='' ):
    
    point_size = 8.
    alpha = 0.8
    errorbar_unit = 1.0E-4
    colors = ['blue', 'red', 'black']
    markers = ['d', 'o', 's']
    
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    fig, ax = plt.subplots( 2, 1, figsize=[6,6], sharex=True, height_ratios=[2,1] )
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    keys = list(tr_spec.keys())
    
    for i, run in enumerate(keys):
        if 'transit_depth' in list(tr_spec[run].keys()):
            if 'upperr' in list(tr_spec[run]['transit_depth'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                asymmetric_error = np.array(list(zip(tr_spec[run]['transit_depth']['lowerr'], tr_spec[run]['transit_depth']['upperr']))).T
                ax[0].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
                ax[1].scatter( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['upperr']/errorbar_unit, marker=markers[i], s=point_size, c=colors[i], alpha=1, label=run)
            else:
                # If not, just use symmetric errorbars
                ax[0].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, yerr=tr_spec[run]['transit_depth']['stderr'] / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
                ax[1].scatter( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['stderr']/errorbar_unit, marker=markers[i], s=point_size, c=colors[i], alpha=1, label=run)
            ax[0].scatter( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, c='white', edgecolor=colors[i], marker=markers[i], s=point_size, alpha=alpha, label=run, zorder=2 )
        else:
            if 'upperr' in list(tr_spec[run]['RpRs'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                percentile16 = tr_spec[run]['RpRs']['value'] - tr_spec[run]['RpRs']['lowerr']
                percentile84 = tr_spec[run]['RpRs']['value'] + tr_spec[run]['RpRs']['upperr']
                upper_error = percentile84**2 - tr_spec[run]['RpRs']['value']**2
                lower_error = tr_spec[run]['RpRs']['value']**2 -percentile16**2
                asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
                ax[0].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
                ax[1].scatter( tr_spec[run]['wavelength'], upper_error/errorbar_unit, marker=markers[i], s=point_size, c=colors[i], alpha=1, label=run)
            else:
                # If not, just use symmetric errorbars
                ax[0].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, yerr=2.*tr_spec[run]['RpRs']['value']*tr_spec[run]['RpRs']['stderr'] / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
                ax[1].scatter( tr_spec[run]['wavelength'], 2.*tr_spec[run]['RpRs']['value']*tr_spec[run]['RpRs']['stderr']/errorbar_unit, marker=markers[i], s=point_size, c=colors[i], alpha=1, label=run)
            ax[0].scatter( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, c='white', edgecolor=colors[i], marker=markers[i], s=point_size, alpha=alpha, label=run, zorder=2 )
    
    if transit_depth_unit == 1.0E-6:
        ax[0].set_ylabel( 'Transit depth [ppm]' )
    elif transit_depth_unit == 1.0E-2:
        ax[0].set_ylabel( 'Transit depth [\%]' )
    else:
        ax[0].set_ylabel( 'Transit depth' )
    ax[0].set_ylim( transit_depth_limits )
    ax[1].set_ylim( uncertainty_limits )
    ax[-1].set_xlabel( 'Wavelength [micron]' )
    ax[1].set_ylabel( '$1\sigma$ uncertainty [ppm]' )
    ax[0].set_title( title )
    ax[0].legend()
    ax[0].yaxis.set_minor_locator(AutoMinorLocator())
    ax[1].yaxis.set_minor_locator(AutoMinorLocator())
    if ( len(opath) > 0):
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    else:
        plt.show()
    plt.close()


def Transmission_Spectrum_2runs_errorbars_residuals( tr_spec, transit_depth_unit=1., transit_depth_limits=None, uncertainty_unit=1., uncertainty_limits=None, residual_limits=None, title=None, opath='' ):
    
    point_size = 8.
    alpha = 0.8
    colors = ['blue', 'red', 'black']
    markers = ['d', 'o', 's']
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    fig, ax = plt.subplots( 3, 1, figsize=[6,8], sharex=True, height_ratios=[2,1,1] )
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    keys = list(tr_spec.keys())
    
    for i, run in enumerate(keys):
        if 'transit_depth' in list(tr_spec[run].keys()):
            if 'upperr' in list(tr_spec[run]['transit_depth'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                asymmetric_error = np.array(list(zip(tr_spec[run]['transit_depth']['lowerr'], tr_spec[run]['transit_depth']['upperr']))).T
                ax[0].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
                ax[1].scatter( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['upperr']/uncertainty_unit, marker=markers[i], s=point_size, c=colors[i], alpha=1, label=run)
            else:
                # If not, just use symmetric errorbars
                ax[0].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, yerr=tr_spec[run]['transit_depth']['stderr'] / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
                ax[1].scatter( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['stderr']/uncertainty_unit, marker=markers[i], s=point_size, c=colors[i], alpha=1, label=run)
            ax[0].scatter( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, c='white', edgecolor=colors[i], marker=markers[i], s=point_size, alpha=alpha, label=run, zorder=2 )
        else:
            if 'upperr' in list(tr_spec[run]['RpRs'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                percentile16 = tr_spec[run]['RpRs']['value'] - tr_spec[run]['RpRs']['lowerr']
                percentile84 = tr_spec[run]['RpRs']['value'] + tr_spec[run]['RpRs']['upperr']
                upper_error = percentile84**2 - tr_spec[run]['RpRs']['value']**2
                lower_error = tr_spec[run]['RpRs']['value']**2 -percentile16**2
                asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
                ax[0].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
                ax[1].scatter( tr_spec[run]['wavelength'], upper_error/uncertainty_unit, marker=markers[i], s=point_size, c=colors[i], alpha=1, label=run)
            else:
                # If not, just use symmetric errorbars
                ax[0].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, yerr=2.*tr_spec[run]['RpRs']['value']*tr_spec[run]['RpRs']['stderr'] / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha, zorder=1)
                ax[1].scatter( tr_spec[run]['wavelength'], 2.*tr_spec[run]['RpRs']['value']*tr_spec[run]['RpRs']['stderr']/uncertainty_unit, marker=markers[i], s=point_size, c=colors[i], alpha=1, label=run)
            ax[0].scatter( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, c='white', edgecolor=colors[i], marker=markers[i], s=point_size, alpha=alpha, label=run, zorder=2 )
    
    # Residuals
    ax[2].scatter( tr_spec[keys[0]]['wavelength'], (tr_spec[keys[1]]['RpRs']['value']**2 - tr_spec[keys[0]]['RpRs']['value']**2)/uncertainty_unit, c='white', edgecolor='purple', s=point_size )
    
    if transit_depth_unit == 1.0E-6:
        ax[0].set_ylabel( 'Transit depth [ppm]' )
    elif transit_depth_unit == 1.0E-2:
        ax[0].set_ylabel( 'Transit depth [\%]' )
    else:
        ax[0].set_ylabel( 'Transit depth' )
    
    if uncertainty_unit == 1.0E-6:
        ax[1].set_ylabel( '$1\sigma$ uncertainty [ppm]' )
        ax[2].set_ylabel( 'Residuals [ppm]' )
    else:
        ax[1].set_ylabel( '$1\sigma$ uncertainty' )
        ax[2].set_ylabel( 'Residuals' )
    
    ax[0].set_ylim( transit_depth_limits )
    ax[1].set_ylim( uncertainty_limits )
    ax[2].set_ylim( residual_limits )
    ax[-1].set_xlabel( 'Wavelength [micron]' )
    ax[0].set_title( title )
    ax[0].legend()
    ax[0].yaxis.set_minor_locator(AutoMinorLocator())
    ax[1].yaxis.set_minor_locator(AutoMinorLocator())
    ax[2].yaxis.set_minor_locator(AutoMinorLocator())
    if ( len(opath) > 0):
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()

def planet_rotation( tr_spec, x_range = [2.7, 5.2], opath = '' ):
    
    point_size = 20.
    errorbar_size = 1.
    color = 'red'
    shading_alpha = 0.3
    
    plt.clf()
    if 'R2' in tr_spec.keys():
        fig, ax = plt.subplots( 3, 1, figsize=[6,9], sharex=True )
        ax[2].set_ylabel('$R_2/R_*$ [day$^{-2}$]')  
    else:
        fig, ax = plt.subplots( 2, 1, figsize=[6,6], sharex=True )
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    for i in range(len(ax)):
        if 'upperr' in list(tr_spec['R' + str(i)].keys()):
            # Check, if the transmission spectrum has asymmetric errorbars stored
            asymmetric_error = np.array(list(zip(tr_spec['R' + str(i)]['lowerr'], tr_spec['R' + str(i)]['upperr']))).T
            ax[i].errorbar( tr_spec['wavelength'], tr_spec['R' + str(i)]['value'], xerr=tr_spec['wavelength_halfwidth'], yerr=asymmetric_error, color=color, elinewidth=errorbar_size, ls='none', zorder=2)
        else:
            # If not, just use symmetric errorbars
            ax[i].errorbar( tr_spec['wavelength'], tr_spec['R' + str(i)]['value'], xerr=tr_spec['wavelength_halfwidth'], yerr=tr_spec['R' + str(i)]['stderr'], color=color, elinewidth=errorbar_size, ls='none', zorder=2 )
        ax[i].scatter( tr_spec['wavelength'], tr_spec['R' + str(i)]['value'], c='white', edgecolor=color, s=point_size, zorder=3 )
        
        ax[i].fill_betweenx( [-10.,10.], np.zeros(2) + x_range[0], np.zeros(2) + 4.0, color='blue', alpha=shading_alpha )
        ax[i].fill_betweenx( [-10.,10.], np.zeros(2) + 4.0, np.zeros(2) + 4.3, color='green', alpha=shading_alpha )
        ax[i].fill_betweenx( [-10.,10.], np.zeros(2) + 4.3, np.zeros(2) + x_range[1], color='yellow', alpha=shading_alpha )
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
        
        y_min = np.min(tr_spec['R' + str(i)]['value'])
        y_max = np.max(tr_spec['R' + str(i)]['value'])
        if i==0:
            y_step = (y_max - y_min)/10.
        elif i==1:
            y_step = (y_max - y_min)/3.5
        else:
            y_step = (y_max - y_min)/2.
        ax[i].set_ylim( y_min-y_step, y_max+y_step )
    
    ax[0].set_xlim( x_range )
    
    # ax[1].set_ylim( -0.005, 0.027 )
    ax[0].set_ylabel('$R_0/R_*$') 
    ax[1].set_ylabel('$R_1/R_*$ [day$^{-1}$]')  
    ax[-1].set_xlabel('Wavelength [micron]')
    ax[0].xaxis.set_minor_locator(AutoMinorLocator())
    # ax.grid()
    if ( len(opath) > 0):
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()

def planet_rotation_nruns( tr_spec, models = None, x_range = [2.7, 5.15], opath = '' ):
    
    point_size = 20.
    errorbar_size = 1.
    data_colors = ['blue', 'green', 'purple']
    model_colors = ['black', 'red', 'orange']
    markers = ['d', 's', '^']
    keys = list(tr_spec.keys())
    try:
        model_keys = list(models.keys())
    except AttributeError:
        model_keys = []
    
    # Define grid for binning
    detectors = ['NRS1', 'NRS2']
    # Identify detector jump
    for i_jump in range(len(tr_spec[keys[0]]['wavelength'])):
        if tr_spec[keys[0]]['wavelength'][i_jump] < 3.7 and tr_spec[keys[0]]['wavelength'][i_jump+1] > 3.7:
            break
    
    bin_edges = {}
    bin_edges[detectors[0]] = np.append(tr_spec[keys[0]]['wavelength'][:i_jump+1] - tr_spec[keys[0]]['wavelength_halfwidth'][:i_jump+1], tr_spec[keys[0]]['wavelength'][i_jump] + tr_spec[keys[0]]['wavelength_halfwidth'][i_jump])
    bin_edges[detectors[1]] = np.append(tr_spec[keys[0]]['wavelength'][i_jump+1:] - tr_spec[keys[0]]['wavelength_halfwidth'][i_jump+1:], tr_spec[keys[0]]['wavelength'][-1] + tr_spec[keys[0]]['wavelength_halfwidth'][-1])
    
    plt.clf()
    fig, ax = plt.subplots( 3, 1, figsize=[6,7.8], sharex=True )    # [6,7.8]
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    data_mean = np.mean(tr_spec[keys[0]]['R0']['value'][:7]*1.0E2)
    
    for i in range(len(ax)):
        wavelength_offset = -0.00
        if i == 0:
            unit_conversion = 1.0E2
        elif i == 1:
            unit_conversion = 1.0E2/24.
        else:
            unit_conversion = 1.0E2/(24.**2)
        
        for j, key in enumerate(keys):
            # Data
            try:
                if 'upperr' in list(tr_spec[key]['R' + str(i)].keys()):
                    # Check, if the transmission spectrum has asymmetric errorbars stored
                    errorbars = np.array(list(zip(tr_spec[key]['R' + str(i)]['lowerr'], tr_spec[key]['R' + str(i)]['upperr']))).T
                else:
                    # If not, just use symmetric errorbars
                    errorbars = tr_spec[key]['R' + str(i)]['stderr']
                    
                ax[i].errorbar( tr_spec[key]['wavelength'] + wavelength_offset, tr_spec[key]['R' + str(i)]['value']*unit_conversion, xerr=tr_spec[key]['wavelength_halfwidth'], yerr=errorbars*unit_conversion, color=data_colors[j], elinewidth=errorbar_size, ls='none', zorder=4 )
                ax[i].scatter( tr_spec[key]['wavelength'] + wavelength_offset, tr_spec[key]['R' + str(i)]['value']*unit_conversion, c='white', edgecolor=data_colors[j], s=point_size, marker=markers[j], zorder=5, label=key )
            except KeyError:
                print(key, 'has no', 'R' + str(i))
            wavelength_offset += 0.01
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
        
        # Models
        for j, key in enumerate(model_keys):
            try:
                model_spectrum = models[key]['R' + str(i)]*unit_conversion
                
                if i==0:
                    model_spectrum -= np.mean(model_spectrum[176:]) - data_mean
                # binned_highres, applied_bin_edges, indices = binned_statistic( models[key]['wavelength'], model_spectrum, bins=model_binedges )
                # ax[i].plot( Utils.bin_edges_into_middles(model_binedges), binned_highres, color=model_colors[j], alpha=0.5, zorder=1 )
                ax[i].plot( models[key]['wavelength'], model_spectrum, color=model_colors[j], alpha=0.5, zorder=1 - float(j)*0.1 )
                # y_offset = np.average( tr_spec[key]['R' + str(i)]['value'], weights=1./tr_spec[key]['R' + str(i)]['stderr'] ) / np.mean(models[key]['R' + str(i)])
                
                # NRS1
                binned_model, applied_bin_edges, indices = binned_statistic( models[key]['wavelength'], model_spectrum, bins=bin_edges['NRS1'] )
                ax[i].scatter( Utils.bin_edges_into_middles(bin_edges['NRS1']), binned_model, c='white', edgecolor=model_colors[j], s=point_size, zorder=3 - float(j)*0.1, label=key )
                ax[i].errorbar( Utils.bin_edges_into_middles(bin_edges['NRS1']), binned_model, ecolor=model_colors[j], xerr=Utils.bin_edges_into_halfwidths(bin_edges['NRS1']), ls='None', zorder=2 - float(j)*0.1 )
                # if i==1:
                #     ax[i].scatter( Utils.bin_edges_into_middles(bin_edges['NRS1']), binned_model * y_offset, c='white', edgecolor='orange', s=point_size, zorder=10 )
                
                # NRS2
                binned_model, applied_bin_edges, indices = binned_statistic( models[key]['wavelength'], model_spectrum, bins=bin_edges['NRS2'] )
                ax[i].scatter( Utils.bin_edges_into_middles(bin_edges['NRS2']), binned_model, c='white', edgecolor=model_colors[j], s=point_size, zorder=3 - float(j)*0.1 )
                ax[i].errorbar( Utils.bin_edges_into_middles(bin_edges['NRS2']), binned_model, ecolor=model_colors[j], xerr=Utils.bin_edges_into_halfwidths(bin_edges['NRS2']), ls='None', zorder=2 - float(j)*0.1 )
                # if i==1:
                #     ax[i].scatter( Utils.bin_edges_into_middles(bin_edges['NRS2']), binned_model * y_offset, c='white', edgecolor='orange', s=point_size, zorder=10 )
                
            except TypeError:
                pass
    
    ax[0].set_xlim( x_range )
    
    ax[1].legend( fontsize='x-small' )
    ax[0].set_ylabel('$R_0/R_*$ [\%]')
    ax[1].set_ylabel('$R_1/R_*$ [\%\,hr$^{-1}$]')
    
    # Lines at zero
    for i in range(2):
        ax[i+1].hlines( 0., x_range[0], x_range[1], linestyle='--', alpha=1.0, color='grey', zorder=1. )
    
    y_offset = 0.02#5
    x_h2o = 3.25
    y_h2o = 12.57
    ax[0].text( x_h2o, y_h2o, 'H$_2$O', color='blue' )
    ax[0].plot( [2.7, x_h2o - 0.035], [y_h2o + y_offset, y_h2o + y_offset], color='blue' )
    ax[0].plot( [x_h2o + 0.175, 4.0], [y_h2o + y_offset, y_h2o + y_offset], color='blue' )
    
    x_sio = 4.1
    y_sio = 12.5
    ax[0].text( x_sio, y_sio, 'SiO', color='green' )
    ax[0].plot( [4.0, x_sio - 0.025], [y_sio + y_offset, y_sio + y_offset], color='green' )
    ax[0].plot( [x_sio + 0.1275, 4.3], [y_sio + y_offset, y_sio + y_offset], color='green' )
    
    x_co = 4.65
    y_co = 12.
    ax[0].text( x_co, y_co, 'CO', color='orange' )
    ax[0].plot( [4.3, x_co - 0.035], [y_co + y_offset, y_co + y_offset], color='orange' )
    ax[0].plot( [x_co + 0.125, 5.2], [y_co + y_offset, y_co + y_offset], color='orange' )
    
    if len(ax)>2:
        ax[2].set_ylabel('$R_2/R_*$ [\%\,hr$^{-2}$]')  
    ax[-1].set_xlabel('Wavelength [$\mu$m]')
    ax[0].xaxis.set_minor_locator(AutoMinorLocator())
    
    if ( len(opath) > 0):
        pylab.savefig( opath + '.pdf', dpi=300, bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()

def planet_rotation_talk( tr_spec, model = None, x_range = [2.7, 5.2], opath = '' ):
    
    point_size = 20.
    errorbar_size = 1.
    colors = ['purple', 'green', 'blue']
    markers = ['o', 'o', 'o']
    keys = list(tr_spec.keys())
    
    # Define grid for binning
    detectors = ['NRS1', 'NRS2']
    # Identify detector jump
    for i_jump in range(len(tr_spec[keys[0]]['wavelength'])):
        if tr_spec[keys[0]]['wavelength'][i_jump] < 3.7 and tr_spec[keys[0]]['wavelength'][i_jump+1] > 3.7:
            break
        
    bin_edges = {}
    bin_edges[detectors[0]] = np.append(tr_spec[keys[0]]['wavelength'][:i_jump+1] - tr_spec[keys[0]]['wavelength_halfwidth'][:i_jump+1], tr_spec[keys[0]]['wavelength'][i_jump] + tr_spec[keys[0]]['wavelength_halfwidth'][i_jump])
    bin_edges[detectors[1]] = np.append(tr_spec[keys[0]]['wavelength'][i_jump+1:] - tr_spec[keys[0]]['wavelength_halfwidth'][i_jump+1:], tr_spec[keys[0]]['wavelength'][-1] + tr_spec[keys[0]]['wavelength_halfwidth'][-1])
    
    plt.clf()
    fig, ax = plt.subplots( 2, 1, figsize=[7,3.5], sharex=True )
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    for i in range(len(ax)):
        wavelength_offset = -0.01
        for j, key in enumerate(keys):
            # Data
            try:
                if 'upperr' in list(tr_spec[key]['R' + str(i)].keys()):
                    # Check, if the transmission spectrum has asymmetric errorbars stored
                    asymmetric_error = np.array(list(zip(tr_spec[key]['R' + str(i)]['lowerr'], tr_spec[key]['R' + str(i)]['upperr']))).T
                    ax[i].errorbar( tr_spec[key]['wavelength'] + wavelength_offset, tr_spec[key]['R' + str(i)]['value'], xerr=tr_spec[key]['wavelength_halfwidth'], yerr=asymmetric_error, color=colors[j], elinewidth=errorbar_size, ls='none', zorder=4)
                else:
                    # If not, just use symmetric errorbars
                    ax[i].errorbar( tr_spec[key]['wavelength'] + wavelength_offset, tr_spec[key]['R' + str(i)]['value'], xerr=tr_spec[key]['wavelength_halfwidth'], yerr=tr_spec[key]['R' + str(i)]['stderr'], color=colors[j], elinewidth=errorbar_size, ls='none', zorder=4 )
                ax[i].scatter( tr_spec[key]['wavelength'] + wavelength_offset, tr_spec[key]['R' + str(i)]['value'], c='white', edgecolor=colors[j], s=point_size, zorder=5, label=key )
            except KeyError:
                pass
                # ax[i].scatter( tr_spec[key]['wavelength'] + wavelength_offset, np.zeros(len(tr_spec[key]['wavelength'])), c='white', edgecolor=colors[j], s=point_size, zorder=3, label=key )
            wavelength_offset += 0.01
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
        
        # Model
        try:
            ax[i].plot( model['wavelength'], model['R' + str(i)], color='gray', zorder=1 )
            
            # NRS1
            binned_model, applied_bin_edges, indices = binned_statistic( model['wavelength'], model['R' + str(i)], bins=bin_edges['NRS1'] )
            ax[i].scatter( Utils.bin_edges_into_middles(bin_edges['NRS1']), binned_model, c='white', edgecolor='black', s=point_size, zorder=3, label='GCM, binned' )
            ax[i].errorbar( Utils.bin_edges_into_middles(bin_edges['NRS1']), binned_model, ecolor='black', xerr=Utils.bin_edges_into_halfwidths(bin_edges['NRS1']), ls='None', zorder=2 )
            
            # NRS2
            binned_model, applied_bin_edges, indices = binned_statistic( model['wavelength'], model['R' + str(i)], bins=bin_edges['NRS2'] )
            ax[i].scatter( Utils.bin_edges_into_middles(bin_edges['NRS2']), binned_model, c='white', edgecolor='black', s=point_size, zorder=3 )
            ax[i].errorbar( Utils.bin_edges_into_middles(bin_edges['NRS2']), binned_model, ecolor='black', xerr=Utils.bin_edges_into_halfwidths(bin_edges['NRS2']), ls='None', zorder=2 )
        except TypeError:
            pass
    
    ax[0].set_xlim( x_range )
    ax[0].set_ylim( 0.118, 0.13 )
    ax[1].legend( fontsize='x-small' )
    ax[0].set_ylabel('$R_0/R_*$') 
    ax[1].set_ylabel('$R_1/R_*$ [day$^{-1}$]')  
    # ax[2].set_ylabel('$R_2/R_*$ [day$^{-2}$]')  
    ax[-1].set_xlabel('Wavelength [micron]')
    ax[0].xaxis.set_minor_locator(AutoMinorLocator())
    # ax.grid()
    if ( len(opath) > 0):
        pylab.savefig( opath + '.png', bbox_inches='tight', dpi=400 )
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()

def Transmission_Spectra_deltaBICs( tr_spec, opath='' ):
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    runs = list(tr_spec.keys())
    n = len(runs)
    
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            deltaBIC = tr_spec[runs[j]]['bic'] - tr_spec[runs[i]]['bic']
            label = runs[j] + ' - ' + runs[i] + ' > 0: {0:.0f}/{1:.0f} channels. Median $\Delta BIC$: {2:.1f}'.format( len(np.where(deltaBIC>0.)[0]), len(tr_spec[runs[j]]['bic']), np.median(deltaBIC) )
            plt.scatter( tr_spec[runs[j]]['wavelength'], deltaBIC, s=1., label=label )
            j += 1
        i += 1
    
    plt.xlabel( 'Wavelength [micron]' )
    plt.ylabel( '$\Delta$BIC' )
    plt.legend( fontsize='small' )
    if ( len(opath) > 0):
        pylab.savefig( opath + '.pdf', bbox_inches='tight', dpi=300 )
    else:
        plt.show()
    plt.close()

def Transmission_Spectrum_2runs( trans_spec, limits=None, title=None, opath='' ):
    
    point_size = 1.
    alpha = 0.3
    transit_depth_unit = 1.0E-2 # %
    residual_unit = 1.0E-6      # ppm
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    fig, ax = plt.subplots( 2, 1, sharex=True, height_ratios=[2,1] )
    
    colors = ['blue', 'red']
    keys = list(trans_spec.keys())
    
    # Prepare binning for calculating residuals
    binned_data = {}
    bin_edges = np.linspace( min(trans_spec[keys[0]]['wavelength']['value']), max(trans_spec[keys[0]]['wavelength']['value']), 350 )
    wavelength = Utils.bin_edges_into_middles(bin_edges)
    
    i = 0
    for run in keys:
        
        # Perform binning
        binned_data[run], applied_bin_edges, indices = binned_statistic( trans_spec[run]['wavelength']['value'], trans_spec[run]['RpRs']['value']**2, bins=bin_edges )
        
        if 'upperr' in list(trans_spec[run]['RpRs'].keys()):
            # Check, if the transmission spectrum has asymmetric errorbars stored
            percentile16 = trans_spec[run]['RpRs']['value'] - trans_spec[run]['RpRs']['lowerr']
            percentile84 = trans_spec[run]['RpRs']['value'] + trans_spec[run]['RpRs']['upperr']
            upper_error = percentile84**2 - trans_spec[run]['RpRs']['value']**2
            lower_error = trans_spec[run]['RpRs']['value']**2 -percentile16**2
            asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
            ax[0].errorbar( trans_spec[run]['wavelength']['value'], trans_spec[run]['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha)
        else:
            # If not, just use symmetric errorbars
            ax[0].errorbar( trans_spec[run]['wavelength']['value'], trans_spec[run]['RpRs']['value']**2 / transit_depth_unit, yerr=2.*trans_spec[run]['RpRs']['value']*trans_spec[run]['RpRs']['stderr'] / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=colors[i], alpha=alpha)
            
        ax[0].scatter( trans_spec[run]['wavelength']['value'], trans_spec[run]['RpRs']['value']**2 / transit_depth_unit, c=colors[i], s=point_size, alpha=alpha, label=run )
        
        i += 1
        
    residuals = (binned_data[keys[0]]-binned_data[keys[1]])/residual_unit
    
    wavelength, residuals = Utils.kick_out_nans(wavelength, residuals)
    a1, a0 = np.polyfit(wavelength, residuals, 1)
    ax[1].plot( wavelength, a1*wavelength + a0, color='purple', label='{0:.0f} + {1:.0f}$\lambda$'.format( a0, a1 ) )
    ax[1].scatter( wavelength, residuals, c='purple', s=point_size )
    
    ax[0].set_ylabel( 'Transit depth [%]' )
    ax[0].set_ylim( limits )
    ax[0].set_title( title )
    ax[0].legend()
    ax[1].set_xlabel( 'Wavelength [micron]' )
    ax[1].set_ylabel( 'Residuals [ppm]' )
    ax[1].legend()
    if ( len(opath) > 0):
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    else:
        plt.show()
    plt.close()

def Transmission_Spectrum_2runs_bin( trans_spec, n_bins = 200, limits = None, title = None, opath='' ):
    
    point_size = 1.
    transit_depth_unit = 1.0E-2 # %
    residual_unit = 1.0E-6      # ppm
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    
    fig, ax = plt.subplots( 2, 1, sharex=True, height_ratios=[2,1] )
    
    alpha_binned = 1.0
    alpha_unbinned = 0.1
    colors = ['blue', 'red']
    
    keys = list(trans_spec.keys())
    binned_data = {}
    
    j = 0
    for run in keys:
        
        bin_edges = np.linspace( min(trans_spec[run]['wavelength']['value']), max(trans_spec[run]['wavelength']['value']), n_bins )
        wavelength = Utils.bin_edges_into_middles(bin_edges)
        
        # asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
        # # asymmetric_error = np.array(list(zip(trans_spec[var]['lowerr'], trans_spec[var]['upperr']))).T
        # ax[i].errorbar( trans_spec[run]['wavelength']['value'], trans_spec[run][var]['value'], yerr=asymmetric_error, elinewidth=0.75, ecolor=color_unbinned, alpha=alpha_unbinned, ls='none')
        
        # Original data
        ax[0].scatter( trans_spec[run]['wavelength']['value'], trans_spec[run]['RpRs']['value']**2 / transit_depth_unit, c=colors[j], s=point_size, alpha=alpha_unbinned )
            
        # Perform binning
        binned_data[run], applied_bin_edges, indices = binned_statistic( trans_spec[run]['wavelength']['value'], trans_spec[run]['RpRs']['value']**2, bins=bin_edges )
        
        try:
            # Bin errorbars
            percentile16 = trans_spec[run]['RpRs']['value'] - trans_spec[run]['RpRs']['lowerr']
            percentile84 = trans_spec[run]['RpRs']['value'] + trans_spec[run]['RpRs']['upperr']
            upper_error = percentile84**2 - trans_spec[run]['RpRs']['value']**2
            lower_error = trans_spec[run]['RpRs']['value']**2 -percentile16**2
            upper_error_binned = Utils.calculate_binned_errorbars( upper_error, indices, len(binned_data[run]) )
            lower_error_binned = Utils.calculate_binned_errorbars( lower_error, indices, len(binned_data[run]) )
            asymmetric_error = np.array(list(zip(lower_error_binned, upper_error_binned))).T
            
            ax[0].errorbar( wavelength, binned_data[run] / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=0.75, zorder=9.0E9, ecolor=colors[j], ls='none', alpha=alpha_binned)
        except KeyError:
            print(run)
            # Bin errorbars
            error_binned = Utils.calculate_binned_errorbars( trans_spec[run]['RpRs']['stderr'], indices, len(binned_data[run]) )
            print(error_binned)
            ax[0].errorbar( wavelength, binned_data[run] / transit_depth_unit, yerr=2.*binned_data[run]*error_binned / transit_depth_unit, elinewidth=0.75, zorder=9.0E9, ecolor=colors[j], ls='none', alpha=alpha_binned)
            
        ax[0].scatter( wavelength, binned_data[run] / transit_depth_unit, c=colors[j], s=point_size, label=run, alpha=alpha_binned )
        
        j += 1
        
    ax[0].set_ylabel( 'Transit depth [%]' )
    
    residuals = (binned_data[keys[0]]-binned_data[keys[1]])/residual_unit
    
    wavelength, residuals = Utils.kick_out_nans(wavelength, residuals)
    a1, a0 = np.polyfit(wavelength, residuals, 1)
    ax[1].plot( wavelength, a1*wavelength + a0, color='purple', label='{0:.0f} + {1:.0f}$\lambda$'.format( a0, a1 ) )
    ax[1].scatter( wavelength, residuals, c='purple', s=point_size )
    ax[1].set_ylabel( 'Residuals [ppm]' )
    ax[1].legend()
    
    ax[0].set_ylim( limits )
    ax[1].set_xlabel( 'Wavelength [micron]' )
    ax[0].set_title( title )
    ax[0].legend(fontsize='small')
    if ( len(opath) > 0):
        pylab.savefig( opath + '.pdf', bbox_inches='tight')#, dpi=300 )
    else:
        plt.show()
    plt.close()

def Transmission_Spectrum_nruns_bin( tr_spec, limits, title, opath ):
    
    point_size = 1.
    n_bins = 200
    transit_depth_unit = 1.0E-2 # %
    residual_unit = 1.0E-6      # ppm
    alpha_binned = 0.5
    alpha_unbinned = 0.1
    colors = ['blue', 'red']
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    
    j = 0
    for key in list(tr_spec.keys()):
            
        bin_edges = np.linspace( min(tr_spec[key]['wavelength']['value']), max(tr_spec[key]['wavelength']['value']), n_bins )
        wavelength = Utils.bin_edges_into_middles(bin_edges)
        
        # asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
        # # asymmetric_error = np.array(list(zip(tr_spec[var]['lowerr'], tr_spec[var]['upperr']))).T
        # ax[i].errorbar( tr_spec[key]['wavelength']['value'], tr_spec[key][var]['value'], yerr=asymmetric_error, elinewidth=0.75, ecolor=color_unbinned, alpha=alpha_unbinned, ls='none')
                
        # Original data
        plt.scatter( tr_spec[key]['wavelength']['value'], tr_spec[key]['RpRs']['value']**2, c=colors[j], s=point_size, alpha=alpha_unbinned )
            
        # Perform binning
        binned_data, applied_bin_edges, indices = binned_statistic( tr_spec[key]['wavelength']['value'], tr_spec[key]['RpRs']['value']**2, bins=bin_edges )
            
        try:
            # Bin errorbars
            percentile16 = tr_spec[key]['RpRs']['value'] - tr_spec[key]['RpRs']['lowerr']
            percentile84 = tr_spec[key]['RpRs']['value'] + tr_spec[key]['RpRs']['upperr']
            upper_error = percentile84**2 - tr_spec[key]['RpRs']['value']**2
            lower_error = tr_spec[key]['RpRs']['value']**2 -percentile16**2
            upper_error_binned = Utils.calculate_binned_errorbars( upper_error, indices, len(binned_data) )
            lower_error_binned = Utils.calculate_binned_errorbars( lower_error, indices, len(binned_data) )
            asymmetric_error = np.array(list(zip(lower_error_binned, upper_error_binned))).T
            
            plt.errorbar( wavelength, binned_data, yerr=asymmetric_error, elinewidth=0.75, zorder=9.0E9, ecolor=colors[j], ls='none', alpha=alpha_binned)
        except KeyError:
                
            # Bin errorbars
            error_binned = Utils.calculate_binned_errorbars( tr_spec[key]['RpRs']['stderr'], indices, len(binned_data) )
            plt.errorbar( wavelength, binned_data, yerr=2.*binned_data*error_binned, elinewidth=0.75, zorder=9.0E9, ecolor=colors[j], ls='none', alpha=alpha_binned)
            
        plt.scatter( wavelength, binned_data, c=colors[j], s=point_size, label=key, alpha=alpha_binned )
            
        j += 1
    
    plt.ylabel( 'Transit depth' )
    
    plt.ylim( limits )
    plt.xlabel( 'Wavelength [micron]' )
    plt.title( title )
    plt.legend()
    pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()

def Transmission_Spectrum_nruns_residuals( tr_spec, limits=None, title=None, opath='' ):
    
    runs = list(tr_spec.keys())
    n = len(runs)
    residual_unit = 1.0E-6      # ppm
    palette = ['black', 'blue', 'red']
    markers = ['o', 'd', '*']
    
    point_size = 1.
    alpha = 0.4
    transit_depth_unit = 1.0E-2 # %
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    fig, ax = plt.subplots( 2, 1, figsize=[10,6], sharex=True )
    
    i = 0
    for run in runs:
        if 'transit_depth' in list(tr_spec[run].keys()):
            ax[0].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, yerr=tr_spec[run]['transit_depth']['stderr'] / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=palette[i], alpha=alpha)
            ax[0].scatter( tr_spec[run]['wavelength'], tr_spec[run]['transit_depth']['value'] / transit_depth_unit, c=palette[i], s=point_size, alpha=alpha, label=run, marker=markers[i] )
        else:
            if 'upperr' in list(tr_spec[run]['RpRs'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                percentile16 = tr_spec[run]['RpRs']['value'] - tr_spec[run]['RpRs']['lowerr']
                percentile84 = tr_spec[run]['RpRs']['value'] + tr_spec[run]['RpRs']['upperr']
                upper_error = percentile84**2 - tr_spec[run]['RpRs']['value']**2
                lower_error = tr_spec[run]['RpRs']['value']**2 - percentile16**2
                asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
                ax[0].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=0.75, ls='none', ecolor=palette[i], alpha=alpha)
            else:
                # If not, just use symmetric errorbars
                ax[0].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, yerr=2.*tr_spec[run]['RpRs']['value']*tr_spec[run]['RpRs']['stderr'], elinewidth=0.75, ls='none', ecolor=palette[i], alpha=alpha)
            
            ax[0].scatter( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, c=palette[i], s=point_size, alpha=alpha, label=run, marker=markers[i] )
        
        i += 1
    
    ax[0].set_ylabel( 'Transit depth [%]' )
    ax[0].set_ylim( limits )
    ax[0].set_title( title )
    ax[0].legend(fontsize='small')
    
    # Bin the first run as a reference for all over runs
    bin_edges = np.linspace( min(tr_spec[runs[0]]['wavelength']), max(tr_spec[runs[0]]['wavelength']), 325 )
    wavelength = Utils.bin_edges_into_middles(bin_edges)
    
    i = 0
    while i<n:
        if 'transit_depth' in list(tr_spec[runs[i]].keys()):
            binned_transit_depth_ref, applied_bin_edges, indices = binned_statistic( tr_spec[runs[i]]['wavelength'],  tr_spec[runs[i]]['transit_depth']['value'], bins=bin_edges )
        else:
            binned_transit_depth_ref, applied_bin_edges, indices = binned_statistic( tr_spec[runs[i]]['wavelength'],  tr_spec[runs[i]]['RpRs']['value']**2, bins=bin_edges )
        
        j = i + 1
        while j < len(runs):
            if 'transit_depth' in list(tr_spec[runs[j]].keys()):
                binned_transit_depth, applied_bin_edges, indices = binned_statistic( tr_spec[runs[j]]['wavelength'],  tr_spec[runs[j]]['transit_depth']['value'], bins=bin_edges )
            else:
                binned_transit_depth, applied_bin_edges, indices = binned_statistic( tr_spec[runs[j]]['wavelength'],  tr_spec[runs[j]]['RpRs']['value']**2, bins=bin_edges )
            
            residuals = (binned_transit_depth - binned_transit_depth_ref)/residual_unit
            residuals = residuals - np.nanmean(residuals)
            wavelength_plot, residuals_plot = Utils.kick_out_nans(wavelength, residuals)
            
            a1, a0 = np.polyfit(wavelength_plot, residuals_plot, 1)
            ax[1].plot( wavelength_plot, a1*wavelength_plot + a0, color=palette[i+j-1] )
            ax[1].scatter( wavelength_plot, residuals_plot, s=1., c=palette[i+j-1], label=runs[j] + ' - ' + runs[i] + ' (' + str(int(abs(a1)+0.5)) + ' ppm/micron)' ) 
            
            j += 1
            
        i += 1
    
    ax[1].set_ylabel( 'Residuals [ppm]' )
    ax[1].set_xlabel( 'Wavelength [micron]' )
    ax[1].legend(fontsize='small')
    if len(opath)>0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    plt.show()
    plt.close()

def Transmission_Spectrum_nruns_residuals_nobin( tr_spec, transit_depth_limits=None, residual_limits=None, opath='' ):
    
    runs = list(tr_spec.keys())
    n = len(runs)
    residual_unit = 1.0E-6      # ppm
    palette = ['black', 'blue', 'red']
    markers = ['o', 'd', '*']
    
    point_size = 1.
    alpha = 0.4
    transit_depth_unit = 1.0E-2 # %
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    fig, ax = plt.subplots( 2, 1, figsize=[10,10], sharex=True )
    
    i = 0
    for run in runs:
        if 'upperr' in list(tr_spec[run]['RpRs'].keys()):
            # Check, if the transmission spectrum has asymmetric errorbars stored
            percentile16 = tr_spec[run]['RpRs']['value'] - tr_spec[run]['RpRs']['lowerr']
            percentile84 = tr_spec[run]['RpRs']['value'] + tr_spec[run]['RpRs']['upperr']
            upper_error = percentile84**2 - tr_spec[run]['RpRs']['value']**2
            lower_error = tr_spec[run]['RpRs']['value']**2 - percentile16**2
            asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
            ax[0].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=0.75, ls='none', alpha=alpha)
        else:
            # If not, just use symmetric errorbars
            ax[0].errorbar( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, yerr=2.*tr_spec[run]['RpRs']['value']*tr_spec[run]['RpRs']['stderr'] / transit_depth_unit, elinewidth=0.75, ls='none', alpha=alpha)
            
        ax[0].scatter( tr_spec[run]['wavelength'], tr_spec[run]['RpRs']['value']**2 / transit_depth_unit, s=point_size, alpha=alpha, label=run )
        
        i += 1
    
    ax[0].set_ylabel( 'Transit depth [%]' )
    ax[0].set_ylim( transit_depth_limits )
    ax[0].legend(fontsize='small')
    
    
    i = 0
    while i<n:
        
        j = i + 1
        while j < len(runs):
            residuals = (tr_spec[runs[j]]['RpRs']['value']**2 - tr_spec[runs[i]]['RpRs']['value']**2)/residual_unit
            # residuals = residuals - np.nanmean(residuals)
            wavelength_plot, residuals_plot = Utils.kick_out_nans(tr_spec[runs[j]]['wavelength'], residuals)
            
            # a1, a0 = np.polyfit(wavelength_plot, residuals_plot, 1)
            # ax[1].plot( wavelength_plot, a1*wavelength_plot + a0 )
            ax[1].scatter( wavelength_plot, residuals_plot, s=1., label=runs[j] + ' - ' + runs[i] ) 
            
            j += 1
            
        i += 1
    
    ax[1].set_ylabel( 'Residuals [ppm]' )
    ax[1].set_xlabel( 'Wavelength [micron]' )
    ax[1].set_ylim( residual_limits )
    ax[1].legend(fontsize='small')
    if len(opath)>0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    plt.show()
    plt.close()

def Transmission_Spectrum_nruns_reference( tr_spec_ref, tr_spec, transit_depth_limits=None, residual_limits=None, opath='' ):
    
    runs = list(tr_spec.keys())
    n = len(runs)
    residual_unit = 1.0E-6      # ppm
    
    point_size = 1.
    alpha = 0.4
    transit_depth_unit = 1.0E-2 # %
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    # fig, ax = plt.subplots( n, 1, figsize=[6,6], sharex=True )
    
    for i, run in enumerate(runs):
        residuals = (tr_spec[run]['RpRs']['value']**2 - tr_spec_ref['RpRs']['value']**2) / residual_unit
        plt.scatter( tr_spec[run]['wavelength'], residuals, s=point_size )
        
        wavelength_plot, residuals_plot = Utils.kick_out_nans(tr_spec[run]['wavelength'], residuals)
        a1, a0 = np.polyfit(wavelength_plot, residuals_plot, 1)
        plt.plot( wavelength_plot, a1*wavelength_plot + a0, label=run + ' (' + str(int(abs(a1)+0.5)) + ' ppm/micron)' )
        
        print(run, 'Median:', np.median(residuals))
        
    plt.legend()
    plt.xlabel( 'Wavelength [micron]' )
    plt.ylabel( 'Transit depth difference [ppm]' )
    if len(opath)>0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    plt.show()
    plt.close()

def Transmission_Spectrum_nruns_reference_uncertainties( tr_spec_ref, tr_spec, transit_depth_limits=None, residual_limits=None, opath='' ):
    
    runs = list(tr_spec.keys())
    n = len(runs)
    residual_unit = 1.0E-6      # ppm
    
    point_size = 1.
    alpha = 0.4
    transit_depth_unit = 1.0E-2 # %
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    # fig, ax = plt.subplots( n, 1, figsize=[6,6], sharex=True )
    
    for i, run in enumerate(runs):
        plt.scatter( tr_spec[run]['wavelength'], 2.*tr_spec[run]['RpRs']['value']*tr_spec[run]['RpRs']['upperr']/(2.*tr_spec_ref['RpRs']['value']*tr_spec_ref['RpRs']['upperr']), s=point_size, label=run )
        
    plt.legend()
    plt.xlabel( 'Wavelength [micron]' )
    plt.ylabel( 'Uncertainty w.r.t. phase curve spectrum' )
    if len(opath)>0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    plt.show()
    plt.close()

def Transmission_Spectrum_nruns_2versions( tr_spec, lower_limit = None, upper_limit = None, fig_size=None, opath='' ):
    
    runs = list(tr_spec.keys())
    n = len(runs)
    colors = {'2':'blue', '10':'black', 'residuals':'red'}
    x_min = 2.7
    x_max = 5.2
    point_size = 8.
    errorbar_width = 0.7
    alpha = {'2':0.15, '10':1.0}
    transit_depth_unit = 1.0E-2 # %
    transit_midtime_unit = 24.*3600. # s
    
    # # Process the transit mid time of one of the runs
    # midtime_key = 'Transit stand-alone (This study)'
    # tr_midtime = {}
    # for px in ['2', '10']:
    #     tr_midtime[px] = tr_spec[midtime_key][px]['Tmid']['value']  - np.median(tr_spec[midtime_key][px]['Tmid']['value'])
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    fig, ax = plt.subplots( 1, n, sharex=True, sharey=True, figsize=fig_size )
    
    i = 0
    for run in runs:
        
        if 'transit_depth' in list(tr_spec[run]['10'].keys()):
            ax[i].errorbar( tr_spec[run]['10']['wavelength']['value'], tr_spec[run]['10']['transit_depth']['value'] / transit_depth_unit, yerr=tr_spec[run]['10']['transit_depth']['stderr'] / transit_depth_unit, elinewidth=errorbar_width, ls='none', ecolor=colors['10'], alpha=alpha['10'], zorder=1.)
            ax[i].scatter( tr_spec[run]['10']['wavelength']['value'], tr_spec[run]['10']['transit_depth']['value'] / transit_depth_unit, c='white', edgecolor=colors['10'], s=point_size, alpha=alpha['10'], label=run )
        else:
            
            # 2 px spectra
            if 'upperr' in list(tr_spec[run]['2']['RpRs'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                percentile16 = tr_spec[run]['2']['RpRs']['value'] - tr_spec[run]['2']['RpRs']['lowerr']
                percentile84 = tr_spec[run]['2']['RpRs']['value'] + tr_spec[run]['2']['RpRs']['upperr']
                upper_error = np.abs(percentile84**2 - tr_spec[run]['2']['RpRs']['value']**2)
                lower_error = np.abs(tr_spec[run]['2']['RpRs']['value']**2 - percentile16**2)
                asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
                ax[i].errorbar( tr_spec[run]['2']['wavelength']['value'], tr_spec[run]['2']['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=errorbar_width, ls='none', ecolor=colors['2'], alpha=alpha['2'], zorder=1.)
            else:
                # If not, just use symmetric errorbars
                ax[i].errorbar( tr_spec[run]['2']['wavelength']['value'], tr_spec[run]['2']['RpRs']['value']**2 / transit_depth_unit, yerr=np.abs(2.*tr_spec[run]['2']['RpRs']['value']*tr_spec[run]['2']['RpRs']['stderr']), elinewidth=errorbar_width, ls='none', ecolor=colors['2'], alpha=alpha['2'], zorder=1.)
            
            ax[i].scatter( tr_spec[run]['2']['wavelength']['value'], tr_spec[run]['2']['RpRs']['value']**2 / transit_depth_unit, c='white', edgecolor=colors['2'], s=point_size, alpha=alpha['2'], label='$R\sim 3000$', zorder=2. )
            
            # 10 px spectra
            if 'upperr' in list(tr_spec[run]['10']['RpRs'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                percentile16 = tr_spec[run]['10']['RpRs']['value'] - tr_spec[run]['10']['RpRs']['lowerr']
                percentile84 = tr_spec[run]['10']['RpRs']['value'] + tr_spec[run]['10']['RpRs']['upperr']
                upper_error = percentile84**2 - tr_spec[run]['10']['RpRs']['value']**2
                lower_error = tr_spec[run]['10']['RpRs']['value']**2 - percentile16**2
                asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
                ax[i].errorbar( tr_spec[run]['10']['wavelength']['value'], tr_spec[run]['10']['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=errorbar_width, ls='none', ecolor=colors['10'], alpha=alpha['10'], zorder=3.)
            else:
                # If not, just use symmetric errorbars
                ax[i].errorbar( tr_spec[run]['10']['wavelength']['value'], tr_spec[run]['10']['RpRs']['value']**2 / transit_depth_unit, yerr=2.*tr_spec[run]['10']['RpRs']['value']*tr_spec[run]['10']['RpRs']['stderr'], elinewidth=errorbar_width, ls='none', ecolor=colors['10'], alpha=alpha['10'], zorder=3.)
            
            ax[i].scatter( tr_spec[run]['10']['wavelength']['value'], tr_spec[run]['10']['RpRs']['value']**2 / transit_depth_unit, c='white', edgecolor=colors['10'], s=point_size, alpha=alpha['10'], label='$R\sim 600$', zorder=4. )
            
        ax[i].set_title( run )
        ax[i].set_xlabel( 'Wavelength [$\mu$m]' )
        ax[i].legend( loc = 'upper left')
        # ax[i].set_ylabel( 'Transit depth [%]' )
        i += 1
    
    # # Tmid
    # ax[-1].hlines(0., x_min, x_max, colors='red', linestyles='--' )
    
    # # 2 px
    # ax[-1].scatter( tr_spec[midtime_key]['2']['wavelength']['value'], tr_midtime['2']*transit_midtime_unit, c=colors['2'], s=point_size, alpha=alpha['2'] )
    # asymmetric_error = np.array(list(zip(tr_spec[midtime_key]['2']['Tmid']['lowerr'], tr_spec[midtime_key]['2']['Tmid']['upperr']))).T
    # ax[-1].errorbar( tr_spec[midtime_key]['2']['wavelength']['value'], tr_midtime['2']*transit_midtime_unit, yerr=asymmetric_error*transit_midtime_unit, elinewidth=errorbar_width, ls='none', ecolor=colors['2'], alpha=alpha['2'])
    
    # # 10 px
    # ax[-1].scatter( tr_spec[midtime_key]['10']['wavelength']['value'], tr_midtime['10']*transit_midtime_unit, c=colors['10'], s=point_size, alpha=alpha['10'] )
    # asymmetric_error = np.array(list(zip(tr_spec[midtime_key]['10']['Tmid']['lowerr'], tr_spec[midtime_key]['10']['Tmid']['upperr']))).T
    # ax[-1].errorbar( tr_spec[midtime_key]['10']['wavelength']['value'], tr_midtime['10']*transit_midtime_unit, yerr=asymmetric_error*transit_midtime_unit, elinewidth=errorbar_width, ls='none', ecolor=colors['10'], alpha=alpha['10'])
    
    # ax[-1].set_ylim(-80., 80.)
    # ax[-1].set_ylabel('Transit midtime [s]')
    
    ax[0].xaxis.set_minor_locator(AutoMinorLocator())
    ax[0].yaxis.set_minor_locator(AutoMinorLocator())
    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim( lower_limit, upper_limit )
    # ax[1].set_ylabel( 'Residuals [ppm]' )
    # ax[-1].set_xlabel( 'Wavelength [$\mu$m]' )
    ax[0].set_ylabel( 'Transit depth [%]' )
    fig.tight_layout()
    if len(opath)>0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    plt.show()
    plt.close()

def Transmission_Spectrum_nruns_nversions( tr_spec, wavelength_limits = [2.7,5.2], transit_depth_limits = [1.4, 1.7], fig_size=None, opath='' ):
    
    runs = list(tr_spec.keys())
    n = len(runs)
    colors = ['blue', 'red']
    point_size = 8.
    errorbar_width = 0.7
    transit_depth_unit = 1.0E-2
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    fig, ax = plt.subplots( n, 1, sharex=True, sharey=True, figsize=fig_size )
    
    for i, run in enumerate(runs):
        for j, version in enumerate(tr_spec[run].keys()):
            if 'upperr' in list(tr_spec[run][version]['RpRs'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                percentile16 = tr_spec[run][version]['RpRs']['value'] - tr_spec[run][version]['RpRs']['lowerr']
                percentile84 = tr_spec[run][version]['RpRs']['value'] + tr_spec[run][version]['RpRs']['upperr']
                upper_error = np.abs(percentile84**2 - tr_spec[run][version]['RpRs']['value']**2)
                lower_error = np.abs(tr_spec[run][version]['RpRs']['value']**2 - percentile16**2)
                asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
                ax[i].errorbar( tr_spec[run][version]['wavelength']['value'], tr_spec[run][version]['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, ecolor=[colors[j]], elinewidth=errorbar_width, ls='none', zorder=1.)
            else:
                # If not, just use symmetric errorbars
                ax[i].errorbar( tr_spec[run][version]['wavelength']['value'], tr_spec[run][version]['RpRs']['value']**2 / transit_depth_unit, yerr=np.abs(2.*tr_spec[run][version]['RpRs']['value']*tr_spec[run][version]['RpRs']['stderr']), elinewidth=errorbar_width, ls='none', zorder=1.)
            
            ax[i].scatter( tr_spec[run][version]['wavelength']['value'], tr_spec[run][version]['RpRs']['value']**2 / transit_depth_unit, c='white', edgecolor=colors[j], label=version, s=point_size, zorder=2. )
        
        ax[i].set_ylabel( 'Transit depth [%]' )
        ax[i].set_title( run )
        ax[i].legend( )
    
    ax[0].xaxis.set_minor_locator(AutoMinorLocator())
    ax[0].yaxis.set_minor_locator(AutoMinorLocator())
    ax[0].set_xlim( wavelength_limits )
    ax[0].set_ylim( transit_depth_limits )
    ax[-1].set_xlabel( 'Wavelength [$\mu$m]' )
    fig.tight_layout()
    if len(opath)>0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    plt.show()
    plt.close()

def Transmission_Spectrum_nxmversions_residuals( tr_spec, wavelength_limits = [2.7,5.2], transit_depth_limits = [1.4, 1.7], fig_size=None, opath='' ):
    
    runs = list(tr_spec.keys())
    n_runs = len(runs)
    residuals_color = 'purple'
    colors = ['blue', 'red']
    point_size = 8.
    errorbar_width = 0.7
    transit_depth_unit = 1.0E-2
    residuals_unit = 1.0E-6
    wavelength_limits = np.array(wavelength_limits)
    alpha = 0.4
    markers = ['d', 'o']
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    fig, ax = plt.subplots( 2, n_runs, sharex=True, sharey='row', figsize=fig_size, height_ratios=[2,1] )
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    for i, run in enumerate(runs):
        for j, version in enumerate(tr_spec[run].keys()):
            if 'upperr' in list(tr_spec[run][version]['RpRs'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                percentile16 = tr_spec[run][version]['RpRs']['value'] - tr_spec[run][version]['RpRs']['lowerr']
                percentile84 = tr_spec[run][version]['RpRs']['value'] + tr_spec[run][version]['RpRs']['upperr']
                upper_error = np.abs(percentile84**2 - tr_spec[run][version]['RpRs']['value']**2)
                lower_error = np.abs(tr_spec[run][version]['RpRs']['value']**2 - percentile16**2)
                asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
                ax[0,i].errorbar( tr_spec[run][version]['wavelength']['value'], tr_spec[run][version]['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, ecolor=[colors[j]], elinewidth=errorbar_width, ls='none', zorder=1., alpha=alpha )
            else:
                # If not, just use symmetric errorbars
                ax[0,i].errorbar( tr_spec[run][version]['wavelength']['value'], tr_spec[run][version]['RpRs']['value']**2 / transit_depth_unit, yerr=np.abs(2.*tr_spec[run][version]['RpRs']['value']*tr_spec[run][version]['RpRs']['stderr']), elinewidth=errorbar_width, ls='none', zorder=1., alpha=alpha )
            
            ax[0,i].scatter( tr_spec[run][version]['wavelength']['value'], tr_spec[run][version]['RpRs']['value']**2 / transit_depth_unit, c='white', edgecolor=colors[j], label=version, s=point_size, marker=markers[j], zorder=2., alpha=alpha  )
        
        # ax[0,i].set_title( run )
        
        versions = list(tr_spec[run].keys())
        n_versions = len(versions)
        
        j = 0
        while j < n_versions:
            k = j + 1
            while k<n_versions:
                wavelength = tr_spec[run][versions[j]]['wavelength']['value']
                residuals = (tr_spec[run][versions[j]]['RpRs']['value']**2 - tr_spec[run][versions[k]]['RpRs']['value']**2) / residuals_unit
                # residuals = residuals - np.nanmedian(residuals)
                ax[1,i].scatter( wavelength, residuals, c='white', edgecolor=residuals_color, s=point_size, label='Residuals', zorder=2.  )
                
                a1, a0 = np.polyfit( wavelength, residuals, 1)
                ax[1,i].plot( wavelength_limits, a1*wavelength_limits + a0, color='magenta', label='Linear Fit' )
                print(run,a0,a1)
                
                # Differences in error bars
                percentile84 = tr_spec[run][versions[j]]['RpRs']['value'] + tr_spec[run][versions[j]]['RpRs']['upperr']
                error_j = np.abs(percentile84**2 - tr_spec[run][versions[j]]['RpRs']['value']**2) / residuals_unit
                
                percentile84 = tr_spec[run][versions[k]]['RpRs']['value'] + tr_spec[run][versions[k]]['RpRs']['upperr']
                error_k = np.abs(percentile84**2 - tr_spec[run][versions[k]]['RpRs']['value']**2) / residuals_unit
                
                print( np.median(error_j/error_k) )
                
                k += 1
                
            j += 1
        
        ax[1,i].plot( wavelength_limits, [0.,0.], '--', color='grey', zorder=1.)
        ax[1,i].set_xlabel( 'Wavelength [$\mu$m]' )
    
    ax[0,0].legend( loc='upper left' )
    ax[1,0].legend( loc='upper left' )
    ax[0,0].xaxis.set_minor_locator(AutoMinorLocator())
    ax[0,0].yaxis.set_minor_locator(AutoMinorLocator())
    ax[1,0].yaxis.set_minor_locator(AutoMinorLocator())
    ax[0,0].set_xlim( wavelength_limits )
    ax[0,0].set_ylim( transit_depth_limits )
    ax[1,0].set_ylim( [-1000.,1000.] )
    ax[0,0].set_ylabel( 'Transit depth [%]' )
    ax[1,0].set_ylabel( 'Residuals [ppm]' )
    fig.tight_layout()
    if len(opath)>0:
        pylab.savefig( opath + '.png', bbox_inches='tight', dpi=400 )
    plt.show()
    plt.close()
    
def Transmission_Spectrum_nxmversions_residuals_errorbars( tr_spec, wavelength_limits = [2.7,5.2], transit_depth_limits = [1.4, 1.7], fig_size=[9,8], opath='' ):
    
    runs = list(tr_spec.keys())
    n_runs = len(runs)
    residuals_color = 'purple'
    colors = ['blue', 'red']
    point_size = 8.
    errorbar_width = 0.7
    transit_depth_unit = 1.0E-2
    residuals_unit = 1.0E-6
    wavelength_limits = np.array(wavelength_limits)
    alpha = 0.4
    markers = ['d', 'o']
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    fig, ax = plt.subplots( 3, n_runs, sharex=True, sharey='row', figsize=fig_size, height_ratios=[2,1,1] )
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    for i, run in enumerate(runs):
        
        # Transmission spectra
        for j, version in enumerate(tr_spec[run].keys()):
            if 'upperr' in list(tr_spec[run][version]['RpRs'].keys()):
                # Check, if the transmission spectrum has asymmetric errorbars stored
                percentile16 = tr_spec[run][version]['RpRs']['value'] - tr_spec[run][version]['RpRs']['lowerr']
                percentile84 = tr_spec[run][version]['RpRs']['value'] + tr_spec[run][version]['RpRs']['upperr']
                upper_error = np.abs(percentile84**2 - tr_spec[run][version]['RpRs']['value']**2)
                lower_error = np.abs(tr_spec[run][version]['RpRs']['value']**2 - percentile16**2)
                asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
                ax[0,i].errorbar( tr_spec[run][version]['wavelength']['value'], tr_spec[run][version]['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, ecolor=[colors[j]], elinewidth=errorbar_width, ls='none', zorder=1., alpha=alpha )
                ax[1,i].scatter( tr_spec[run][version]['wavelength']['value'], lower_error/residuals_unit, c='white', edgecolor=colors[j], label=version, s=point_size, marker=markers[j], zorder=2.  )
            else:
                # If not, just use symmetric errorbars
                ax[0,i].errorbar( tr_spec[run][version]['wavelength']['value'], tr_spec[run][version]['RpRs']['value']**2 / transit_depth_unit, yerr=np.abs(2.*tr_spec[run][version]['RpRs']['value']*tr_spec[run][version]['RpRs']['stderr']), elinewidth=errorbar_width, ls='none', zorder=1., alpha=alpha )
            
            ax[0,i].scatter( tr_spec[run][version]['wavelength']['value'], tr_spec[run][version]['RpRs']['value']**2 / transit_depth_unit, c='white', edgecolor=colors[j], label=version, s=point_size, marker=markers[j], zorder=2., alpha=alpha  )
        
        ax[0,i].set_title( run )
        
        versions = list(tr_spec[run].keys())
        n_versions = len(versions)
        
        # Residuals
        j = 0
        while j < n_versions:
            k = j + 1
            while k<n_versions:
                wavelength = tr_spec[run][versions[j]]['wavelength']['value']
                residuals = (tr_spec[run][versions[j]]['RpRs']['value']**2 - tr_spec[run][versions[k]]['RpRs']['value']**2) / residuals_unit
                # print( np.max(np.abs((tr_spec[run][versions[j]]['RpRs']['value']**2 - tr_spec[run][versions[k]]['RpRs']['value']**2)/(2.*tr_spec[run][versions[k]]['RpRs']['upperr']*tr_spec[run][versions[k]]['RpRs']['value']) )))
                errorbar_differences = (2.*tr_spec[run][versions[k]]['RpRs']['upperr']*tr_spec[run][versions[k]]['RpRs']['value'])-(2.*tr_spec[run][versions[j]]['RpRs']['upperr']*tr_spec[run][versions[j]]['RpRs']['value'])
                print( np.median(errorbar_differences) )
                # residuals = residuals - np.nanmedian(residuals)
                ax[2,i].scatter( wavelength, residuals, c='white', edgecolor=residuals_color, s=point_size, label='Residuals', zorder=2.  )
                
                a1, a0 = np.polyfit( wavelength, residuals, 1)
                ax[2,i].plot( wavelength_limits, a1*wavelength_limits + a0, color='magenta', label='Linear Fit' )
                print(run,a0,a1)
                print('At',tr_spec[run][versions[j]]['wavelength']['value'][0],':',a0 + a1 * tr_spec[run][versions[j]]['wavelength']['value'][0])
                print('At',tr_spec[run][versions[j]]['wavelength']['value'][-1],':',a0 + a1 * tr_spec[run][versions[j]]['wavelength']['value'][-1])
                
                # Differences in error bars
                percentile84 = tr_spec[run][versions[j]]['RpRs']['value'] + tr_spec[run][versions[j]]['RpRs']['upperr']
                error_j = np.abs(percentile84**2 - tr_spec[run][versions[j]]['RpRs']['value']**2) / residuals_unit
                
                percentile84 = tr_spec[run][versions[k]]['RpRs']['value'] + tr_spec[run][versions[k]]['RpRs']['upperr']
                error_k = np.abs(percentile84**2 - tr_spec[run][versions[k]]['RpRs']['value']**2) / residuals_unit
                
                print( np.median(error_k/error_j) )
                
                k += 1
                
            j += 1
        
        
        ax[2,i].plot( wavelength_limits, [0.,0.], color='grey', zorder=1.)
        ax[-1,i].set_xlabel( 'Wavelength [$\mu$m]' )
    
    ax[0,0].xaxis.set_minor_locator(AutoMinorLocator())
    for i in range(3):
        ax[i,0].yaxis.set_minor_locator(AutoMinorLocator())
        ax[i,0].legend( loc = 'upper left' )
    
    ax[0,0].set_xlim( wavelength_limits )
    ax[0,0].set_ylim( transit_depth_limits )
    ax[2,0].set_ylim( [-1000.,1000.] )
    ax[1,0].set_ylim( [50.,800.] )
    ax[0,0].set_ylabel( 'Transit depth [\%]' )
    ax[2,0].set_ylabel( 'Residuals [ppm]' )
    ax[1,0].set_ylabel( '$1\sigma$ uncertainties [ppm]' )
    fig.tight_layout()
    if len(opath)>0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
    plt.show()
    plt.close()

def Transmission_Spectrum_2runs_withTmid( tr_spec, broad_fit_params, opath ):
    
    point_size = 1.
    n_bins = 200
    variables = ['RpRs', 'Tmid']
    
    for key in list(tr_spec.keys()):
        tr_spec[key] = Utils.calculate_Tmid_anomaly( tr_spec[key], broad_fit_params )
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    
    fig, ax = plt.subplots( len(variables), 1, sharex=True)#, figsize=[6,4] )
    
    alpha_binned = 0.5
    alpha_unbinned = 0.1
    colors = ['blue', 'red']
    lower_limits = [0.0135, -100.]#2459867.641]
    upper_limits = [0.0165, 100.]#2459867.644]
    
    i = 0
    for var in variables:
        
        j = 0
        for key in list(tr_spec.keys()):
            
            bin_edges = np.linspace( min(tr_spec[key]['wavelength']['value']), max(tr_spec[key]['wavelength']['value']), n_bins )
            wavelength = Utils.bin_edges_into_middles(bin_edges)
            
            if (var=='RpRs'):
                
                # asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
                # # asymmetric_error = np.array(list(zip(tr_spec[var]['lowerr'], tr_spec[var]['upperr']))).T
                # ax[i].errorbar( tr_spec[key]['wavelength']['value'], tr_spec[key][var]['value'], yerr=asymmetric_error, elinewidth=0.75, ecolor=color_unbinned, alpha=alpha_unbinned, ls='none')
                
                # Original data
                ax[i].scatter( tr_spec[key]['wavelength']['value'], tr_spec[key]['RpRs']['value']**2, c=colors[j], s=point_size, alpha=alpha_unbinned )
            
                # Perform binning
                binned_data, applied_bin_edges, indices = binned_statistic( tr_spec[key]['wavelength']['value'], tr_spec[key]['RpRs']['value']**2, bins=bin_edges )
                
                # Bin errorbars
                percentile16 = tr_spec[key]['RpRs']['value'] - tr_spec[key]['RpRs']['lowerr']
                percentile84 = tr_spec[key]['RpRs']['value'] + tr_spec[key]['RpRs']['upperr']
                upper_error = percentile84**2 - tr_spec[key]['RpRs']['value']**2
                lower_error = tr_spec[key]['RpRs']['value']**2 -percentile16**2
                upper_error_binned = Utils.calculate_binned_errorbars( upper_error, indices, len(binned_data) )
                lower_error_binned = Utils.calculate_binned_errorbars( lower_error, indices, len(binned_data) )
                asymmetric_error = np.array(list(zip(lower_error_binned, upper_error_binned))).T
                
                ax[i].errorbar( wavelength, binned_data, yerr=asymmetric_error, elinewidth=0.75, zorder=9.0E9, ecolor=colors[j], ls='none', alpha=alpha_binned)
                ax[i].scatter( wavelength, binned_data, c=colors[j], s=point_size, label=key, alpha=alpha_binned )
                ax[i].set_ylabel( 'Transit depth' )
            
            else:
                # Original data
                ax[i].scatter( tr_spec[key]['wavelength']['value'], tr_spec[key][var]['value'], c=colors[j], s=point_size, alpha=alpha_unbinned )
            
                # Perform binning
                binned_data, applied_bin_edges, indices = binned_statistic( tr_spec[key]['wavelength']['value'], tr_spec[key][var]['value'], bins=bin_edges )
                upper_error_binned = Utils.calculate_binned_errorbars( tr_spec[key][var]['upperr'], indices, len(binned_data) )
                lower_error_binned = Utils.calculate_binned_errorbars( tr_spec[key][var]['lowerr'], indices, len(binned_data) )
                asymmetric_error = np.array(list(zip(lower_error_binned, upper_error_binned))).T
                
                ax[i].errorbar( wavelength, binned_data, yerr=asymmetric_error, elinewidth=0.75, zorder=9.0E9, ecolor=colors[j], alpha=alpha_binned, ls='none')
                ax[i].scatter( wavelength, binned_data, c=colors[j], s=point_size, label=key, alpha=alpha_binned )
                ax[i].set_ylabel( var + ' anomaly [s]' )
                ax[i].legend( fontsize='small' )
            
            j += 1
            
        ax[i].set_ylim( lower_limits[i], upper_limits[i] )
        
        i += 1
    ax[i-1].set_xlabel( 'Wavelength [micron]' )
    pylab.savefig( opath + '_2runs-with-Tmid.pdf', bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()

def Transmission_Spectrum_models( data, models, ylim=None, opath = '' ):
    
    point_size = 1.
    alpha_binned = 0.7
    alpha_unbinned = 0.2
    colors = ['black', 'red', 'orange', 'blue']
    transit_depth_unit = 1.0E-2
    lower_wavelength = 2.6
    upper_wavelength = 5.3
    
    model_keys = list(models.keys())
    bin_edges = Utils.array_edges( models[model_keys[0]]['wavelength']['value'] )
    wavelength = Utils.bin_edges_into_middles( bin_edges )
    
    # Set up plot
    plt.clf()
    plt.figure()
    
    # ------------ Data ------------
    
    plt.scatter( data['wavelength']['value'], data['RpRs']['value']**2/transit_depth_unit, c=colors[0], s=point_size, alpha=alpha_unbinned )

    # Perform binning
    binned_data, applied_bin_edges, indices = binned_statistic( data['wavelength']['value'], data['RpRs']['value']**2, bins=bin_edges )
    
    # Bin errorbars
    try:
        percentile16 = data['RpRs']['value'] - data['RpRs']['lowerr']
        percentile84 = data['RpRs']['value'] + data['RpRs']['upperr']
        upper_error = percentile84**2 - data['RpRs']['value']**2
        lower_error = data['RpRs']['value']**2 -percentile16**2
        upper_error_binned_data = Utils.calculate_binned_errorbars( upper_error, indices, len(binned_data) )
        lower_error_binned_data = Utils.calculate_binned_errorbars( lower_error, indices, len(binned_data) )
        asymmetric_error = np.array(list(zip(lower_error_binned_data, upper_error_binned_data))).T
        
        plt.errorbar( wavelength, binned_data/transit_depth_unit, yerr=asymmetric_error/transit_depth_unit, elinewidth=0.75, ecolor=colors[0], ls='none', alpha=alpha_binned)
    
    except KeyError:
        error_binned_data = Utils.calculate_binned_errorbars( data['RpRs']['stderr'], indices, len(binned_data) )
        
        plt.errorbar( wavelength, binned_data/transit_depth_unit, yerr=error_binned_data/transit_depth_unit, elinewidth=0.75, ecolor=colors[0], ls='none', alpha=alpha_binned)
    
    plt.scatter( wavelength, binned_data/transit_depth_unit, c=colors[0], s=point_size, label='JWST NIRSpec G395H', alpha=alpha_binned )
    
    # ------------ Model ------------
    i = 1
    for key in model_keys:
        model_wavelength, model_transit_depth = Utils.trim_spectrum(models[key]['wavelength']['value'], models[key]['transit_depth']['value'], lower_wavelength, upper_wavelength)
        offset = np.nanmean(binned_data) - np.nanmean(model_transit_depth)
        scale = np.nanmean(binned_data)/np.nanmean(model_transit_depth)
        plt.scatter(model_wavelength, (model_transit_depth + offset)/transit_depth_unit, c=colors[i], s=point_size, zorder=9.9E9, alpha=alpha_binned, label=key )
        
        i += 1
        
    plt.ylabel( 'Transit depth [%]' )
    plt.legend(fontsize='small', loc='upper left')
    plt.xlim( lower_wavelength, upper_wavelength )
    plt.ylim( ylim )
    
    plt.xlabel( 'Wavelength [micron]' )
    if len(opath)>0:
        pylab.savefig( opath + '_model.pdf', bbox_inches='tight' )
    else:
        plt.show()
    plt.close()

def Transmission_Spectrum_nmodels_residuals( data, models, transit_depth_limits=None, fig_size = [9,4.75], opath = '' ):
    
    point_size = 8.
    errorbar_size = 0.8
    alpha_unbinned = 0.2
    data_color = 'black'
    model_colors = ['blue', 'orange', 'red']
    model_linestyles = [':', '--', '-']
    model_linewidth = 1.
    transit_depth_unit = 1.0E-2
    residual_unit = 1.0E-6
    lower_wavelength = 2.6
    upper_wavelength = 5.3
    # fig_size=[7.5,5]                          # Talk
    
    model_keys = list(models.keys())
    model_wavelength, model_transit_depth = Utils.trim_spectrum(models[model_keys[0]]['wavelength']['value'], models[model_keys[0]]['transit_depth']['value'], lower_wavelength, upper_wavelength)
    bin_edges = Utils.array_edges( model_wavelength )
    wavelength = Utils.bin_edges_into_middles( bin_edges )
    
    # Set up plot
    plt.clf()
    fig, ax = plt.subplots( 2, 1, figsize=fig_size, sharex=True, height_ratios=[7,3] )
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    # ------------ Data ------------
    
    ax[0].errorbar( data['wavelength']['value'], data['RpRs']['value']**2/transit_depth_unit, yerr=2.*data['RpRs']['value']*data['RpRs']['stderr'] / transit_depth_unit, elinewidth=errorbar_size, ls='none', ecolor=data_color, alpha=alpha_unbinned, zorder=1.)
    ax[0].scatter( data['wavelength']['value'], data['RpRs']['value']**2/transit_depth_unit, c='white', edgecolor=data_color, s=point_size, alpha=alpha_unbinned, zorder=2. )
    
    # Perform binning
    binned_data, applied_bin_edges, indices = binned_statistic( data['wavelength']['value'], data['RpRs']['value']**2, bins=bin_edges )
    
    # Bin errorbars
    try:
        percentile16 = data['RpRs']['value'] - data['RpRs']['lowerr']
        percentile84 = data['RpRs']['value'] + data['RpRs']['upperr']
        upper_error = percentile84**2 - data['RpRs']['value']**2
        lower_error = data['RpRs']['value']**2 -percentile16**2
        upper_error_binned_data = Utils.calculate_binned_errorbars( upper_error, indices, len(binned_data) )
        lower_error_binned_data = Utils.calculate_binned_errorbars( lower_error, indices, len(binned_data) )
        errors_binned_data = np.array(list(zip(lower_error_binned_data, upper_error_binned_data))).T
    except KeyError:
        errors_binned_data = Utils.calculate_binned_errorbars( data['RpRs']['stderr'], indices, len(binned_data) )
    
    ax[0].errorbar( wavelength, binned_data/transit_depth_unit, yerr=errors_binned_data/transit_depth_unit, elinewidth=errorbar_size, ecolor=data_color, ls='none', zorder=3. )
    ax[0].scatter( wavelength, binned_data/transit_depth_unit, c='white', edgecolor=data_color, s=point_size, label='JWST NIRSpec G395H', zorder=4. )
    
    data_wavelength_nrs1, data_transitdepth_nrs1 = Utils.trim_spectrum( data['wavelength']['value'], data['RpRs']['value']**2, 2.5, 3.7)
    
    comparison_interval = np.asarray([2.9, 3.7])
    # comparison_interval = [2.8,5.1]
    
    # ------------ Models ------------
    ax[1].plot( [lower_wavelength, upper_wavelength], [0.,0.], color='grey', zorder=1.)
    i = 0
    for key in model_keys:
        model_wavelength, model_transitdepth = Utils.trim_spectrum(models[key]['wavelength']['value'], models[key]['transit_depth']['value'], lower_wavelength, upper_wavelength)
        
        # Calculate the offset to fit NRS1
        model_wavelength_nrs1, model_transitdepth_nrs1 = Utils.trim_spectrum(models[key]['wavelength']['value'], models[key]['transit_depth']['value'], comparison_interval[0], comparison_interval[1])
        offset = np.nanmean(data_transitdepth_nrs1) - np.nanmean(model_transitdepth_nrs1)
        
        model_shifted = model_transitdepth + offset
        
        red_chi2 = Utils.reduced_chisquare(binned_data, np.mean(errors_binned_data, axis=0), model_shifted[1:-1], 1)
        print( key, 'reduced chi2: ' + str(red_chi2) )
        
        ax[0].plot(model_wavelength, model_shifted/transit_depth_unit, model_linestyles[i], color=model_colors[i], linewidth=model_linewidth, zorder=5., label=key)# + ', $\chi_\\nu^2=${:.1f}'.format(red_chi2) )
        
        residuals = binned_data - (model_transitdepth[1:-1]+offset)
        
        ax[1].plot(wavelength, residuals/residual_unit, model_linestyles[i], color=model_colors[i], linewidth=model_linewidth, zorder=2.)
        ax[1].errorbar(wavelength, residuals/residual_unit, yerr=errors_binned_data/residual_unit, elinewidth=errorbar_size, ecolor=model_colors[i], ls='none', zorder=3.)
        ax[1].scatter(wavelength, residuals/residual_unit, c='white', edgecolor=model_colors[i], s=point_size, zorder=4.)
        
        i += 1
        
    # ax[0].vlines( comparison_interval, ylim[0], ylim[1])
    
    ax[0].xaxis.set_minor_locator(AutoMinorLocator())
    ax[0].yaxis.set_minor_locator(AutoMinorLocator())
    ax[1].yaxis.set_minor_locator(AutoMinorLocator())
    ax[0].set_ylabel( 'Transit depth [\%]' )
    ax[0].legend(loc='upper left')
    ax[0].set_xlim( [2.7,5.2] )
    ax[0].set_ylim( transit_depth_limits )
    ax[1].set_ylabel('Residuals [ppm]')
    ax[1].set_xlabel( 'Wavelength [$\mu$m]' )
    fig.tight_layout()
    if len(opath)>0:
        pylab.savefig( opath + '.pdf', bbox_inches='tight' )
        # pylab.savefig( opath + '.png', bbox_inches='tight', dpi=600 )
    plt.show()
    plt.close()
    
    # Reduced chi2 without one wavelength region
    for i in range(len(binned_data)):
        if wavelength[i] > 3.9 and wavelength[i] < 4.2:
            binned_data[i] = np.nan
    
    print('No SiO: ' +  str(Utils.reduced_chisquare(binned_data, np.mean(errors_binned_data, axis=0), model_transitdepth[1:-1] + offset, 1)))

def retrieval_spectra( data, retrievals, colors, linestyles, m = 0, fig_size = [9,4.5], wavelength_limits=[2.7,5.2], transit_depth_limits=[1.425, 1.625], opath='' ):
    
    # Set colors
    alpha_uncertainties = 0.1
    transit_depth_unit = 1.0E-2
    point_size = 8.
    errorbar_size = 0.7
    alpha_data = 1.0
    
    plt.clf()
    fig, ax = plt.subplots(figsize=fig_size)
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    if 'upperr' in list(data['RpRs'].keys()):
        # Check, if the transmission spectrum has asymmetric errorbars stored
        percentile16 = data['RpRs']['value'] - data['RpRs']['lowerr']
        percentile84 = data['RpRs']['value'] + data['RpRs']['upperr']
        upper_error = percentile84**2 - data['RpRs']['value']**2
        lower_error = data['RpRs']['value']**2 - percentile16**2
        asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
        ax.errorbar( data['wavelength']['value'], data['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=errorbar_size, ecolor='black', ls='none', zorder=1., alpha=alpha_data)
    else:
        # If not, just use symmetric errorbars
        ax.errorbar( data['wavelength']['value'], data['RpRs']['value']**2/transit_depth_unit, yerr=2.*data['RpRs']['value']*data['RpRs']['stderr']/transit_depth_unit, elinewidth=errorbar_size, ecolor='black', ls='none', zorder=1., alpha=alpha_data )
    
    ax.scatter( data['wavelength']['value'], data['RpRs']['value']**2/transit_depth_unit, s=point_size, c='white', edgecolors='black', zorder=2., alpha=alpha_data, label = 'JWST NIRSpec G395H' )
    
    for model in retrievals.keys():
        if ('lower' in retrievals[model].keys() and 'upper' in retrievals[model].keys()):
            ax.fill_between( retrievals[model]['wavelength'], retrievals[model]['lower']/transit_depth_unit, retrievals[model]['upper']/transit_depth_unit, color=colors[model], alpha=alpha_uncertainties, zorder=2.)
        if m!=0:
            red_chi2 = Utils.reduced_chisquare(data['RpRs']['value']**2, 2.*data['RpRs']['value']*data['RpRs']['stderr'], retrievals[model]['median'], m[model])
            # print(model, r'$\chi_\\nu^2 = $', red_chi2)
    
        ax.plot( retrievals[model]['wavelength'], retrievals[model]['median']/transit_depth_unit, linestyles[model], color=colors[model], zorder=3., label=model + ', $\chi_\\nu^2=${:.2f}'.format(red_chi2))
    
    ax.set_xlabel('Wavelength [$\mu$m]')
    ax.set_ylabel('Transit depth [\%]')
    ax.set_xlim(wavelength_limits)
    ax.set_ylim(transit_depth_limits)
    ax.legend(loc='upper left')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    
    if len(opath)>0:
        fig.savefig( opath + '.pdf', bbox_inches='tight' )
    
    plt.show()
    plt.close()

def retrieval_spectra2( data1, retrievals1, data2, retrievals2, colors, linestyles, m = 0, fig_size = [9,4.5], wavelength_limits=[2.7,5.2], transit_depth_limits=[1.425, 1.625], opath='' ):
    
    # Set colors
    alpha_uncertainties = 0.1
    transit_depth_unit = 1.0E-2
    point_size = 5.5
    errorbar_size = 0.5
    alpha_data = 1.0
    alpha_models = 0.7
    
    plt.clf()
    fig, ax = plt.subplots( 1, 2, figsize=fig_size, sharex=True, sharey=True)
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    # First data set
    if 'upperr' in list(data1['RpRs'].keys()):
        # Check, if the transmission spectrum has asymmetric errorbars stored
        percentile16 = data1['RpRs']['value'] - data1['RpRs']['lowerr']
        percentile84 = data1['RpRs']['value'] + data1['RpRs']['upperr']
        upper_error = percentile84**2 - data1['RpRs']['value']**2
        lower_error = data1['RpRs']['value']**2 - percentile16**2
        asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
        ax[0].errorbar( data1['wavelength']['value'], data1['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=errorbar_size, ecolor='black', ls='none', zorder=1., alpha=alpha_data)
    else:
        # If not, just use symmetric errorbars
        ax[0].errorbar( data1['wavelength']['value'], data1['RpRs']['value']**2/transit_depth_unit, yerr=2.*data1['RpRs']['value']*data1['RpRs']['stderr']/transit_depth_unit, elinewidth=errorbar_size, ecolor='black', ls='none', zorder=1., alpha=alpha_data )
    
    ax[0].scatter( data1['wavelength']['value'], data1['RpRs']['value']**2/transit_depth_unit, s=point_size, c='white', edgecolors='black', zorder=2., alpha=alpha_data, label = 'Phase-curve analysis' )
    
    for model in retrievals1.keys():
        if ('lower' in retrievals1[model].keys() and 'upper' in retrievals1[model].keys()):
            ax[0].fill_between( retrievals1[model]['wavelength'], retrievals1[model]['lower']/transit_depth_unit, retrievals1[model]['upper']/transit_depth_unit, color=colors[model], alpha=alpha_uncertainties, zorder=2.)
        if m!=0:
            red_chi2 = Utils.reduced_chisquare(data1['RpRs']['value']**2, 2.*data1['RpRs']['value']*data1['RpRs']['stderr'], retrievals1[model]['median'], m[model])
            # print(model, r'$\chi_\\nu^2 = $', red_chi2)
    
        ax[0].plot( retrievals1[model]['wavelength'], retrievals1[model]['median']/transit_depth_unit, linestyles[model], color=colors[model], zorder=3., alpha=alpha_models, label=model + ', $\chi_\\nu^2=${:.2f}'.format(red_chi2))
    
    # Second data set
    if 'upperr' in list(data2['RpRs'].keys()):
        # Check, if the transmission spectrum has asymmetric errorbars stored
        percentile16 = data2['RpRs']['value'] - data2['RpRs']['lowerr']
        percentile84 = data2['RpRs']['value'] + data2['RpRs']['upperr']
        upper_error = percentile84**2 - data2['RpRs']['value']**2
        lower_error = data2['RpRs']['value']**2 - percentile16**2
        asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
        ax[1].errorbar( data2['wavelength']['value'], data2['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=errorbar_size, ecolor='black', ls='none', zorder=1., alpha=alpha_data)
    else:
        # If not, just use symmetric errorbars
        ax[1].errorbar( data2['wavelength']['value'], data2['RpRs']['value']**2/transit_depth_unit, yerr=2.*data2['RpRs']['value']*data2['RpRs']['stderr']/transit_depth_unit, elinewidth=errorbar_size, ecolor='black', ls='none', zorder=1., alpha=alpha_data )
    
    ax[1].scatter( data2['wavelength']['value'], data2['RpRs']['value']**2/transit_depth_unit, s=point_size, c='white', edgecolors='black', zorder=2., alpha=alpha_data, label = 'Transit-only analysis' )
    
    for model in retrievals2.keys():
        if ('lower' in retrievals2[model].keys() and 'upper' in retrievals2[model].keys()):
            ax[1].fill_between( retrievals2[model]['wavelength'], retrievals2[model]['lower']/transit_depth_unit, retrievals2[model]['upper']/transit_depth_unit, color=colors[model], alpha=alpha_uncertainties, zorder=2.)
        if m!=0:
            red_chi2 = Utils.reduced_chisquare(data2['RpRs']['value']**2, 2.*data2['RpRs']['value']*data2['RpRs']['stderr'], retrievals2[model]['median'], m[model])
            # print(model, r'$\chi_\\nu^2 = $', red_chi2)
    
        ax[1].plot( retrievals2[model]['wavelength'], retrievals2[model]['median']/transit_depth_unit, linestyles[model], color=colors[model], zorder=3., alpha=alpha_models, label=model + ', $\chi_\\nu^2=${:.2f}'.format(red_chi2))
    
    ax[0].set_xlabel('Wavelength [$\mu$m]')
    ax[1].set_xlabel('Wavelength [$\mu$m]')
    ax[0].set_ylabel('Transit depth [\%]')
    ax[0].set_xlim(wavelength_limits)
    ax[0].set_ylim(transit_depth_limits)
    ax[0].legend(fontsize='small')
    ax[1].legend(fontsize='small')
    ax[0].xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    fig.tight_layout()
    
    if len(opath)>0:
        fig.savefig( opath + '.pdf', bbox_inches='tight' )
    
    plt.show()
    plt.close()

def transmission_contributions( data, wavelength, ml, continuum, contributions, colors, linestyles, fig_size = [8,4], wavelength_limits=[2.7,5.2], transit_depth_limits=[1.425, 1.625], opath='' ):
    
    # Set colors
    transit_depth_unit = 1.0E-2
    point_size = 8.
    errorbar_size = 0.7
    alpha_data = 1.0
    
    plt.clf()
    fig, ax = plt.subplots(figsize=fig_size)
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    if 'upperr' in list(data['RpRs'].keys()):
        # Check, if the transmission spectrum has asymmetric errorbars stored
        percentile16 = data['RpRs']['value'] - data['RpRs']['lowerr']
        percentile84 = data['RpRs']['value'] + data['RpRs']['upperr']
        upper_error = percentile84**2 - data['RpRs']['value']**2
        lower_error = data['RpRs']['value']**2 - percentile16**2
        asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
        ax.errorbar( data['wavelength']['value'], data['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=errorbar_size, ecolor='black', ls='none', zorder=1., alpha=alpha_data)
    else:
        # If not, just use symmetric errorbars
        ax.errorbar( data['wavelength']['value'], data['RpRs']['value']**2/transit_depth_unit, yerr=2.*data['RpRs']['value']*data['RpRs']['stderr']/transit_depth_unit, elinewidth=errorbar_size, ecolor='black', ls='none', zorder=1., alpha=alpha_data )
    
    ax.scatter( data['wavelength']['value'], data['RpRs']['value']**2/transit_depth_unit, s=point_size, c='white', edgecolors='black', zorder=2., alpha=alpha_data, label = 'JWST NIRSpec G395H' )
    ax.plot( wavelength, ml, ':', color='magenta', label='ML NEMESIS model', zorder=4.)
    for source in contributions.keys():
        ax.plot( wavelength, contributions[source], linestyles[source], color=colors[source], label=source, zorder=3.)
        ax.fill_between( wavelength, continuum, contributions[source], color=colors[source], alpha=0.3)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.set_xlabel('Wavelength [$\mu$m]')
    ax.set_ylabel('Transit depth [\%]')
    ax.set_xlim(wavelength_limits)
    ax.set_ylim(transit_depth_limits)
    ax.legend(loc='upper left')
    
    if len(opath)>0:
        fig.savefig( opath + '.png', bbox_inches='tight', dpi=400 )
    else:
        plt.show()
    plt.close()

def transmission_contributions_residuals( data, wavelength, ml, continuum, contributions, colors, linestyles, fig_size = [11,6], wavelength_limits=[2.7,5.2], transit_depth_limits=[1.425, 1.625], opath='' ):
    
    # Set colors
    transit_depth_unit = 1.0E-2
    residual_unit = 1.0E-6
    point_size = 8.
    errorbar_size = 0.7
    alpha_data = 1.0
    # fig_size=[8,5]        # Talk plot
    
    plt.clf()
    fig, ax = plt.subplots(2, 1, figsize=fig_size, height_ratios=[3,1], sharex=True )
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    # Plot the data
    if 'upperr' in list(data['RpRs'].keys()):
        # Check, if the transmission spectrum has asymmetric errorbars stored
        percentile16 = data['RpRs']['value'] - data['RpRs']['lowerr']
        percentile84 = data['RpRs']['value'] + data['RpRs']['upperr']
        upper_error = percentile84**2 - data['RpRs']['value']**2
        lower_error = data['RpRs']['value']**2 - percentile16**2
        asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
        ax[0].errorbar( data['wavelength']['value'], data['RpRs']['value']**2 / transit_depth_unit, yerr=asymmetric_error / transit_depth_unit, elinewidth=errorbar_size, ecolor='black', ls='none', zorder=1., alpha=alpha_data)
    else:
        # If not, just use symmetric errorbars
        ax[0].errorbar( data['wavelength']['value'], data['RpRs']['value']**2/transit_depth_unit, yerr=2.*data['RpRs']['value']*data['RpRs']['stderr']/transit_depth_unit, elinewidth=errorbar_size, ecolor='black', ls='none', zorder=1., alpha=alpha_data )
    ax[0].scatter( data['wavelength']['value'], data['RpRs']['value']**2/transit_depth_unit, s=point_size, c='white', edgecolors='black', zorder=2., alpha=alpha_data, label = 'JWST NIRSpec G395H' )
    
    # Molecular contributions
    ax[0].plot( wavelength, ml, ':', color='magenta', label='ML NEMESIS model', zorder=4.)
    for source in contributions.keys():
        ax[0].plot( wavelength, contributions[source], linestyles[source], color=colors[source], label=source, zorder=3.)
        ax[0].fill_between( wavelength, continuum, contributions[source], color=colors[source], alpha=0.3)
    
    # Residuals
    ax[1].plot( wavelength_limits, [0., 0.], color='grey', zorder=1. )
    ax[1].scatter( wavelength, (data['RpRs']['value']**2-ml*1.0E-2) / residual_unit, s=point_size, c='white', edgecolors=colors['Residuals'], zorder=3. )
    asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
    ax[1].errorbar( wavelength, (data['RpRs']['value']**2-ml*1.0E-2) / residual_unit, yerr=asymmetric_error/ residual_unit, ecolor=colors['Residuals'], ls='none', elinewidth=errorbar_size, zorder=2. )
    
    ax[0].xaxis.set_minor_locator(AutoMinorLocator())
    ax[0].yaxis.set_minor_locator(MultipleLocator(0.01))
    ax[1].yaxis.set_minor_locator(AutoMinorLocator())
    ax[1].set_xlabel('Wavelength [$\mu$m]')
    ax[0].set_ylabel('Transit depth [\%]')
    ax[1].set_ylabel('Residuals [ppm]')
    ax[0].set_xlim(wavelength_limits)
    ax[0].set_ylim(transit_depth_limits)
    ax[1].set_ylim( [-800, 800] )
    ax[0].legend(loc='upper left')
    
    if len(opath)>0:
        fig.savefig( opath + '.pdf', bbox_inches='tight', dpi=400 )
    plt.show()
    plt.close()

def Transmission_Spectrum_both_res_models( data, models, opath ):
    
    unit = 1.0E02      # %
    point_size = 1.
    alpha_binned = 0.7
    alpha_unbinned = 0.2
    colors = ['blue', 'red', 'orange']
    
    # Set up plot
    plt.clf()
    plt.figure(figsize=(5,3.5))
    
    # ------------ Data ------------
    
    keys = list(data.keys())
    
    plt.scatter( data[keys[0]]['wavelength']['value'], unit * data[keys[0]]['RpRs']['value']**2, c=colors[0], s=point_size, alpha=alpha_unbinned )

    plt.errorbar( data[keys[1]]['wavelength']['value'], unit * data[keys[1]]['RpRs']['value']**2, yerr=unit*2.*data[keys[1]]['RpRs']['value']*data[keys[1]]['RpRs']['stderr'], elinewidth=0.75, ecolor=colors[0], ls='none', alpha=alpha_binned)
    plt.scatter( data[keys[1]]['wavelength']['value'], unit* data[keys[1]]['RpRs']['value']**2, c=colors[0], s=point_size, label='JWST NIRSpec G395H', alpha=alpha_binned )
    
    
    # ------------ Model ------------
    
    i = 0
    for key in list(models.keys()):
        plt.scatter( models[key]['wavelength']['value'], unit*models[key]['RpRs2']['value'], c=colors[1 + i], s=point_size, zorder=9.0E09, alpha=alpha_binned, label=key )
        i += 1

    plt.ylabel( 'Transit depth [%]' )
    plt.legend(fontsize='small', loc='upper left')
    plt.xlim( 2.6, 5.3 )
    plt.ylim( 1.43, 1.64 )
    
    plt.xlabel( 'Wavelength [micron]' )
    # pylab.savefig( opath + '_model.pdf', bbox_inches='tight' )
    pylab.savefig( opath + 'model.png', dpi=300, bbox_inches='tight' )
    
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()

def Transmission_Spectrum_model_residuals( data, model, opath ):
    
    bin_edges = np.linspace( min(data['wavelength']['value']), max(data['wavelength']['value']), 150 )
    wavelength = Utils.bin_edges_into_middles( bin_edges )
    
    point_size = 1.
    alpha_binned = 0.5
    alpha_unbinned = 0.1
    colors = ['blue', 'red']
    wavelength_range = [2.6, 5.3]
    
    # Set up plot
    plt.clf()
    fig, ax = plt.subplots( 2, 1, sharex=True, figsize=[7,4] ) # Default: figsize=[6,4]
    
    # First, plot the transmission spectrum
    
    # ------------ Data ------------
    
    ax[0].scatter( data['wavelength']['value'], data['RpRs']['value']**2, c=colors[0], s=point_size, alpha=alpha_unbinned )

    # Perform binning
    binned_data, applied_bin_edges, indices = binned_statistic( data['wavelength']['value'], data['RpRs']['value']**2, bins=bin_edges )
    
    # Bin errorbars
    percentile16 = data['RpRs']['value'] - data['RpRs']['lowerr']
    percentile84 = data['RpRs']['value'] + data['RpRs']['upperr']
    upper_error = percentile84**2 - data['RpRs']['value']**2
    lower_error = data['RpRs']['value']**2 -percentile16**2
    upper_error_binned_data = Utils.calculate_binned_errorbars( upper_error, indices, len(binned_data) )
    lower_error_binned_data = Utils.calculate_binned_errorbars( lower_error, indices, len(binned_data) )
    asymmetric_error = np.array(list(zip(lower_error_binned_data, upper_error_binned_data))).T
    
    ax[0].errorbar( wavelength, binned_data, yerr=asymmetric_error, elinewidth=0.75, zorder=9.0E9, ecolor=colors[0], ls='none', alpha=alpha_binned)
    ax[0].scatter( wavelength, binned_data, c=colors[0], s=point_size, label='Data', alpha=alpha_binned )
    
    
    # ------------ Model ------------
    
    ax[0].scatter( model['Wavelength (um)'], model['Mean (Fp/Fs)']**2, c=colors[1], s=point_size, alpha=alpha_unbinned/3. )

    # Perform binning
    binned_model, applied_bin_edges, indices = binned_statistic( model['Wavelength (um)'], model['Mean (Fp/Fs)']**2, bins=bin_edges )
    
    # Bin errorbars
    upper_error = model['Upper 1-sig (Fp/Fs)']**2 - model['Mean (Fp/Fs)']**2
    lower_error = model['Mean (Fp/Fs)']**2 - model['Lower 1-sig (Fp/Fs)']**2
    upper_error_binned = Utils.calculate_binned_errorbars( upper_error, indices, len(binned_model) )
    lower_error_binned = Utils.calculate_binned_errorbars( lower_error, indices, len(binned_model) )
    asymmetric_error = np.array(list(zip(lower_error_binned, upper_error_binned))).T
    
    ax[0].errorbar( wavelength, binned_model, yerr=asymmetric_error, elinewidth=0.75, zorder=9.0E9, ecolor=colors[1], ls='none', alpha=alpha_binned)
    ax[0].scatter( wavelength, binned_model, c=colors[1], s=point_size, label='Model', alpha=alpha_binned )
    
    ax[0].set_ylabel( 'Transit depth $(R_p/R_s)^2$' )
    ax[0].legend(fontsize='small')
    ax[0].set_xlim( wavelength_range )
    ax[0].set_ylim( 0.0135, 0.0165 )
    
    # ------------ Residuals ------------
    
    unit = 1.0E02       # %
    
    ax[1].plot( wavelength_range, [0., 0.], '--', color='grey' )
    ax[1].scatter( wavelength, (binned_data-binned_model)/binned_data * unit, c='black', s=point_size )
    asymmetric_error = np.array(list(zip(lower_error_binned_data/binned_data, upper_error_binned_data/binned_data))).T
    ax[1].errorbar( wavelength, (binned_data-binned_model)/binned_data * unit, yerr=asymmetric_error*unit, ecolor='black', ls='none', elinewidth=0.75, alpha=alpha_binned )
    ax[1].set_ylim( -6., 4.5 )
    ax[1].set_ylabel( 'Relative difference [%]' )
    
    ax[1].set_xlabel( 'Wavelength [micron]' )
    pylab.savefig( opath + '_model-residuals.png', dpi=300, bbox_inches='tight' )
    pylab.savefig( opath + '_model-residuals.pdf', bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()

def Transmission_Spectrum_model_Tmid( data, model, broad_fit_params, opath ):
    
    bin_edges = np.linspace( min(data['wavelength']['value']), max(data['wavelength']['value']), 150 )
    wavelength = Utils.bin_edges_into_middles( bin_edges )
    
    point_size = 1.
    alpha_binned = 0.5
    alpha_unbinned = 0.1
    colors = ['blue', 'red']
    wavelength_range = [2.6, 5.3]
    
    # Set up plot
    plt.clf()
    fig, ax = plt.subplots( 2, 1, sharex=True, figsize=[7.8,4] ) # Default: figsize=[6,4]
    
    # ------------ Transmission spectrum data ------------
    
    ax[0].scatter( data['wavelength']['value'], data['RpRs']['value']**2, c=colors[0], s=point_size, alpha=alpha_unbinned )

    # Perform binning
    binned_data, applied_bin_edges, indices = binned_statistic( data['wavelength']['value'], data['RpRs']['value']**2, bins=bin_edges )
    
    # Bin errorbars
    percentile16 = data['RpRs']['value'] - data['RpRs']['lowerr']
    percentile84 = data['RpRs']['value'] + data['RpRs']['upperr']
    upper_error = percentile84**2 - data['RpRs']['value']**2
    lower_error = data['RpRs']['value']**2 -percentile16**2
    upper_error_binned_data = Utils.calculate_binned_errorbars( upper_error, indices, len(binned_data) )
    lower_error_binned_data = Utils.calculate_binned_errorbars( lower_error, indices, len(binned_data) )
    asymmetric_error = np.array(list(zip(lower_error_binned_data, upper_error_binned_data))).T
    
    ax[0].errorbar( wavelength, binned_data, yerr=asymmetric_error, elinewidth=0.75, zorder=9.0E9, ecolor=colors[0], ls='none', alpha=alpha_binned)
    ax[0].scatter( wavelength, binned_data, c=colors[0], s=point_size, label='Data', zorder=9.1E9, alpha=alpha_binned )
    
    
    # ------------ Transmission spectrum model ------------
    
    ax[0].scatter( model['Wavelength (um)'], model['Mean (Fp/Fs)']**2, c=colors[1], s=point_size, alpha=alpha_unbinned/3. )

    # Perform binning
    binned_model, applied_bin_edges, indices = binned_statistic( model['Wavelength (um)'], model['Mean (Fp/Fs)']**2, bins=bin_edges )
    
    # Bin errorbars
    upper_error = model['Upper 1-sig (Fp/Fs)']**2 - model['Mean (Fp/Fs)']**2
    lower_error = model['Mean (Fp/Fs)']**2 - model['Lower 1-sig (Fp/Fs)']**2
    upper_error_binned = Utils.calculate_binned_errorbars( upper_error, indices, len(binned_model) )
    lower_error_binned = Utils.calculate_binned_errorbars( lower_error, indices, len(binned_model) )
    asymmetric_error = np.array(list(zip(lower_error_binned, upper_error_binned))).T
    
    ax[0].errorbar( wavelength, binned_model, yerr=asymmetric_error, elinewidth=0.75, zorder=9.0E9, ecolor=colors[1], ls='none', alpha=alpha_binned)
    ax[0].scatter( wavelength, binned_model, c=colors[1], s=point_size, label='Model', zorder=9.1E9, alpha=alpha_binned )
    
    ax[0].set_ylabel( 'Transit depth' )
    ax[0].legend(fontsize='small')
    ax[0].set_xlim( wavelength_range )
    ax[0].set_ylim( 0.0135, 0.0165 )
    ax[0].grid()
    
    # ------------ Tmid data ------------
    
    data = Utils.calculate_Tmid_anomaly( data, broad_fit_params )
    
    # Original data
    ax[1].scatter( data['wavelength']['value'], data['Tmid']['value'], c=colors[0], s=point_size, alpha=alpha_unbinned )

    # Perform binning
    binned_data, applied_bin_edges, indices = binned_statistic( data['wavelength']['value'], data['Tmid']['value'], bins=bin_edges )
    upper_error_binned = Utils.calculate_binned_errorbars( data['Tmid']['upperr'], indices, len(binned_data) )
    lower_error_binned = Utils.calculate_binned_errorbars( data['Tmid']['lowerr'], indices, len(binned_data) )
    asymmetric_error = np.array(list(zip(lower_error_binned, upper_error_binned))).T
    
    ax[1].errorbar( wavelength, binned_data, yerr=asymmetric_error, elinewidth=0.75, zorder=9.0E9, ecolor=colors[0], alpha=alpha_binned, ls='none')
    ax[1].scatter( wavelength, binned_data, c=colors[0], s=point_size, alpha=alpha_binned )
    ax[1].set_ylim( -100., 100. )
    ax[1].set_ylabel( 'Transit midtime anomaly [s]' )
    ax[1].grid()
    
    ax[1].set_xlabel( 'Wavelength [micron]' )
    pylab.savefig( opath + '_limb-asymmetries.png', dpi=300, bbox_inches='tight' )
    pylab.savefig( opath + '_limb-asymmetries.pdf', bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()

def Transmission_Spectrum_uncertainties( tr_spec1, tr_spec2, opath ):
    
    plt.clf()
    fig, ax = plt.subplots( 2, 1, sharex=True )
    
    ax[0].plot( tr_spec2['wavelength']['value'], 2.*tr_spec2['RpRs']['value']*tr_spec2['RpRs']['stderr'], color='black', label='Tmid fixed' )
    ax[0].plot( tr_spec1['wavelength']['value'], 2.*tr_spec1['RpRs']['value']*tr_spec1['RpRs']['stderr'], color='red', label='Tmid free', alpha=0.6 )
    ax[0].set_ylabel( '$u$((R$_p$/R$_s$)$^2$)' )
    ax[0].legend()
    
    ax[1].plot( tr_spec1['wavelength']['value'], 2.*(tr_spec2['RpRs']['value']*tr_spec2['RpRs']['stderr']-tr_spec1['RpRs']['value']*tr_spec1['RpRs']['stderr']), color='blue' )
    ax[1].set_ylabel( 'Tmid fixed - Tmid free' )
    
    
    fig.savefig( opath + '_uncertainties.pdf', bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()

def Transmission_Spectrum_withTmid( tr_spec, broad_fit_params, opath ):
    
    point_size = 2.
    
    tr_spec = Utils.calculate_Tmid_anomaly( tr_spec, broad_fit_params )
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    
    fig, ax = plt.subplots( 2, 1, sharex=True, figsize=[7,4] )
    
    ax[0].scatter( tr_spec['wavelength']['value'], tr_spec['RpRs']['value']**2, c='black', s=point_size )
    ax[0].errorbar( tr_spec['wavelength']['value'], tr_spec['RpRs']['value']**2, yerr=2.*tr_spec['RpRs']['value']*tr_spec['RpRs']['stderr'], elinewidth = point_size/3., ecolor='black', ls='none')
    ax[0].set_ylabel( 'Transit depth (R$_p$/R$_s$)$^2$' )
    ax[0].set_ylim( [0.0139, 0.0164] )
    
    ax[1].scatter( tr_spec['wavelength']['value'], tr_spec['Tmid']['value'], c='black', s=point_size )
    ax[1].errorbar( tr_spec['wavelength']['value'], tr_spec['Tmid']['value'], yerr=tr_spec['Tmid']['stderr'], elinewidth = point_size/3., ecolor='black', ls='none')
    ax[1].set_xlabel( 'Wavelength (micron)' )
    ax[1].set_ylabel( 'Transit midtime offset [s]' )
    ax[1].set_ylim( [-80., 80.] )
    
    pylab.savefig( opath + '_withTmid.png', dpi=300, bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()
    
def mcmc_acceptance_fraction( fit ):
    """
    Plots the acceptance fraction for all walkers for an lmfit emcee run.

    Parameters
    ----------
    fit : lmfit minimizer result
        lmfit emcee minimizer result .
        
    """
    
    plt.clf()
    plt.plot( fit.acceptance_fraction, 'o' )
    plt.xlabel('Walker')
    plt.ylabel('Acceptance fraction')
    plt.show()
    plt.close()
    
def mcmc_chain( fit ):
    
    variables = fit.flatchain.keys()
    point_size = 2.
    
    # Clear plot memory and plot the Transmission spectrum
    plt.clf()
    
    fig, ax = plt.subplots( len(variables), 1, figsize=[8,10], sharex=True )
    
    i = 0
    for var in variables:
        ax[i].plot( fit.flatchain[var])#, c='black', s=point_size )
        ax[i].set_ylabel( var )
        i += 1
    
    ax[i-1].set_xlabel('Steps')
    #pylab.savefig( filepath + '_all-parameters.pdf', bbox_inches='tight' )
    plt.show()
    plt.close()

def emcee_acceptance_fraction( acceptance_fraction, odir ):
    """
    Plots the acceptance fraction for all walkers for an lmfit emcee run.

    Parameters
    ----------
    fit : lmfit minimizer result
        lmfit emcee minimizer result .
        
    """
    
    plt.clf()
    
    plt.plot( acceptance_fraction, 'o' )
    plt.xlabel('Walker')
    plt.ylabel('Acceptance fraction')
    if (len(odir)>0):
        pylab.savefig( os.path.join(odir, 'emcee_acceptance-fraction.pdf'), bbox_inches='tight' )
    else:
        plt.show()
        
    plt.close()

def emcee_walkers( samples, variables, opath='' ):
    
    plt.clf()
    n_variables = len(variables)
    fig, axes = plt.subplots(n_variables, figsize=(8, 60), sharex=True)
    
    for k in range(n_variables):
        ax = axes[k]
        ax.plot(samples[:, :, k], alpha=0.1)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(variables[k])
        #ax.yaxis.set_label_coords(-0.1, 0.5)
        
    axes[-1].set_xlabel("step number")
    
    if len(opath)>0:
        fig.savefig( opath + '.png', dpi=600, bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    
    plt.close()
    
def corner_plot( mcmc_result, truths, opath):
    
    axis_labels = []
    for parameter in mcmc_result['parameters']:
        axis_labels += [Utils.corner_labels( parameter )]
    
    # # Unit conversion into ppm
    # for parameter in ['sigma_sys_NRS1', 'sigma_sys_NRS2']:
    #     try:
    #         i = np.where( np.array(mcmc_result['parameters']) == parameter )[0][0]
    #         mcmc_result['flat_chain'][:,i] = mcmc_result['flat_chain'][:,i] * 1.0E6
    #     except IndexError:
    #         pass
    
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    cornerplot = corner(mcmc_result['flat_chain'], labels=axis_labels, show_titles=True, truths=truths, quantiles=[0.16, 0.5, 0.84] )
    if pr.USE_ASTRONODES:
        cornerplot.savefig( opath + '.png', dpi=50 )
    else:
        cornerplot.savefig( opath + '.pdf', bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        cornerplot.show()
    plt.close()
    
def allen( residuals, opath ):
    """
    Plot RMS vs. bin size looking for time-correlated noise
    residuals are residuals between model and data divided by the model
    Taken from POET: https://github.com/kevin218/POET/blob/master/code/lib/plots.py
    
    Input: residuals between data and model divided by the model values
    """
    # time = data['bjd_tdb']
    # residuals = model.norm_resid
    # residuals = residuals[np.argsort(time)]
    
    rms, stderr, binsz = Utils.computeRMS(residuals, binstep=1)
    normfactor = 1e-6
    plt.rcParams.update({'legend.fontsize':11})
    plt.figure(1111, figsize=(8,6))
    plt.clf()
    # plt.suptitle('Correlated Noise at wvl = x micron', size=16)
    plt.loglog(binsz, rms/normfactor, color='black', lw=1.5, label='Fit RMS', zorder=3)    # our noise
    plt.loglog(binsz, stderr/normfactor, color='red', ls='-', lw=2, label='Std. Err.', zorder=1) # expected noise
    plt.xlim(0.95, binsz[-1]*2)
    plt.ylim(stderr[-1]/normfactor/2., stderr[0]/normfactor*2.)
    plt.xlabel("Bin Size", fontsize=14)
    plt.ylabel("RMS (ppm)", fontsize=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.legend()
    if (len(opath)>0):
        plt.savefig( opath + '_allen.pdf', bbox_inches='tight' )
    else:
        plt.show()
    plt.close('all')

def limb( w, q1, q2 ):
    
    plt.clf()
    plt.plot( [min(w), max(w)], [0.,0.], '--', color='grey' )
    plt.plot(w,q1, label='q1', alpha=0.75)
    plt.plot(w,q2, label='q2', alpha=0.75)
    plt.legend()
    plt.xlabel('Wavelength [Micron]')
    plt.savefig( 'LD_parameters.pdf', bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    
def fp_fs( w, fp_fs ):
    unit = 1.0E06
    plt.clf()
    fig, ax = plt.subplots(2, sharex=True)
    i = 0
    for key in list(fp_fs.keys()):
        ax[i].plot( w, fp_fs[key]*unit)
        ax[i].set_ylabel('$F_p/F_s$ [ppm] ' + key)
        i += 1
    ax[i-1].set_xlabel('Wavelength [Micron]')
    
    plt.savefig( 'fp_fs.pdf', bbox_inches='tight' )
    if pr.SHOW_PLOTS:
        plt.show()
    plt.close()

def retrieved_atmospheres( T, CO, H2O, SiO, H2, H, colors, linestyles, T_limits = None, fig_size=[9,8], legend_size = 'small', opath = '' ):
    
    # Set colors and such
    alpha_uncertainties = 0.1
    p_min = 1.0E-6
    p_max = 1.0E0
    
    plt.clf()
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    fig, ax = plt.subplots( 3, 2, sharey=True, figsize=fig_size )
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    # Temperature
    for model in T.keys():
        ax[0,0].plot( T[model]['median'], T[model]['p'], linestyles[model], color=colors[model] )
        ax[0,0].fill_betweenx( T[model]['p'], T[model]['lower'], T[model]['upper'], color=colors[model], alpha=alpha_uncertainties )
    ax[0,0].set_ylim( p_max, p_min )
    ax[0,0].set_xlim( T_limits )
    ax[0,0].set_yscale('log')
    ax[0,0].set_xlabel('Temperature [K]')
    ax[0,0].set_ylabel('Pressure [bar]')
    
    # CO
    for model in CO.keys():
        ax[0,1].plot( CO[model]['median'], CO[model]['p'], linestyles[model], color=colors[model] )
        if ('lower' in CO[model].keys() and 'upper' in CO[model].keys()):
            ax[0,1].fill_betweenx( CO[model]['p'], CO[model]['lower'], CO[model]['upper'], color=colors[model], alpha=alpha_uncertainties )
    ax[0,1].set_xlim( 2.0E-5, 1.1E-1 )
    ax[0,1].set_xscale('log')
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlabel('CO abundance')
    
    # H2O
    for model in H2O.keys():
        ax[1,0].plot( H2O[model]['median'], H2O[model]['p'], linestyles[model], color=colors[model] )
        if ('lower' in H2O[model].keys() and 'upper' in H2O[model].keys()):
            ax[1,0].fill_betweenx( H2O[model]['p'], H2O[model]['lower'], H2O[model]['upper'], color=colors[model], alpha=alpha_uncertainties )
    ax[1,0].set_xlim( 1.0E-18, 1.0E-1 )
    ax[1,0].set_xscale('log')
    ax[1,0].set_yscale('log')
    ax[1,0].set_xlabel('H$_2$O abundance')
    ax[1,0].set_ylabel('Pressure [bar]')
    
    # SiO
    for model in SiO.keys():
        ax[1,1].plot( SiO[model]['median'], SiO[model]['p'], linestyles[model], color=colors[model] )
        if ('lower' in SiO[model].keys() and 'upper' in SiO[model].keys()):
            ax[1,1].fill_betweenx( SiO[model]['p'], SiO[model]['lower'], SiO[model]['upper'], color=colors[model], alpha=alpha_uncertainties )
    ax[1,1].set_xlim( 1.0E-10, 1.0E-3 )
    ax[1,1].set_xscale('log')
    ax[1,1].set_yscale('log')
    ax[1,1].set_xlabel('SiO abundance')
    
    # H2
    for model in H2.keys():
        ax[2,0].plot( H2[model]['median'], H2[model]['p'], linestyles[model], color=colors[model], label=model )
        if ('lower' in H2[model].keys() and 'upper' in H2[model].keys()):
            ax[2,0].fill_betweenx( H2[model]['p'], H2[model]['lower'], H2[model]['upper'], color=colors[model], alpha=alpha_uncertainties )
    unshown_model = 'NEMESIS (globally uniform)'
    ax[2,0].plot( [1.,1.], [1.,1.], linestyles[unshown_model], color=colors[unshown_model], label=unshown_model )
    # unshown_model = 'PETRA (free chemistry)'
    # ax[2,0].plot( [1.,1.], [1.,1.], linestyles[unshown_model], color=colors[unshown_model], label=unshown_model )
    ax[2,0].set_xlim( 1.0E-6, 1.0E0 )
    ax[2,0].set_xscale('log')
    ax[2,0].set_yscale('log')
    ax[2,0].set_xlabel('H$_2$ abundance')
    ax[2,0].set_ylabel('Pressure [bar]')
    ax[2,0].legend(fontsize=legend_size)
    
    # H
    for model in H.keys():
        ax[2,1].plot( H[model]['median'], H[model]['p'], linestyles[model], color=colors[model] )
        if ('lower' in H[model].keys() and 'upper' in H[model].keys()):
            ax[2,1].fill_betweenx( H[model]['p'], H[model]['lower'], H[model]['upper'], color=colors[model], alpha=alpha_uncertainties )
    ax[2,1].set_xlim( 1.0E-12, 1.0E0 )
    ax[2,1].set_xscale('log')
    ax[2,1].set_yscale('log')
    ax[2,1].set_xlabel('H abundance')
    ax[0,0].xaxis.set_minor_locator(AutoMinorLocator())
    
    plt.tight_layout()
    if len(opath)>0:
        plt.savefig( opath + '.pdf', bbox_inches='tight' )
    plt.show()
    plt.close()
    
    # Print out Si/O and Si/C ratios
    models = list(SiO.keys())
    for model in models[:3]:
        i_pmax = np.argmax( SiO[model]['p'] )
        Si = SiO[model]['median'][i_pmax]
        C = CO[model]['median'][i_pmax]
        O = SiO[model]['median'][i_pmax] + CO[model]['median'][i_pmax] + H2O[model]['median'][i_pmax]
        print( model, 'O/Si:', O/Si, 'C/Si:', C/Si)
    
    # Jo
    i_pmax = 0
    Si = SiO['NEMESIS (globally uniform)']['median'][i_pmax]
    C = CO['NEMESIS (globally uniform)']['median'][i_pmax]
    O = SiO['NEMESIS (globally uniform)']['median'][i_pmax] + CO['NEMESIS (globally uniform)']['median'][i_pmax] + H2O['NEMESIS (nightside)']['median'][i_pmax]
    print( 'NEMESIS', 'O/Si:', O/Si, 'C/Si:', C/Si)
    
def posterior_histograms( samples, size=[5,4], opath='' ):
    
    parameters = list(samples.keys())
    nbins = 20
    
    plt.clf()
    fig, ax = plt.subplots(1,len(parameters)+1, sharey=True, figsize=size )
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    i_plot = 0
    for parameter in parameters:
        for model in samples[parameter].keys():
            percentiles = np.percentile( samples[parameter][model], [1, 16, 50, 84, 99] )
            
            # First, find out the maximum bin value for normalization purposes
            bin_edges = np.linspace( percentiles[0], percentiles[-1], nbins+1)
            counts, applied_bin_edges, indices = binned_statistic( samples[parameter][model], 'nonsense', bins=bin_edges, statistic='count' )
            
            ax[i_plot].hist(samples[parameter][model], range=[percentiles[0], percentiles[-1]], weights=np.zeros(len(samples[parameter][model])) + 1./np.max(counts), bins=nbins, alpha=0.4 )
            # ax[i_plot,j_plot].scatter( percentiles[2], 0.05 + 0.05 *float(k_plot), c=color, zorder=9.9E9, s=10.)
            # ax[i_plot,j_plot].errorbar( percentiles[2], 0.05 + 0.05 *float(k_plot), xerr=[[percentiles[2]-percentiles[1]], [percentiles[3]-percentiles[2]]], ls=None, c=color, zorder=9.0E9 )
                
        # ax[i_plot,j_plot].get_yaxis().set_ticks([])
            # if i_plot==0:
            #     ax[i_plot,j_plot].set_title(chemical_label(parameter))
            # else:
            #     ax[i_plot,j_plot].set_xlabel('log10(Mass mixing ratio)')
                
        i_plot += 1
        
    fig.tight_layout()
    if len(opath)>0:
        pylab.savefig( opath + '.pdf' )
    else:
        plt.show()

def histogram( samples, size=[5,4], opath='' ):
    
    nbins = 20
    
    plt.clf()
    fig, ax = plt.subplots(1, sharey=True, figsize=size )
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    
    percentiles = np.percentile( samples, [1, 16, 50, 84, 99] )
    
    ax.hist(samples, range=[percentiles[0], percentiles[-1]], bins=nbins, alpha=0.4 )
    # ax[i_plot,j_plot].scatter( percentiles[2], 0.05 + 0.05 *float(k_plot), c=color, zorder=9.9E9, s=10.)
    # ax[i_plot,j_plot].errorbar( percentiles[2], 0.05 + 0.05 *float(k_plot), xerr=[[percentiles[2]-percentiles[1]], [percentiles[3]-percentiles[2]]], ls=None, c=color, zorder=9.0E9 )
        
    fig.tight_layout()
    if len(opath)>0:
        pylab.savefig( opath + '.pdf' )
    else:
        plt.show()

def posterior_histogram( samples, probabilities, colors, x_limits=[0.,1.], xlabel=None, size=[5,4], nbins=25, opath='' ):
    
    models = list(samples.keys())
    alpha = 0.5
    
    plt.clf()
    fig, ax = plt.subplots(figsize=size)
    plt.rcParams.update({'text.usetex' : True, 'font.family' : 'sans-serif'})
    i = 0
    for model in models:
        try:
            ax.hist(samples[model], weights=probabilities[model], range=x_limits, bins=nbins, alpha=alpha, color=colors[model], hatch='/', density=True, label=model, edgecolor='black' ) #
            
            absolute_val_array = np.abs(np.cumsum(probabilities[model]) - 0.5)
            smallest_difference_index = absolute_val_array.argmin()
            median=samples[model][smallest_difference_index]
            
            #Calculate 1, sigma interval from cummulative distribution of probability which has sorted by the parameter value
            smallest_difference_index=np.abs(np.cumsum(probabilities[model]) - 0.5E0+0.68268E0/2.).argmin()
            lowerr=(samples[model][smallest_difference_index])
            smallest_difference_index=np.abs(np.cumsum(probabilities[model]) - 0.5E0-0.68268E0/2.).argmin()
            upperr=(samples[model][smallest_difference_index])
            
        except KeyError:
            ax.hist(samples[model], range=x_limits, bins=nbins, alpha=alpha, color=colors[model], density=True, hatch='\\', label=model, edgecolor='black' ) #
            
            lowerr, median, upperr = np.percentile( samples[model], [15.85, 50, 84.15] )
            
        # Evaluate and plot 1 sigma range and median
        ax.axvline(x=lowerr, linestyle='--', color=colors[model])
        ax.axvline(x=median, linestyle='-', color=colors[model])
        ax.axvline(x=upperr, linestyle='--', color=colors[model])
        
        print(xlabel, model, median, '+', upperr-median, '-', median-lowerr)
        i += 1
    
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Normalized probability')
    ax.set_xlim( x_limits )
    ax.set_ylim( 0, 8 )
    ax.legend( fontsize='small' )
    fig.tight_layout()
    if len(opath)>0:
        fig.savefig( opath + '.pdf' )
    # else:
    plt.show()
