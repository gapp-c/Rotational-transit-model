import pdb, sys, os
import numpy as np
import pandas as pd
import pickle
import xarray as xr
import scipy.stats
import parameters as pr
from lmfit import Parameters
import Plot, calculateBatmanModel
from scipy import constants
from scipy import stats

def pretty_labels( var ):
    
    if var == 'RpRs':
        return '$R_p/R_*$'
    elif var == 'Tmid':
        return 'Transit mid-time anomaly [s]'
    elif var == 'c0':
        return '$c_0$'
    elif var == 'c1':
        return '$c_1$'
    elif var == 'c2':
        return '$c_2$'
    elif var == 'u1':
        return '$u_1$'
    elif var == 'u2':
        return '$u_2$'
    elif var == 'sigma_sys':
        return '$\sigma_{sys}$ [ppm]'
    else:
        return var

def corner_labels( var ):
    
    if var == 'inc':
        return '$i$ [$^\circ$]'
    elif var == 'aRs':
        return '$a/R_*$'
    elif var == 'Tmid_NRS1':
        return '$T_{mid}$ (NRS1)'
    elif var == 'RpRs_NRS1':
        return '$R_p/R_*$ (NRS1)'
    elif var == 'c0_NRS1':
        return '$c_0$ (NRS1)'
    elif var == 'c1_NRS1':
        return '$c_1$ (NRS1)'
    elif var == 'c2_NRS1':
        return '$c_2$ (NRS1)'
    elif var == 'u1_NRS1':
        return '$u_1$ (NRS1)'
    elif var == 'u2_NRS1':
        return '$u_2$ (NRS1)'
    elif var == 'Tmid_NRS2':
        return '$T_{mid}$ (NRS2)'
    elif var == 'RpRs_NRS2':
        return '$R_p/R_*$ (NRS2)'
    elif var == 'c0_NRS2':
        return '$c_0$ (NRS2)'
    elif var == 'c1_NRS2':
        return '$c_1$ (NRS2)'
    elif var == 'c2_NRS2':
        return '$c_2$ (NRS2)'
    elif var == 'u1_NRS2':
        return '$u_1$'
    elif var == 'u2_NRS2':
        return '$u_2$'
    elif var == 'sigma_sys_NRS1':
        return '$\sigma_{sys}$ (NRS1, ppm)'
    elif var == 'sigma_sys_NRS2':
        return '$\sigma_{sys}$ (NRS2, ppm)'
    elif var == 'R0_NRS2':
        return '$R_0/R_*$ [\%]'
    elif var == 'R1_NRS2':
        return '$R_1/R_*$ [\% day$^{-1}$]'
    elif var == 'R2_NRS2':
        return '$R_2/R_*$ [day$^{-2}$]'
    else:
        return var

def impact_parameter2inc(b, aRs):
    cosi = b/aRs
    return np.rad2deg( np.arccos( cosi ) )

def inc2impact_parameter(inc, aRs):
    return aRs * np.cos( np.deg2rad(inc) )
    
def LD_coefficients_q2u( q1, q2 ):
    return 2.*np.sqrt(q1)*q2, (1.-2.*q2)*np.sqrt(q1)

def LD_coefficients_u2q( u1, u2 ):
    return (u1+u2)**2, u1/(2.*(u1+u2))

# def computeQuadLD( wavMicr=[] ):
#     """
#     Routine that uses the exotic_ld package to compute theoretical 
#     limb darkening coefficients.
#     """
    
#     sld = StellarLimbDarkening( pr.METALLICITY, pr.EFFECTIVE_TEMPERATURE, pr.LOG_GRAVITY, '3D', pr.EXOTIC_DATA_DIR )
    
#     # Start and end of wavelength interval [angstroms].
#     wavelength_range = 1.0e4*wavMicr
    
#     # Instrument mode. Load custom G395H throughput.
#     thput = pd.read_csv( os.path.join( pr.CUSTOM_INPUT_DIR, pr.THROUGHPUT_FILE) ).values
#     thput_x = 1.0e4 * thput[:,0]
#     thput_y = thput[:,1]
    
#     mode = 'custom'
#     u1, u2 = sld.compute_quadratic_ld_coeffs( wavelength_range, mode, thput_x, thput_y )
    
#     return u1, u2


# log prior functions
def log_boxcar( parameter_value, lower_limit, upper_limit ):
    if parameter_value < lower_limit or parameter_value > upper_limit:
        return -np.inf
    return 0.

def log_gaussian(parameter_value, mu, sigma):
    return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(parameter_value-mu)**2/sigma**2

def prior_transform( pars, priors ):
    """
    Transforms priors for dynesty
    """
    n = len(pars)
    outp = np.zeros(n)
    parameter_names = list( priors.keys() )
    for i in range(n):
        parameter = parameter_names[i]
        if priors[parameter]['prior'] == 'uniform':
            outp[i] = prior_transform_uniform( pars[i], priors[parameter] )
        elif priors[parameter]['prior'] == 'gaussian':
            outp[i] = prior_transform_gaussian( pars[i], priors[parameter] )
        else:
            print('UNKNOWN PRIOR')
    
    return outp

def prior_transform_uniform( u, prior ):
    """
    Transform uniform variable from interval [0,1] to [lower_limit,upper_limit]
    """
    
    return u * (prior['upper_limit'] - prior['lower_limit']) + prior['lower_limit']

def prior_transform_gaussian( u, prior ):
    """
    Transform uniform variable from interval [0,1] to [lower_limit,upper_limit]
    """
    
    return scipy.stats.norm.ppf(u, loc=prior['mu'], scale=prior['sigma'])

def lmfit2dict( pars ):
    """
    Takes lmfit parameters and returns them in a dictionary.
    """
    
    dictionary = {}
    
    for var in pars.keys():
        dictionary[var] = {}
        
        dictionary[var]['value'] = pars[var].value
        dictionary[var]['minimum'] = pars[var].min
        dictionary[var]['maximum'] = pars[var].max
        
    return dictionary

def lmfit2arrays( pars ):
    """
    Takes lmfit parameters and returns the varied ones in numpy arrays.
    """
    
    n = count_varied_parameters_lmfit(pars)
    
    parameter_names = np.empty( n , dtype='<U16' )
    parameter_values = np.zeros( n )
    
    parameters = list(pars.keys())
    
    j = 0
    for parameter in parameters:
        if pars[parameter].vary:
            parameter_names[j] = parameter
            parameter_values[j] = pars[parameter].value
            j += 1
        
    return parameter_values, parameter_names

def arrays2dict( parameter_values, parameter_names ):
    
    dictionary = {}
    
    i = 0
    for name in parameter_names:
        dictionary[name] = parameter_values[i]
        i += 1
        
    return dictionary

def merge_dictionaries (dict1, dict2):
    
    """
    Adds all variables of dict2 that are not listed in dict1 into dict1
    """
    
    variables1 = list(dict1.keys())
    variables2 = list(dict2.keys())
    
    for var in variables2:
        if not( var in variables1 ):
            dict1[var] = dict2[var]
            
    return dict1

def merge_spectral_lightcurves( lc ):
    
    detectors = list(lc.keys())
    n_chan = 0
    for detector in detectors:
        n_chan += len(list(lc[detector].keys()))
    
    dictionary = {}
    i = 0
    for detector in detectors:
        keys = list( lc[detector].keys() )
        for key in keys:
            new_key = 'ch{0:.0f}of{1:.0f}'.format( i+1, n_chan )
            dictionary[new_key] = {}
            lc_keys = list(lc[detector][key].keys())
            for lc_key in lc_keys:
                dictionary[new_key][lc_key] = lc[detector][key][lc_key]
            dictionary[new_key]['detector'] = detector
            i += 1
    
    return dictionary

def switch_off_emission( params ):
    
    new_params = {}
    variables = list(params.keys())
    
    for var in variables:
        new_params[var] = params[var]
    
    for nrs in ['NRS1', 'NRS2']:
        new_params['RpRs_' + nrs] = 0.
        new_params['fpfs_night_' + nrs] = 0.
        new_params['fpfs_day_' + nrs] = 0.

    return new_params

def uncertainty_medians(trans_spec):
    
    keys = list(trans_spec.keys())
    
    standard_uncertainties = {}
    for key in keys:
        if not(key in ['wavelength', 'wavelength_halfwidth', 'chisquare', 'outliers']):
            standard_uncertainties[key] = np.nanmedian(trans_spec[key]['stderr'])
            
    return standard_uncertainties

def emcee_sampler2dictionary ( sampler, parameters, burn_number, thin_number ):
    
    dictionary = {}
    
    dictionary['parameters'] = parameters
    dictionary['acceptance_fraction'] = sampler.acceptance_fraction
    try:
        dictionary['autocorrelation_time'] = sampler.get_autocorr_time()
    except Exception:
        pass
    dictionary['flat_chain'] = sampler.get_chain( discard=burn_number, thin=thin_number, flat=True )
    dictionary['flat_log_probability'] = sampler.get_log_prob( discard=burn_number, thin=thin_number, flat=True )
    
    return dictionary

def samples2dictionary ( samples, parameters ):
    
    dictionary = {}
    
    dictionary['parameters'] = parameters
    dictionary['samples'] = samples
    
    return dictionary

def emcee_distributions_into_lmfit_parameters( emcee_result, prelim_params ):
    """
    Converts emcee fit results into lmfit parameters, taking most likely step as parameter values
    and the half width between 84th and 16th percentile as the standard error.
    
    """
    
    i_max = np.argmax( emcee_result['flat_log_probability'] )
    
    values = np.zeros( len(emcee_result['parameters']) )
    pars = Parameters()
    
    j = 0
    for parameter in emcee_result['parameters']:
        percentiles = np.percentile(emcee_result['flat_chain'][:, j], [16, 50, 84])
        
        # Maximum posterior sample
        # pars.add( parameter, value = emcee_result['flat_chain'][i_max,j] )
        
        # Median
        pars.add( parameter, value = percentiles[1] )
        
        values[j] = pars[parameter].value
        pars[parameter].stderr = abs( percentiles[2] - percentiles[0])/2.
        try:
            pars[parameter].min = prelim_params[parameter].min
            pars[parameter].max = prelim_params[parameter].max
        except KeyError:
            pass
        j += 1
    
    
    return pars, values

def add_fixed_parameters2lmfit( lmfit_params, prelim_params ):
    
    # Add the fixed parameters back into the parameters object
    prelim_params_names = list( prelim_params.keys() )
    
    for parameter in prelim_params_names:
        if not(prelim_params[parameter].vary):
            lmfit_params.add( parameter, value=prelim_params[parameter].value, vary=False )
            lmfit_params[parameter].min = prelim_params[parameter].min
            lmfit_params[parameter].max = prelim_params[parameter].max
        
    return lmfit_params

def emcee_distributions_into_dictionaries( emcee_result ):
    """
    Converts emcee fit results into dictionaries listing medians, the half width between 84th and
    16th percentile as the standard error as well as the distances of the medians from the 16th and 84th
    percentile as asymmetric error bars.
    """
    
    # Index of the most probable sample
    i_max = np.argmax( emcee_result['flat_log_probability'] )
    
    dictionary = {}
    for key in ['medians', 'stderrs', 'upperrs', 'lowerrs', 'max_probabilities']:
        dictionary[key] = {}
    
    for j, parameter in enumerate(emcee_result['parameters']):
        percentiles = np.percentile(emcee_result['flat_chain'][:, j], [16, 50, 84])
        dictionary['medians'][parameter] = percentiles[1]
        dictionary['stderrs'][parameter] = abs( percentiles[2] - percentiles[0])/2.
        dictionary['upperrs'][parameter] = percentiles[2] - percentiles[1]
        dictionary['lowerrs'][parameter] = percentiles[1] - percentiles[0]
        dictionary['max_probabilities'][parameter] = emcee_result['flat_chain'][i_max,j]
    
    return dictionary

def dictionary2array( dictionary ):
    
    keys = list(dictionary.keys())
    array = np.zeros(len(keys))
    for i, key in enumerate(keys):
        array[i] = dictionary[key]
    
    return array

def add_fixed_parameters2dict( parameter_dict, lmfit_parameters ):
    
    variables = list(lmfit_parameters.keys())
    
    for var in variables:
        if not(lmfit_parameters[var].vary):
            parameter_dict[var] = lmfit_parameters[var].value
    
    return parameter_dict

def add_fixed_parameters2dicts( mcmc_dict, prelim_params ):
    
    # Add the fixed parameters back into the parameters object
    prelim_params_names = list( prelim_params.keys() )
    
    for parameter in prelim_params_names:
        if not(prelim_params[parameter].vary):
            for key in ['medians', 'max_probabilities']:
                mcmc_dict[key][parameter] = prelim_params[parameter].value
            for key in ['stderrs', 'upperrs', 'lowerrs']:
                mcmc_dict[key][parameter] = 0.
    
    return mcmc_dict

def samples2medians( samples, parameters ):
    
    dictionary = {}
    for i in range(len(parameters)):
        dictionary[parameters[i]] = np.median(samples[:,i])
    
    return dictionary

def standardized_time( bjd ):
    """
    Turns a bjd array into a time in hours from the beginning of the observation.
    """
    
    return bjd - pr.TIME_0

def pickle_save(directory, ofile, variable):
    """
    Saves the variable under ofile in directory. If the directory does not exist, it creates it.
    """
    if os.path.isdir( directory )==False:
        os.makedirs( directory )
    opath = os.path.join( directory, ofile + '.pkl' )
    ofile = open( opath, 'wb' )
    pickle.dump( variable, ofile )
    ofile.close()
    print( '\nSaved:', opath  )

def T14_Winn( period, a_Rs, Rp_Rs, inclination):
    b = a_Rs * np.cos(np.deg2rad(inclination))
    return period/np.pi * np.arcsin( np.sqrt( (1.+Rp_Rs)**2 - b**2 )/np.sin(np.deg2rad(inclination))/a_Rs )

def T23_Winn( period, a_Rs, Rp_Rs, inclination):
    b = a_Rs * np.cos(np.deg2rad(inclination))
    return period/np.pi * np.arcsin( np.sqrt( (1.-Rp_Rs)**2 - b**2 )/np.sin(np.deg2rad(inclination))/a_Rs )

def pickle_read(fpath):
    """
    Reads a .pkl file from fpath and returns the content.
    """
    ifile = open( fpath + '.pkl', 'rb' )
    z = pickle.load( ifile )
    ifile.close()
    return z

def trim_spectrum(x,y,low,up):
    """   
    Only returns the values of the x and y arrays with x values between low and up. x has to be sorted!
    """
    i_low = np.argmin(abs(x-low))
    i_up = np.argmin(abs(x-up))
    
    return x[i_low:i_up], y[i_low:i_up]

def trim_transmission_spectrum(trans_spec,low,up):
    """   
    Only returns the values of the x and y arrays with x values between low and up. x has to be sorted!
    """
    i_low = np.argmin(abs(trans_spec['Wavelength (um)']-low))
    i_up = np.argmin(abs(trans_spec['Wavelength (um)']-up))
    
    for var in list(trans_spec.keys()):
        trans_spec[var] = trans_spec[var][i_low:i_up]
    
    return trans_spec

def gaussian_ball( parameters, n_walkers, priors, std_unc = None, lc = None ):
    """
    Creates a matrix of n_walkers walker positions for all parameters in emcee format. The walker positions are normally
    distributed around their values in parameters with a standard deviation according to their uncertainties.
    If the uncertainty for a parameter is not given, the standard value given in std_unc is adopted.

    Parameters
    ----------
    parameters : lmfit parameters
        Parameters with initial values and standard deviations the walkers will walk through.
    n_walkers : integer
        Number of walkers in the ensemble
    std_unc : dictionary
        DESCRIPTION. The default is None.
        
    Returns
    -------
    walker_positions : np array
        n_walkers x len(parameters) matrix with all walker positions.
        
    """
    
    m = count_varied_parameters_lmfit(parameters)
    walker_positions = np.zeros(( n_walkers, m ))
    # gaussian_prior_parameters = list_of_priors(priors, 'gaussian')
    parameters_dict = parameters.valuesdict()
    keys = list(parameters_dict.keys())
    
    j = 0
    for key in keys:
        if parameters[key].vary:
            # Prioritize gaussian Priors over standard uncertainties for some parameters at the white light curve
            if priors[key]['prior'] == 'xxx':
                walker_positions[:,j] = priors[key]['sigma'] * np.random.randn(n_walkers)/10. + priors[key]['mu']
            else:
                if (key[:4] == 'Tmid'):
                    try:
                        walker_positions[:,j] = parameters[key].stderr * np.random.randn(n_walkers) + parameters[key].value
                    except TypeError:
                        walker_positions[:,j] = std_unc[key] * np.random.randn(n_walkers) + parameters[key].value
                # Hard code systematic uncertainties
                elif (key == 'sigma_sys_ch1of14'):
                    walker_positions[:,j] = 146. + 22. * np.random.randn(n_walkers) / 10.
                elif (key == 'sigma_sys_ch2of14'):
                    walker_positions[:,j] = 87. + 19. * np.random.randn(n_walkers) / 10.
                elif (key == 'sigma_sys_ch3of14'):
                    walker_positions[:,j] = 71. + 23. * np.random.randn(n_walkers) / 10.
                elif (key == 'sigma_sys_ch4of14'):
                    walker_positions[:,j] = 42. + 27. * np.random.randn(n_walkers) / 10.
                elif (key == 'sigma_sys_ch5of14'):
                    walker_positions[:,j] = 75. + 24. * np.random.randn(n_walkers) / 10.
                elif (key == 'sigma_sys_ch6of14'):
                    walker_positions[:,j] = 96. + 19. * np.random.randn(n_walkers) / 10.
                elif (key == 'sigma_sys_ch7of14'):
                    walker_positions[:,j] = 104. + 24. * np.random.randn(n_walkers) / 10.
                elif (key == 'sigma_sys_ch8of14'):
                    walker_positions[:,j] = 146. + 20. * np.random.randn(n_walkers) / 10.
                elif (key == 'sigma_sys_ch9of14'):
                    walker_positions[:,j] = 98. + 37. * np.random.randn(n_walkers) / 10.
                elif (key == 'sigma_sys_ch10of14'):
                    walker_positions[:,j] = 95. + 46. * np.random.randn(n_walkers) / 10.
                elif (key == 'sigma_sys_ch11of14'):
                    walker_positions[:,j] = 184. + 28. * np.random.randn(n_walkers) / 10.
                elif (key == 'sigma_sys_ch12of14'):
                    walker_positions[:,j] = 146. + 46. * np.random.randn(n_walkers) / 10.
                elif (key == 'sigma_sys_ch13of14'):
                    walker_positions[:,j] = 227. + 33. * np.random.randn(n_walkers) / 10.
                elif (key == 'sigma_sys_ch14of14'):
                    walker_positions[:,j] = 250. + 36. * np.random.randn(n_walkers) / 10.
                elif (key[:13] == 'sigma_sys_NRS'):
                    walker_positions[:,j] = np.random.randn(n_walkers) * 5. / 10. + 120.
                elif (key[:9] == 'sigma_sys'):
                    walker_positions[:,j] = np.random.randn(n_walkers) * 5. / 10. + 200.
                elif (key == 'q2'):
                    # Spectral light curves
                    walker_positions[:,j] = np.random.uniform(size=n_walkers)
                else:
                    try:
                        walker_positions[:,j] = parameters[key].stderr * np.random.randn(n_walkers)/10. + parameters[key].value
                    except TypeError:
                        print(key)
                        walker_positions[:,j] = std_unc[key] * np.random.randn(n_walkers)/10. + parameters[key].value
                walker_positions[:,j] = limit_array_entries_to_interval(walker_positions[:,j], [parameters[key].min, parameters[key].max])
                    # elif (key in ['q1_NRS1', 'q2_NRS1', 'q1_NRS2', 'q2_NRS2']):
                    #     walker_positions[:,j] = np.abs(walker_positions[:,j])
            # # Debugging purposes
            # print(key, walker_positions[:10,j])
            j += 1
    
    return walker_positions

def list_of_priors( priors, prior ):
    parameters = list(priors.keys())
    gaussian_priors = []
    
    for parameter in parameters:
        if priors[parameter]['prior'] == prior:
            gaussian_priors += [parameter]
            
    return gaussian_priors

def limit_array_entries_to_interval( in_array, limits ):
    
    out_array = np.zeros( len(in_array) )
    
    for i in range(len(in_array)):
        out_array[i] = np.min( [np.max([limits[0], in_array[i]]), limits[1]] )
    
    return out_array

def compare_residuals_to_poisson( data, uncertainties, model ):
    return np.std( data - model )/np.median( uncertainties )

def count_varied_parameters_lmfit ( p ):
    """
    Returns the number of varied parameters in lmfit parameters object p.
    """
    
    n_params = 0
    for var in p:
        if (p[var].vary == True):
            n_params += 1
            
    return n_params

def varied_parameters_lmfit ( p ):
    """
    Returns the number of varied parameters in lmfit parameters object p.
    """
    varied_parameters = []
    for var in p:
        if (p[var].vary == True):
            varied_parameters += [var]
            
    return varied_parameters

def sigma_clipping_white( z, fit_params, sigma_cutoff ):
    
    print('Sigma clipping. Number of outliers: ',)
    detectors = list(z.keys())
    
    for nrs in detectors:
        batpar, batmodel = calculateBatmanModel.create_Batman_arrays( z[nrs]['bjd_tdb'] )
        model = calculateBatmanModel.Broad( fit_params.valuesdict(), z[nrs], batpar, batmodel, nrs )
        z[nrs], model = sigma_clipping( z[nrs], model, sigma_cutoff, max_errorbar=0.002 )
        print(nrs + ' ' + str(z[nrs]['outliers']),)
    return z

def sigma_clipping_spec( z, fit_params, broad_fit_params, sigma_cutoff):
    
    # Calculate the model
    batpar, batmodel = calculateBatmanModel.create_Batman_arrays( z['bjd_tdb'] )
    for j in range(len(batpar)):
        batpar[j] = calculateBatmanModel.fix_broad_fit_parameters(batpar[j], broad_fit_params)
    model = calculateBatmanModel.Spec( fit_params.valuesdict(), z, batpar, batmodel )
    
    z, model = sigma_clipping( z, model, sigma_cutoff )
    print( 'Number of outliers: ' + str(z['outliers']) )
    return z

def ith_value2dictionary( in_dictionary, i ):
    
    out_dictionary = {}
    keys = list(in_dictionary.keys())
    remove_these = ['wavelength', 'wavelength_halfwidth', 'chisquare', 'bic']
    for key in remove_these:
        try:
            keys.remove(key)
        except ValueError:
            pass
    for key in keys:
        out_dictionary[key] = in_dictionary[key]['value'][i]
    
    return out_dictionary

def sigma_clipping( data, model, limit, max_errorbar = np.inf ):
    """
    
    Removes outliers > limit sigma away from the model. Additionally removes data points with errorbars
    that are greater than 2%. Add an entry 'outliers' to the data dictionary with the number of removed outliers.
    
    Parameters
    ----------
    data : Dictionary
        Data
    model : numpy array
        Model array that were fit to data['lcVals']
    limit : float, optional
        Remove all data points > limit sigma away from the model. The default is 5..

    Returns
    -------
    data : Dictionary
        The data dictionary with all outliers removed
    model : numpy array
        Model array with all outliers removed
    """
    
    i = 0
    count = 0
    while (i < len(data['lcVals'])):
        if ( abs(data['lcVals'][i] - model[i]) / data['lcUncs'][i] > limit ):
            data = remove_outlier( data, i )
            model = np.delete( model, i )
            count += 1
        elif ( data['lcUncs'][i]/data['lcVals'][i] > max_errorbar ):
            data = remove_outlier( data, i )
            model = np.delete( model, i )
            count += 1
        else:
            i = i + 1
    
    data['outliers'] = count
    
    return data, model

def remove_outlier( dictionary, n ):
    """
    Removes the n-th element of all arrays stored in dictionary that are as long
    as the light curve array.
    """
    
    max_length = len(dictionary['lcVals'])
    
    for k in dictionary.keys():
        try:
            if( len(dictionary[k]) ==  max_length ):
                dictionary[k] = np.delete( dictionary[k], n )
        except TypeError:
            pass
    
    return dictionary

def remove_outlier_trans_spec( trans_spec, n ):
    """
    Removes the n-th element of all arrays stored in dictionary.
    """
    
    for var in list(trans_spec.keys()):
        for key in list(trans_spec[var].keys()):
            trans_spec[var][key] = np.delete( trans_spec[var][key], n )
    
    return trans_spec

def bin_edges_into_middles( edges ):
    """
    Takes array of bin edges and returns an array with one entry less containing the midpoints of the
    bins.
    """
    middles = np.zeros( len(edges) - 1 )
    i = 0
    while( i < len(middles) ):
        middles[i] = (edges[i] + edges[i + 1])/2.
        i += 1

    return middles

def bin_edges_into_halfwidths( edges ):
    """
    Takes array of bin edges and returns an array with one entry less containing the midpoints of the
    bins.
    """
    halfwidths = np.zeros( len(edges) - 1 )
    i = 0
    while( i < len(halfwidths) ):
        halfwidths[i] = (edges[i+1] - edges[i])/2.
        i += 1

    return halfwidths

def array2bin_edges( array ):
    mean_step = np.mean(difference_array(array))
    edges = np.append(array - 0.5*mean_step, array[-1] + 0.5*mean_step)
    return edges

def array_edges( array ):
    n = len(array)
    edges = np.zeros(n-1)
    for i in range(n-1):
        edges[i] = (array[i] + array[i+1])/2.
        
    return edges

def difference_array(array):
    n = len(array)
    diff_array = np.zeros(n - 1)
    
    i = 0
    while i<n-1:
        diff_array[i] = array[i + 1] - array[i]
        i += 1
    
    return diff_array

def array_egdes_by_bin_widths( middles, widths ):
    n = len(middles)
    edges = np.zeros(n + 1)
    
    edges[0] = middles[0] - widths[0]
    
    i = 1
    while (i<=n):
        edges[i] = middles[i-1] + widths[i-1]
        i += 1
    
    return edges

def chisquare( f_measured, sigma_measured, f_modeled ):
    """
    Calculates \sum_{i} \frac {(f_measured_i - f_modeled_i)^2} {sigma_measured_i^2}

    """
    if(len(f_measured) != len(f_modeled)):
        print('Problem: f_measured has a different length than f_modeled')
    elif(len(f_measured) != len(f_modeled)):
        print('Problem: f_measured has a different length than sigma_measured')
    
    chi_squared = 0.
    for i in range(len(f_measured)):
        chi_squared_i = (f_measured[i]-f_modeled[i])**2 / sigma_measured[i]**2
        if not(np.isnan(chi_squared_i)):
            chi_squared += chi_squared_i
        
    return chi_squared

def reduced_chisquare( f_measured, sigma_measured, f_modeled, m ):
    """
    Calculates \sum_{i} \frac {(f_measured_i - f_modeled_i)^2} {sigma_measured_i^2} / (n-m)

    """
    
    if(len(f_measured) != len(f_modeled)):
        print('Problem: f_measured has a different length than f_modeled')
    elif(len(f_measured) != len(f_modeled)):
        print('Problem: f_measured has a different length than sigma_measured')
    
    count = 0       # Count non-nan values in f_measured
    chi_squared = 0.
    for i in range(len(f_measured)):
        chi_squared_i = (f_measured[i]-f_modeled[i])**2 / sigma_measured[i]**2
        if not(np.isnan(chi_squared_i)):
            chi_squared += chi_squared_i
            count += 1
        
    return chi_squared/(float(count) - float(m))

def write_transmission_spectrum_to_csv_file( tr_spec, filename ):
    
    f = open(filename + '.csv','w')
    
    # Write header
    f.write( '# Central wavelength [microns], Half-width of the wavelength bin [microns], \
                (R_p/R_s)^2, 1 sigma upper error bar on (R_p/R_s)^2, 1 sigma lower error bar on (R_p/R_s)^2, \
                Transit mid time [BJD], 1 sigma upper error bar on T_mid [BJD], 1 sigma lower error bar on T_mid [BJD] \n' )
    
    for i in range( len( tr_spec['RpRs']['value'] ) ):
        f.write( '{:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f} \n'.format( \
                tr_spec['wavelength']['value'][i], tr_spec['wavelength']['stderr'][i], \
                tr_spec['RpRs']['value'][i]**2, 2.*tr_spec['RpRs']['value'][i]*tr_spec['RpRs']['upperr'][i], 2.*tr_spec['RpRs']['value'][i]*tr_spec['RpRs']['lowerr'][i], \
                tr_spec['Tmid']['value'][i], tr_spec['Tmid']['upperr'][i], tr_spec['Tmid']['lowerr'][i]    ) )
    
    f.close()

def write_fit_dictionary_to_csv_file( fit_results, filename ):
    
    f = open(filename + '.csv','w')
    
    values = list(fit_results.keys())
    parameters = list( fit_results[values[0]].keys() )
    
    # Write header
    f.write( '# Parameter, Median, 1 sigma upper error, 1 sigma lower error \n' )
    
    for parameter in parameters:
        f.write( parameter + ', {:.8f}, {:.8f}, {:.8f} \n'.format( \
                fit_results['medians'][parameter], fit_results['upperrs'][parameter], fit_results['lowerrs'][parameter]    ) )
    
    f.close()
    
    print('\nSaved:',filename + '.csv')

def write_dictionary_to_csv_file( dictionary, opath ):
    
    f = open(opath + '.csv','w')
    keys = list(dictionary.keys())
    n = len(dictionary[keys[0]])
    m = len(keys)
    
    # Write header
    f.write('# ')
    
    i = 0
    for key in keys:
        f.write(key)
        if (i < len(keys) - 1):
            f.write(', ')
        i += 1
        
    f.write('\n')

    # Write data
    for i in range(n):
        for j in range(m):
            f.write( '{:.32f}'.format(dictionary[keys[j]][i]) )
            if (j < m-1):
                f.write(', ')
        f.write('\n')
    
    f.close()
    
    print('\nSaved:',opath + '.csv')
    
def read_dat( ipath, header_delimiter = ' ', data_delimiter = ' '):
    
    f = open(ipath,'r')
    header = f.readline()
    keys = header.split(header_delimiter)
    
    for i in range(len(keys)):
        keys[i] = keys[i].strip()
    
    dictionary = {}
    data = np.loadtxt( ipath, delimiter=data_delimiter, skiprows=1)

    for j in range(len(data[0,:])):
        dictionary[keys[j]] = data[:,j]

    return dictionary

def read_csv( ipath ):
    
    f = open(ipath + '.csv','r')
    header = f.readline()
    keys = header[2:len(header)-1].split(', ')
    
    dictionary = {}
    try:
        data = np.loadtxt( ipath + '.csv', delimiter=',' )
    except ValueError:
        data = np.loadtxt( ipath + '.csv', delimiter=',', dtype=str )

    for j in range(len(data[0,:])):
        dictionary[keys[j]] = data[:,j]

    return dictionary
    

def scale_uncertainties_broad( z, fit_params, p0, batpar, batmodel ):
    """
    CAUTION, UNTESTED FUNCTION!!!
    """
    n_params = count_varied_parameters_lmfit( p0 )
    model = {}
    reduced_chisquare = {}
    for nrs in z.keys():
        model[nrs], baseline = calculateBatmanModel.Broad( fit_params, z[nrs]['t'], batpar[nrs], batmodel[nrs], nrs )
        reduced_chisquare[nrs] = reduced_chisquare( z[nrs]['lcVals'], z[nrs]['lcUncs'], model[nrs], n_params)
        if reduced_chisquare[nrs] > 1.:
            z[nrs]['lcUncs'] = z[nrs]['lcUncs'] * np.sqrt(reduced_chisquare[nrs])

def bayes2Nsigma_benneke_seager_2013( b ):
    """
    Calculates the sigma significance by which a null hypothesis given a Bayes factor can be ruled out by.
    Equations were taken from Benneke & Seager (2013), eq.s (10) & (11)

    Parameters
    ----------
    b : float
        Bayes factor

    Returns
    -------
    float
        Sigma significance

    """
    y = -np.exp(-1)/b
    
    # For some reason, we have to take the -1 branch of the Lambert W function
    p_value = y / np.real(scipy.special.lambertw(y, -1))
    
    return np.sqrt(2.) * scipy.special.erfinv(1.-p_value)

def bayes2Nsigma_daniel ( bayes_factor ):
    ''' Converts a bayes factor between two models to a probability and
    corresponding Bayesian Sigma .
    Args :
        bayes_factor : the Bayes factor to be converted , usually > 1
    Returns :
        The probability that the model was true given that one of the models
        is correct ( including priors ), and the corresponding Bayesian sigma . '''
    p = 1. / (1 + bayes_factor )
    sigma = -stats . norm . ppf (p /2.)
    return sigma

def extract_spec_lc( data, detector ):
    """
    Extracts the spectroscopic light curves from an xarray object and modifies the format.
    """
    
    dictionary = {}
    n_chan, n_int = np.shape( data[detector + '_light_curves'] )
    print(n_chan, n_int)
    for i in range(n_chan):
        dictionary[str(i)] = {}
        dictionary[str(i)]['bmjd_tdb'] = np.array( data[detector + '_time(BJD_TDB)'] )
        dictionary[str(i)]['bjd_tdb'] = dictionary[str(i)]['bmjd_tdb'] + 2400000.5
        dictionary[str(i)]['lcVals'] = data[detector + '_light_curves'][i,:]
        dictionary[str(i)]['lcUncs'] = data[detector + '_light_curves_err'][i,:]
        dictionary[str(i)]['wavEdgesMicr'] = np.array( [data[detector + '_wavelengths(micron)'][i] - data[detector + '_wavelengths_fullwidth(micron)'][i]/2., data[detector + '_wavelengths(micron)'][i] + data[detector + '_wavelengths_fullwidth(micron)'][i]/2.] )
        dictionary[str(i)]['t'] = standardized_time( dictionary[str(i)]['bjd_tdb'] )
        dictionary[str(i)]['detector'] = detector
    
        # Remove NaNs
        count = 0
        j = 0
        while (j < len(dictionary[str(i)]['lcVals'])):
            if np.isnan(dictionary[str(i)]['lcVals'][j]) or np.isnan(dictionary[str(i)]['lcUncs'][j]) or np.isnan(dictionary[str(i)]['bmjd_tdb'][j]):
                dictionary[str(i)] = remove_outlier( dictionary[str(i)], j )
                count += 1
            else:
                j += 1
        print(i, 'removed outliers because of NaN:', count)
    
    
    return dictionary

def kick_out_nans( x,y ):
    
    x_clean = np.array([])
    y_clean = np.array([])
    
    for i in range(len(x)):
        if not( np.isnan(x[i]) or np.isnan(y[i]) ):
            x_clean = np.append( x_clean, x[i] )
            y_clean = np.append( y_clean, y[i] )
            
    return x_clean, y_clean


def error_weighted_mean( data, uncertainties ):
    """
    Returns the error weighted mean of data with associated uncertainties. Kicks out nan uncertainties.

    Parameters
    ----------
    data : array
    uncertainties : array

    Returns
    -------
    mean : float

    """
    weights = 0.
    mean = 0.
    for i in range(len(data)):
        if not(np.isnan(uncertainties[i])):
            mean += data[i] / uncertainties[i]
            weights += 1./uncertainties[i]
    
    return mean/weights

def remove_nans(dictionary):
    
    # Remove NaNs
    count = 0
    i = 0
    while (i < len(dictionary['lcVals'])):
        if np.isnan(dictionary['lcVals'][i]) or np.isnan(dictionary['lcUncs'][i]) or np.isnan(dictionary['bjd_tdb'][i]):
            dictionary = remove_outlier( dictionary, i )
            count += 1
        else:
            i += 1
    
    return dictionary


def compare_data_and_model_broad( z, fit_params, odir, ofile, emcee_result=None ):
    """
    Takes the broad band light curve z and lmfit fit_params and plots model and data.
    """
    
    detectors = list(z.keys())
    best_params = fit_params.valuesdict()
    
    if ( len(odir) > 0 and os.path.isdir( odir ) == False ):
        os.makedirs( odir )
    opath = os.path.join(odir, ofile)
    
    # Count the number of varied parameters for the reduced chisquare calculation
    n_params = count_varied_parameters_lmfit( fit_params )
    data_alldetectors = np.array([])
    uncertainties_alldetectors = np.array([])
    model_alldetectors = np.array([])
    
    print('Detector, Reduced Chisquares, RMS [ppm]')
    for nrs in detectors:
        batpar, batmodel = calculateBatmanModel.create_Batman_object( z[nrs]['bjd_tdb'] )
        
        model = calculateBatmanModel.Broad( best_params, z[nrs], batpar, batmodel, nrs )
        # print(nrs, compare_residuals_to_poisson( z[nrs]['lcVals'], z[nrs]['lcUncs'], model) )
        chisquare = reduced_chisquare( z[nrs]['lcVals'], z[nrs]['lcUncs'], model, n_params)
        root_mean_square = rms(z[nrs]['lcVals'], model) / 1.0E-06
        print(nrs, chisquare, root_mean_square)
        
        model_alldetectors = np.append(model_alldetectors, model)
        data_alldetectors = np.append(data_alldetectors, z[nrs]['lcVals'])
        uncertainties_alldetectors = np.append(uncertainties_alldetectors, z[nrs]['lcUncs'])
    print('Reduced chi squared:', reduced_chisquare(data_alldetectors, uncertainties_alldetectors, model_alldetectors, n_params) )
    
    if pr.SHOW_PLOTS or pr.SAVE_PLOTS:
        Plot.broad_light_curves_residuals( z, best_params, opath, emcee_result )

def compare_data_and_model_fatman_broad( z, fit_params, odir, ofile, emcee_result=None ):
    """
    Takes the broad band light curve z and lmfit fit_params and plots model and data.
    """
    
    detectors = list(z.keys())
    best_params = fit_params.valuesdict()
    
    if ( len(odir) > 0 and os.path.isdir( odir ) == False ):
        os.makedirs( odir )
    opath = os.path.join(odir, ofile)
    
    # Count the number of varied parameters for the reduced chisquare calculation
    n_params = count_varied_parameters_lmfit( fit_params )
    data_alldetectors = np.array([])
    uncertainties_alldetectors = np.array([])
    model_alldetectors = np.array([])
    
    batpar = {}
    batmodel = {}
    print('Detector, Reduced Chisquares, RMS [ppm]')
    for nrs in detectors:
        
        # First, initialize a batman model for every data point
        batpar[nrs] = []
        batmodel[nrs] = []
        for i in range(len(z[nrs]['bjd_tdb'])):
            bp, bm = calculateBatmanModel.create_Batman_object( z[nrs]['bjd_tdb'][i:i+1] )
            batpar[nrs] += [bp]
            batmodel[nrs] += [bm]
        
        model = calculateBatmanModel.fatman_Broad( best_params, z[nrs], batpar[nrs], batmodel[nrs], nrs )
        # print(nrs, compare_residuals_to_poisson( z[nrs]['lcVals'], z[nrs]['lcUncs'], model) )
        chisquare = reduced_chisquare( z[nrs]['lcVals'], z[nrs]['lcUncs'], model, n_params)
        root_mean_square = rms(z[nrs]['lcVals'], model) / 1.0E-06
        print(nrs, chisquare, root_mean_square)
        
        model_alldetectors = np.append(model_alldetectors, model)
        data_alldetectors = np.append(data_alldetectors, z[nrs]['lcVals'])
        uncertainties_alldetectors = np.append(uncertainties_alldetectors, z[nrs]['lcUncs'])
    print('Reduced chi squared:', reduced_chisquare(data_alldetectors, uncertainties_alldetectors, model_alldetectors, n_params) )
    
    if pr.SHOW_PLOTS or pr.SAVE_PLOTS:
        Plot.broad_light_curves_residuals( z, best_params, opath, emcee_result )
    
def compare_data_and_model_spec( z, fit_params, broad_fit_params, odir, ofile, emcee_result=None ):
    """
    Takes a spectral light curve z and a parameter dictionary and plots model and data
    """
    
    # Count the number of varied parameters for the reduced chisquare calculation
    n_params = len(fit_params)
    
    # Calculate the model
    batpar, batmodel = calculateBatmanModel.create_Batman_object( z['bjd_tdb'] )
    batpar = calculateBatmanModel.fix_broad_fit_parameters( batpar, broad_fit_params )
    
    model = calculateBatmanModel.Spec( fit_params, z, batpar, batmodel )
    
    print('Residual scatter / Poisson distribution:', compare_residuals_to_poisson( z['lcVals'], z['lcUncs'], model) )
    chi2 = reduced_chisquare( z['lcVals'], z['lcUncs'], model, n_params)
    
    print('Reduced Chisquare (Original error bars):', chi2)
    
    opath = os.path.join(odir, ofile)
    if pr.SAVE_PLOTS or pr.SHOW_PLOTS:
        Plot.spectral_light_curve( z, fit_params, broad_fit_params, opath, emcee_result )
    
    # Allen plot
    
    # Plot.allen( (model-z['lcVals'])/model, opath )
    
    return chi2

def calculate_binned_errorbars( errors, indices, n ):
    """
    Calculates the root of the squares of all errors for each bin and divides by the number of data points
    in each bin.
    
    Parameters
    ----------
    errors : np array
        Original, unbinned error bars
    indices : np array
        Indices listing which bin each unbinned error bar goes into
    n : int
        Number of bins
        
    Returns
    -------
    binned_errors : np.array
    
    """
    
    binned_errors = np.zeros( n ) + float('nan')
    
    for i in range(n):
        count = 0
        binned_error = 0.
        for j in range(len(errors)):
            if (indices[j]-1 == i):
                binned_error += errors[j]**2
                count += 1
        if count>0:
            binned_errors[i] = np.sqrt(binned_error) / float(count)
    
    return binned_errors

def gaussian_error_propagation( errors ):
    
    final_error = 0.
    for error in errors:
        final_error += error**2
    
    return np.sqrt(final_error)/len(errors)

def prepare_transmission_dictionary( fit_parameters, nchan, only_varied_params=False, outliers=False, asymmetric_errors=False, bic=False ):
    """
    Write down the transmission spectrum's wavelengths, chisquare values and fit parameters
    including uncertainties into a dictionary for later plotting.
    """
    
    parameter_names = list(fit_parameters.keys())
    trans_spec = {}
    
    for var in ['wavelength', 'wavelength_halfwidth', 'chisquare']:
        trans_spec[var] = np.zeros(nchan) + float('NaN')
    
    if bic:
        trans_spec['bic'] = np.zeros(nchan) + float('NaN')
    
    if only_varied_params:
        for var in parameter_names:
            if fit_parameters[var].vary:
                trans_spec[var] = {}
                trans_spec[var]['value'] = np.zeros(nchan) + float('NaN')
                trans_spec[var]['stderr'] = np.zeros(nchan) + float('NaN')
                if asymmetric_errors:
                    trans_spec[var]['upperr'] = np.zeros(nchan) + float('NaN')
                    trans_spec[var]['lowerr'] = np.zeros(nchan) + float('NaN')
    else:
        for var in parameter_names:
            trans_spec[var] = {}
            trans_spec[var]['value'] = np.zeros(nchan) + float('NaN')
            trans_spec[var]['stderr'] = np.zeros(nchan) + float('NaN')
            if asymmetric_errors:
                trans_spec[var]['upperr'] = np.zeros(nchan) + float('NaN')
                trans_spec[var]['lowerr'] = np.zeros(nchan) + float('NaN')
    
    if outliers:
        trans_spec['outliers'] = {}
        trans_spec['outliers']['value'] = np.zeros(nchan, dtype=int)
    
    return trans_spec


def add_lmfit_params_2_transmission_spectrum( trans_spec, i, wavelength_edges, lmfit_params ):
    
    trans_spec['wavelength'][i] = np.sum(wavelength_edges) / 2.
    trans_spec['wavelength_halfwidth'][i] = (wavelength_edges[1] - wavelength_edges[0]) / 2.
    
    variables = list(lmfit_params.keys())
    for var in variables:
        trans_spec[var]['value'][i] = lmfit_params[var].value
        trans_spec[var]['stderr'][i] = lmfit_params[var].stderr
        
    return trans_spec

def rms( data, model ):
    n = len(data)
    rms = 0.
    for i in range(n):
        rms += (data[i]-model[i])**2
    return np.sqrt(rms/float(n))

def rms_allen( residuals ):
    rms, stderr, binsz = computeRMS(residuals, binstep=1)
    return binsz, rms, stderr

def computeRMS(data, maxnbins=None, binstep=1, isrmserr=False):
    """
    COMPUTE ROOT-MEAN-SQUARE AND STANDARD ERROR OF DATA FOR VARIOUS BIN SIZES
    Taken from POET: https://github.com/kevin218/POET/blob/master/code/lib/correlated_noise.py
    """
    # data    = fit.normresiduals
    # maxnbin = maximum # of bins
    # binstep = Bin step size
    
    # bin data into multiple bin sizes
    npts = data.size
    if maxnbins is None:
        maxnbins = npts / 10.
    binsz = np.arange(1, maxnbins + binstep, step=binstep, dtype=int)
    nbins = np.zeros(binsz.size, dtype=int)
    rms = np.zeros(binsz.size)
    rmserr = np.zeros(binsz.size)
    for i in range(binsz.size):
        nbins[i] = int(np.floor(data.size / binsz[i]))
        bindata = np.zeros(nbins[i], dtype=float)
        # bin data
        # ADDED INTEGER CONVERSION, mh 01/21/12
        for j in range(nbins[i]):
            bindata[j] = data[j * binsz[i]:(j + 1) * binsz[i]].mean()
        # get rms
        rms[i] = np.sqrt(np.mean(bindata ** 2))
        rmserr[i] = rms[i] / np.sqrt(2. * nbins[i])
    # expected for white noise (WINN 2008, PONT 2006)
    stderr = (data.std() / np.sqrt(binsz)) * np.sqrt(nbins / (nbins - 1.))
    if isrmserr is True:
        return rms, stderr, binsz, rmserr
    else:
        return rms, stderr, binsz

def read_spec1d( f1, f2 ):
    """
    Reads in the time series of 1D spectra.
    """
    ds = {}
        
    print( '\nReading in 1D spectra for each detector:\n{0}\n{1}\n'.format( f1, f2 ) )
    ds['NRS1'] = xr.open_dataset( f1 )
    ds['NRS2'] = xr.open_dataset( f2 )
    spec1d = {}
    detectors = [ 'NRS1', 'NRS2' ]
    for k in detectors:
        spec1d[k] = { 'wavMicr':ds[k]['wavelength'].values,
                      'bjd':ds[k]['time'].values.flatten(),
                      'fluxVals':ds[k]['flux'].values,
                      'fluxUncs':ds[k]['flux_error'].values,
                      'x_shift':ds[k]['x_shift'].values.flatten(),
                      'y_shift':ds[k]['y_shift'].values.flatten() }
        ds[k].close()
    return spec1d

def transit_cutout_white( inp ):
    
    Tmid = pr.TIME_0 + pr.TRANSIT_MID_TIME
    outp = {}
    detectors = list(inp.keys())
    for k in detectors:
        bjd = inp[k]['bjd_tdb']
        thr = 24.*( bjd - Tmid )
        ixs = ( thr>=-pr.CUTOUT_HALFTIME )*( thr<=pr.CUTOUT_HALFTIME )

        outp[k] = {}
        outp[k]['bjd_tdb'] = inp[k]['bjd_tdb'][ixs]
        outp[k]['x_shift'] = inp[k]['x_shift'][ixs]
        outp[k]['y_shift'] = inp[k]['y_shift'][ixs]
        outp[k]['wavEdgesMicr'] = inp[k]['wavEdgesMicr']
        if inp[k]['lcVals'].ndim==1:
            outp[k]['lcVals'] = inp[k]['lcVals'][ixs]
            outp[k]['lcUncs'] = inp[k]['lcUncs'][ixs]
        else:
            # This shouldn't happen because if NRS1 is one of the
            # keys, this should be the white light curve; but this
            # is sloppy formatting and might change in the future.
            pdb.set_trace()
    return outp

def transit_and_eclipses_white( inp ):
    
    Tmid = pr.TIME_0 + pr.TRANSIT_MID_TIME
    outp = {}
    detectors = list(inp.keys())
    
    for k in detectors:
        bjd = inp[k]['bjd_tdb']
        thr = 24.*( bjd - Tmid )
        n = len(bjd)
        
        outp[k] = {}
        for key in ['bjd_tdb', 'x_shift', 'y_shift', 'lcVals', 'lcUncs']:
            try:
                outp[k][key] = np.array([])
            except KeyError:
                pass
        for i in range(n):
            if (bjd[i]>2459866.96668 and bjd[i]<2459867.048) or (bjd[i]>2459868.23894 and bjd[i]<2459868.32446) or (thr[i]>-pr.CUTOUT_HALFTIME and thr[i]<pr.CUTOUT_HALFTIME):
                for key in ['bjd_tdb', 'x_shift', 'y_shift', 'lcVals', 'lcUncs']:
                    try:
                        outp[k][key] = np.append( outp[k][key], inp[k][key][i])
                    except KeyError:
                        pass
        try:
            outp[k]['wavEdgesMicr'] = inp[k]['wavEdgesMicr']
        except KeyError:
            pass
        try:
            outp[k]['detector'] = inp[k]['detector']
        except KeyError:
            pass
        
    return outp

def transit_and_eclipses_spec( inp ):
    Tmid = pr.TIME_0 + pr.TRANSIT_MID_TIME
    outp = {}
    
    # This is an awkward format because I store time stamps for
    # every light curve separately, because the time stamps can
    # differ slightly between NRS1 and NRS2.
    nint, nchan = np.shape( inp['lcVals'] )
    for key in ['bjd_tdb', 'x_shift', 'y_shift', 'lcVals', 'lcUncs']:
        outp[key] = []
    for key in ['wavEdgesMicr', 'detector', 'npix']:
        outp[key] = inp[key]
    
    for i in range( nchan ):
        bjd = inp['bjd_tdb'][:,i]
        thr = 24*( bjd - Tmid )
        n = len(bjd)
        # ixs = ( thr>=-pr.CUTOUT_HALFTIME )*( thr<=pr.CUTOUT_HALFTIME )
        
        for key in ['bjd_tdb', 'x_shift', 'y_shift', 'lcVals', 'lcUncs']:
            tmp = []
            for j in range(n):
                if (bjd[j]>2459866.96668 and bjd[j]<2459867.048) or (bjd[j]>2459868.23894 and bjd[j]<2459868.32446) or (thr[j]>-pr.CUTOUT_HALFTIME and thr[j]<pr.CUTOUT_HALFTIME):
                    tmp += [inp[key][j,i]]
            outp[key] += [ tmp ]
    
    for key in ['bjd_tdb', 'x_shift', 'y_shift', 'lcVals', 'lcUncs']:
        outp[key] = np.column_stack( outp[key] )
    
    return outp

def transit_cutout_spec( inp ):
    Tmid = pr.TIME_0 + pr.TRANSIT_MID_TIME
    outp = {}
    
    # This is an awkward format because I store time stamps for
    # every light curve separately, because the time stamps can
    # differ slightly between NRS1 and NRS2.
    nint, nchan = np.shape( inp['lcVals'] )
    outp['bjd_tdb'] = []
    outp['x_shift'] = []
    outp['y_shift'] = []
    outp['lcVals'] = []
    outp['lcUncs'] = []
    outp['wavEdgesMicr'] = inp['wavEdgesMicr']
    outp['detector'] = inp['detector']
    outp['npix'] = inp['npix']
    
    for i in range( nchan ):
        bjd = inp['bjd_tdb'][:,i]
        thr = 24*( bjd - Tmid )
        ixs = ( thr>=-pr.CUTOUT_HALFTIME )*( thr<=pr.CUTOUT_HALFTIME )
        
        outp['bjd_tdb'] += [ inp['bjd_tdb'][ixs,i] ]
        outp['x_shift'] += [ inp['x_shift'][ixs,i] ]
        outp['y_shift'] += [ inp['y_shift'][ixs,i] ]
        outp['lcVals'] += [ inp['lcVals'][ixs,i] ]
        outp['lcUncs'] += [ inp['lcUncs'][ixs,i] ]
    outp['bjd_tdb'] = np.column_stack( outp['bjd_tdb'] )
    outp['x_shift'] = np.column_stack( outp['x_shift'] )
    outp['y_shift'] = np.column_stack( outp['y_shift'] )
    outp['lcVals']  = np.column_stack( outp['lcVals'] )
    outp['lcUncs']  = np.column_stack( outp['lcUncs'] )
    
    return outp

def integrateFlux( specVals, specUncs, dispIxs, mask_column = 'None' ):
    """
    Integrates the flux between wavelengths defined by coordinates
    along the dispersion axis to produce a light curve with 
    associated measurement uncertainties.
    
    specVals = N x M array containing time series of 1D spectra.
    specUncs = N x M array containing 1D spectra uncertainties.
    dispIxs = [ ix1, ix2 ]
    
    where:
       N = the number of spectra.
       M = the number of pixels along the dispersion axis.
     ix1 = the pixel coordinate of the lower bound along the dispersion axis.
     ix2 = the pixel coordinate of the upper bound along the dispersion axis.
    
    """
    ixL = dispIxs[0]
    ixU = dispIxs[1]
    fluxes = specVals[:,:]
    uncertainties = specUncs[:,:]
    
    # Mask weird pixel column
    try:
        fluxes[:,mask_column] = np.zeros(len(fluxes[:,0]))
        uncertainties[:,mask_column] = np.zeros(len(uncertainties[:,0]))
    except IndexError:
        pass
    
    lcVals = np.sum( fluxes[:,ixL:ixU+1], axis=1 )
    lcUncs = np.sqrt( np.sum( uncertainties[:,ixL:ixU+1]**2., axis=1 ) )
    # norm = np.median( lcVals[-15:] )
    # lcVals /= norm
    # lcUncs /= norm
    return lcVals, lcUncs


def normalize_lightcurve( fluxes, uncertainties ):
    
    factor = np.median( fluxes[:150] )
    
    return np.array(fluxes) / factor, np.array(uncertainties) / factor

def get_opath( dset ):
    if dset=='FIREFly_20221018':
        opath_white = pr.WHITELC_FPATH_FFLY20221018
        opath_spec = pr.SPECLCS_FPATH_FFLY20221018
    elif dset=='FIREFly_20221029':
        opath_white = pr.WHITELC_FPATH_FFLY20221029
        opath_spec = pr.SPECLCS_FPATH_FFLY20221029
    elif dset=='FIREFly_20230113':
        opath_white = pr.WHITELC_FPATH_FFLY20230113
        opath_spec = pr.SPECLCS_FPATH_FFLY20230113
    elif dset=='FIREFly_20230213':
        opath_white = pr.WHITELC_FPATH_FFLY20230213
        opath_spec = pr.SPECLCS_FPATH_FFLY20230213
    else:
        pdb.set_trace()
    for opath in [ opath_white, opath_spec ]:
        odir = os.path.dirname( opath )
        if os.path.isdir( odir )==False:
            os.makedirs( odir )
    return opath_white, opath_spec

def calculate_Tmid_anomaly( trans_spec, broad_fit_params ):
    
    n = len(trans_spec['Tmid']['value'])
    
    unit = 86400.       # Seconds/day
    
    for i in range(n):
        if (trans_spec['wavelength']['value'][i] < 3.8):
            trans_spec['Tmid']['value'][i] = (trans_spec['Tmid']['value'][i] - broad_fit_params['Tmid_NRS1']) * unit
        else:
            trans_spec['Tmid']['value'][i] = (trans_spec['Tmid']['value'][i] - broad_fit_params['Tmid_NRS2']) * unit
    
    # trans_spec['Tmid']['stderr'] = trans_spec['Tmid']['stderr'] * unit
    trans_spec['Tmid']['upperr'] = trans_spec['Tmid']['upperr'] * unit
    trans_spec['Tmid']['lowerr'] = trans_spec['Tmid']['lowerr'] * unit
    
    return trans_spec


def calculate_planetary_over_stellar_flux( wavelength, Ts, Tp, rp_rs ):
    
    return (np.exp( constants.h * constants.c / (wavelength*constants.k*Ts) ) - 1.) / \
        (np.exp( constants.h * constants.c / (wavelength*constants.k*Tp) ) - 1.) * rp_rs**2

def calculate_planetary_flux( ft_night, fp_fs, bjd, Tmid ):
    """
    Calculates theoretical planetary and stellar fluxes of a light curve using fp/fs values measured for the planet's day and night sides.
    The model is a simple cosine whose frequency is derived from the planet's orbital period.
    
    Parameters
    ----------
    ft_night : float
        Measurement of the total flux of the planet's night side and the star.
    fp_fs : dictionary
        Planetary over stellar flux for both planet's day and night sides
    bjd : array
        Times to which the planetary flux component shall be calculated for
    Tmid : float
        Measured transit mid time the cosine shall have a minimum at

    Returns
    -------
    fp : array
        Planetary fluxes in times bjd
    fs : float
        Stellar flux component constant in time
        
    """
    
    t0 = bjd - Tmid
    omega = 2.*np.pi/pr.ORBITAL_PERIOD
    
    A = ft_night * ( fp_fs['day'] - fp_fs['night'] ) / ( 2. + 2.*fp_fs['night'] )
    B = ft_night * ( fp_fs['day'] + fp_fs['night'] ) / ( 2. + 2.*fp_fs['night'] )
    
    return B - A * np.cos( omega * t0 ), ft_night / ( 1. + fp_fs['night'] )

def partial_phasecurve( A, B, bjd, Tcold ) :
    t0 = bjd - Tcold
    omega = 2.*np.pi/pr.ORBITAL_PERIOD['median']
    
    return B - A * np.cos( omega * t0 )


def partial_phasecurve_fpfs( fp_fs_day, fp_fs_night, bjd, Tcold ) :
    """
    fp_fs always in permill 
    """
    
    t0 = bjd - Tcold
    omega = 2.*np.pi/pr.ORBITAL_PERIOD['median']
    
    return 0.5 * ( fp_fs_day/1.0E3 + fp_fs_night/1.0E3 - (fp_fs_day/1.0E3 - fp_fs_night/1.0E3 ) * np.cos( omega * t0 ) )

def calculate_phasecurve_coefficients( fp_fs_day, fp_fs_night ):
    A = 0.5 * (fp_fs_day/1.0E3 - fp_fs_night/1.0E3 )
    B = 0.5 * ( fp_fs_day/1.0E3 + fp_fs_night/1.0E3)
    return A, B/A

def make_prior_gaussian( mu, sigma ):
    
    dictionary = {}
    dictionary['prior'] = 'gaussian'
    dictionary['mu'] = mu
    dictionary['sigma'] = sigma
    
    return dictionary

def make_prior_boxcar( lower_limit, upper_limit ):
    
    dictionary = {}
    dictionary['prior'] = 'boxcar'
    dictionary['lower_limit'] = lower_limit
    dictionary['upper_limit'] = upper_limit
    
    return dictionary

def evaluate_priors( priors, pars_dict ):
    log_prior = 0.
    parameter_names = list(pars_dict.keys())
    for parameter in parameter_names:
        if priors[parameter]['prior'] == 'gaussian':
            log_prior += log_gaussian( pars_dict[parameter], priors[parameter]['mu'], priors[parameter]['sigma'] )
        elif priors[parameter]['prior'] == 'boxcar':
            log_prior += log_boxcar( pars_dict[parameter], priors[parameter]['lower_limit'], priors[parameter]['upper_limit'])
    
    return log_prior
    
def percentile_indeces( array, percentiles ):
    n = len(percentiles)
    values = np.percentile(array, percentiles, method='closest_observation')
    # print(values)
    indeces = np.zeros(n, dtype=int)
    for i in range(n):
        indeces[i] = np.where( array == values[i])[0]
    
    return indeces

def subtract_phase_curve_from_lightcurves( white_lc, params ):
    
    corrected_lcs = {}
    detectors = list(white_lc.keys())
    
    for nrs in detectors:
        corrected_lcs[nrs] = {}
        
        A = (params['fpfs_day_' + nrs] - params['fpfs_night_' + nrs]) / 2.
        B = (params['fpfs_day_' + nrs] + params['fpfs_night_' + nrs]) / 2.
         
        phase_curve = params['baseline_' + nrs] * (B + A * np.cos(2.*np.pi*white_lc[nrs]['t']/pr.ORBITAL_PERIOD['median'] + params['phi_' + nrs]))
        
        # Data to keep
        for var in ['bjd_tdb', 'lcVals', 'lcUncs']:
            corrected_lcs[nrs][var] = white_lc[nrs][var]
            
        corrected_lcs[nrs]['lcVals'] -= phase_curve
        
    return corrected_lcs

def twoD_lc2coefficients( t, rprs, w ):
    
    model = {'wavelength' : w, 'R0' : np.zeros(len(w)), 'R1' : np.zeros(len(w)), 'R2' : np.zeros(len(w))}
    n, m = rprs.shape
    t_days = t/3600./24.
    d = []
    for j in range(m):
        # a0, a1, a2, a3, a4, a5, a6, a7, a8, model['R2'][j], model['R1'][j], model['R0'][j] = np.polyfit( t_days, rprs[:,j], 11)
        model['R2'][j], model['R1'][j], model['R0'][j] = np.polyfit( t_days, rprs[:,j], 2)
        model_rprs = model['R2'][j] * t_days**2 + model['R1'][j] * t_days + model['R0'][j]
        d += [ max( abs((rprs[:,j]-model_rprs))/rprs[:,j] * 100. ) ]
    # print(max(d))
    return model

def simultaneous_mcmc2trans_spec( mcmc_result, lc ):
    
    parameters = list(mcmc_result['medians'].keys())
    
    # Find out the number of channels
    for parameter in parameters:
        separated = parameter.split('_')
        if len(separated) > 1:
            ch = separated[-1]
            break
    
    # Create a list of all parameters that per channel
    n_channels = ch.split('of')[-1]
    spectroscopic_parameters = []
    for parameter in parameters:
        separated = parameter.split('_')
        if separated[-1] == 'ch1of' + n_channels:
            separated_more = parameter.split('_ch1of')
            spectroscopic_parameters += [separated_more[0]]
    
    # Create the dictionary
    n_channels = int(n_channels)
    trans_spec = {}
    for key in ['wavelength', 'wavelength_halfwidth']:
        trans_spec[key] = np.zeros(n_channels)
        
    for i, channel in enumerate(list(lc.keys())):
        wavelength_edges = lc[channel]['wavEdgesMicr']
        trans_spec['wavelength'][i] = np.mean(wavelength_edges)
        trans_spec['wavelength_halfwidth'][i] = (wavelength_edges[1]-wavelength_edges[0])/2.
    
    for spec_parameter in spectroscopic_parameters:
        # Prepare dictionary for each parameter
        trans_spec[spec_parameter] = {}
        for key in ['value', 'stderr', 'upperr', 'lowerr']:
            trans_spec[spec_parameter][key] = np.zeros( n_channels )
    
    for i in range(n_channels):
        for spec_parameter in spectroscopic_parameters:
            orig_key = spec_parameter + '_ch' + str(i+1) + 'of' + str(n_channels)
            trans_spec[spec_parameter]['value'][i] = mcmc_result['medians'][orig_key]
            for key in ['stderr', 'upperr', 'lowerr']:
                trans_spec[spec_parameter][key][i] = mcmc_result[key + 's'][orig_key]
    
    return trans_spec

def reduce_cornerplot( mcmc_result, plot_params ):
    
    orig_params = mcmc_result['parameters']
    
    # Reduce corner plot to manageable size
    plot_chain = { 'parameters' : plot_params, 'flat_chain' : np.zeros( (len(mcmc_result['flat_log_probability']), len(plot_params)) ), 'flat_log_probability' : mcmc_result['flat_log_probability'] }
    n = len(mcmc_result['flat_log_probability'])
    
    for i, param in enumerate(plot_params):
        
        # Find index in original samples
        j = 0
        while j<len(orig_params):
            if orig_params[j] == param:
                break
            else:
                j += 1
        
        if j == len(orig_params):
            plot_chain['flat_chain'][:,i] = np.zeros(n) + np.random.randn(n)*1.0E-10
        else:
            plot_chain['flat_chain'][:,i] = mcmc_result['flat_chain'][:,j]
    
    return plot_chain

