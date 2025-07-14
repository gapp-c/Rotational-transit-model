import os
import numpy as np

"""
Definitions for various parameters and file paths used throughout the analysis.

Edit as needed according to your own setup.

"""

######### LITERATURE VALUES FOR WASP-121b #########

# STELLAR PARAMETERS

METALLICITY = 0.13                                  # M/H [dex] Mikal-Evans et al. 2019
EFFECTIVE_TEMPERATURE = 6459.                       # T_eff [K] Delrez et al. 2016
LOG_GRAVITY = 4.23431                               # log(g) [log10(cm/s**2)] TICv8

# PLANETARY PARAMETERS

# a/R_s: Bourrier et al. 2020
SEMIMAJOR_AXIS_OVER_STELLAR_RADIUS = {}
SEMIMAJOR_AXIS_OVER_STELLAR_RADIUS['median'] = 3.8131
SEMIMAJOR_AXIS_OVER_STELLAR_RADIUS['upperr'] = 0.0075
SEMIMAJOR_AXIS_OVER_STELLAR_RADIUS['lowerr'] = 0.006

# P [days]: Bourrier et al. 2020
ORBITAL_PERIOD = {}
ORBITAL_PERIOD['median'] = 1.27492504
ORBITAL_PERIOD['upperr'] = 0.00000015
ORBITAL_PERIOD['lowerr'] = 0.00000014

# Transit durations [hours]: Bourrier et al. 2020
T14 = 2.9059
T23 = 2.2639

# inc [degrees]: Bourrier et al. 2020
ORBITAL_INCLINATION = {}
ORBITAL_INCLINATION['median'] = 88.49
ORBITAL_INCLINATION['sigma'] = 0.16

ORBITAL_ECCENTRICITY = 0.                           # e [degrees]
ARGUMENT_OF_PERIAPSIS = 90.                         # [degrees]

# Planetary over stellar radii
PHASECURVE_RP_RS = {}
PHASECURVE_RP_RS['NRS1'] = 0.122551
PHASECURVE_RP_RS['NRS2'] = 0.123668

######### INITIAL GUESSES FOR LIGHT CURVE MODEL PARAMETERS ############

TIME_0 = 2459867.                       # Based on light curve inspection
TRANSIT_MID_TIME = 0.6436               # Time of the transit - TIME_0
BASELINE_C0 = 0.997
BASELINE_C1 = 0.1
BASELINE_C2 = 0.0

########## ANALYSIS PARAMETERS ###########

CUTOUT_HALFTIME = 3.501                   # Transit duration ~ 2.91 hours
NUMBER_OF_PIXEL_COLUMNS = { 'NRS1' : 244, 'NRS2' : 237}
SIGMA_CLIPPING = np.inf
BASELINE_BROAD = 'linear'
BASELINE_SPEC = 'linear'

IMPACT_PARAMETER = True
FIT_ORBIT_SPEC = False
ROTATION_1 = True
ROTATION_2 = True
FIT_ROTATION_2_SPEC = True
ROTATION_2_NEGATIVE = False
KIPPING_LD_PARAMETERS = False
FIT_1ST_LD_PARAMETER = True
FIT_2ND_LD_PARAMETER = True
LDC_PRIORS = False
FIT_TMID_SPEC = True               # Fit for transit midtimes in the spectroscopic light curves?
FIT_TMID_DETECTORS = False          # Fit for different transit midtimes between NRS1 and NRS2? NOT TESTED
ADD_SYSTEMATIC_UNCERTAINTY = True
CATWOMAN = False

########## TECHNICAL PARAMETERS ###########

SHOW_PLOTS = True
SAVE_PLOTS = True
USE_ASTRONODES = False
NUMBER_OF_CORES = 20              # Only relevant when using Astronodes
LIVE_POINTS = 500                 # Nested sampling

######### FILEPATHS & FILENAMES #########

# INPUT

if USE_ASTRONODES:
    DATA_DIR = '/data/beegfs/astro-storage/groups/kreidberg/_ongoing_to_mpcdf/gapp/'
    WORK_DIR = os.path.join( DATA_DIR, 'WASP-121b/' )
    EXOTIC_DATA_DIR = os.path.join( DATA_DIR, '.exotic-ld_data/' )
else:
    HOME_DIR = os.path.expanduser( '~' )
    WORK_DIR = os.path.dirname(os.getcwd())
    EXOTIC_DATA_DIR = os.path.join( HOME_DIR, '.exotic-ld_data/' )

SPECTRA1D_DIR = os.path.join( WORK_DIR, 'data_reduced/xarray_formatted/' )
EUREKA_DIR = os.path.join( WORK_DIR, 'eureka' )
MODEL_DIR = os.path.join( WORK_DIR, 'models_david/' )

SPEC1D_NRS1 = 'spec1d_FIREFly_20230113_NRS1.nc'
SPEC1D_NRS2 = 'spec1d_FIREFly_20230113_NRS2.nc'

THROUGHPUT_FILE = 'custom-throughput.txt'

# OUTPUT

ASTRONODE_DIR = os.path.join( WORK_DIR, 'astronode_runs/' )
if USE_ASTRONODES:
    RESULTS_DIR = os.path.join( WORK_DIR, os.getcwd().split('/')[-1] )
else:
    RESULTS_DIR = os.path.join( WORK_DIR, '250429' )

if KIPPING_LD_PARAMETERS:
    RESULTS_DIR += '_kipping'

if ROTATION_2:
    RESULTS_DIR += '_quadratic'
elif ROTATION_1:
    RESULTS_DIR += '_linear'
else:
    RESULTS_DIR += '_translational'

if LDC_PRIORS:
    RESULTS_DIR += '_ldc-priors'

if ROTATION_2_NEGATIVE:
    RESULTS_DIR += '_R2-negative'

if not(FIT_ROTATION_2_SPEC):
    RESULTS_DIR += '_R2-tied'

CUSTOM_INPUT_DIR = os.path.join( WORK_DIR, 'custom_input/' )

BROAD_DIR = os.path.join( RESULTS_DIR, 'white/' )
SPEC_DIR = os.path.join( RESULTS_DIR, 'spec/' )
MOLECULES = ['co']

TRANS_DIR = 'transmission_spectrum/'
PRELIM_DIR = 'leastsq/'
MCMC_DIR = 'mcmc/'
NESTED_DIR = 'nested/'

LD_FILE = 'limb-darkening-coefficients'
LC_FILE = 'lc'
LC_EDITED_FILE = LC_FILE + '_sigma-clipped'

FIT_FILE = 'fit'
CHAIN_FILE = 'chain'
SAMPLES_FILE = 'samples'
PARAMETERS_FILE = 'parameters'
TRANS_FILE = 'trans_spec'
