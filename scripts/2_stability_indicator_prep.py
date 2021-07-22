import pandas as pd 
import numpy as np
import os
import sys

sys.path.append('./')

from utils.stability_indicators import calc_rocof, make_frequency_data_hdf


# Time zones of frequency recordings
tzs = {'CE':'CET', 'Nordic':'Europe/Helsinki', 'GB':'GB'}

# Datetime parameters for output data generation
start = pd.Timestamp('2015-01-01 00:00:00', tz='UTC')
end = pd.Timestamp('2019-12-31 00:00:00', tz='UTC')
time_resol = pd.Timedelta('1H')

# Pre-processed frequency csv files
frequency_csv_folder = '../Frequency_data_base/' 
tso_names = {'GB': 'Nationalgrid', 'CE': 'TransnetBW', 'Nordic': 'Fingrid' } 
 
# HDF frequency files (for faster access than csv files)
frequency_hdf_folder = {'GB': '../Frequency_data_preparation/Nationalgrid/',
                        'CE': '../Frequency_data_preparation/TransnetBW/',
                        'Nordic': '../Frequency_data_preparation/Fingrid/' } 

# Nan treatment
skip_hour_with_nan = True

# Parameters for rocof estimation
smooth_windows =  {'CE': 60, 'GB': 60, 'Nordic':30}
lookup_windows =   {'CE': 60, 'GB': 60, 'Nordic':30}


for area in ['GB', 'CE', 'Nordic']:
    
    print('\n######', area, '######')
    
    # If not existent, create HDF file from csv files 
    # (for faster access when trying out things)
    hdf_file = make_frequency_data_hdf(frequency_csv_folder, tso_names[area], frequency_hdf_folder[area],
                                       start, end, tzs[area])
    
    # Output data folder
    folder = './data/{}/'.format(area)
    if not os.path.exists(folder):
        os.makedirs(folder)    
    
    # Load frequency data 
    freq = pd.read_hdf(hdf_file).loc[start:end]
    freq = freq -50

    # Setup datetime index for output data
    index = pd.date_range(start, end, freq=time_resol, tz='UTC').tz_convert(tzs[area])
    if os.path.exists(folder+'outputs.h5'):
        outputs= pd.read_hdf(folder+'outputs.h5')           
    else:
        outputs = pd.DataFrame(index= index)
        
    # Extract stability indicators  
    print('Extracting stability indicators ...')
    outputs['f_integral'] = freq.groupby(pd.Grouper(freq='1H')).sum()
    outputs['f_ext'] = freq.groupby(pd.Grouper(freq='1H')).apply(lambda x: x[x.abs().idxmax()] if x.notnull().any() else np.nan)
    outputs['f_rocof'] = calc_rocof(freq, smooth_windows[area], lookup_windows[area])
    outputs['f_msd'] = (freq**2).groupby(pd.Grouper(freq='1H')).mean()
    
    # Set hour to NaN if frequency contains at least one NaN in that hour
    if skip_hour_with_nan==True:
        hours_with_nans = freq.groupby(pd.Grouper(freq='1H')).apply(lambda x: x.isnull().any())
        outputs.loc[hours_with_nans]=np.nan
    
    # Save data 
    outputs.to_hdf(folder+'outputs.h5', key='df')
    
    # Save outputs also for "country"-areas
    if area=='CE':
        folder = './data/{}/'.format('DE')
        if not os.path.exists(folder):
            os.makedirs(folder)   
        outputs.to_hdf(folder+'outputs.h5', key='df')     

        folder = './data/{}/'.format('CH')
        if not os.path.exists(folder):
            os.makedirs(folder)   
        outputs.to_hdf(folder+'outputs.h5', key='df')   
          
    if area=='Nordic':
        folder = './data/{}/'.format('SE')
        if not os.path.exists(folder):
            os.makedirs(folder)   
        outputs.to_hdf(folder+'outputs.h5', key='df')     
        
        