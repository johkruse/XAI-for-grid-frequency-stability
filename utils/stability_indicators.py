import pandas as pd 
import numpy as np 
import os
from glob import glob

def calc_rocof(data,  smooth_window_size, lookup_window_size, method='increment_smoothing'):
  
    if data.index[0].minute!=0 or data.index[0].second!=0:
        print('Data is not starting with full hour!')
        return None
    
    full_hours = data.index[::3600]
    full_hours = full_hours[1:-1]
    
    result = pd.Series(index = full_hours)
    
    if method=='frequency_smoothing':

        for i in np.arange(len(full_hours)):

        
            smoothed_snipped = data.iloc[i*3600:(i+2)*3600].rolling(smooth_window_size, center=True).mean()

            df_dt = smoothed_snipped.diff(periods=5).iloc[3600-lookup_window_size:3600+lookup_window_size]
            
            
            if df_dt.isnull().any():
                result.iloc[i]=np.nan
            else:
                result.iloc[i] = df_dt[df_dt.abs().idxmax()] / 5.   

    if method=='increment_smoothing':

        for i in np.arange(len(full_hours)):
            
            df_dt = data.iloc[i*3600:(i+2)*3600].diff().rolling(smooth_window_size , center=True).mean()
            df_dt = df_dt.iloc[3600-lookup_window_size:3600+lookup_window_size]
                    
            if df_dt.isnull().any():
                result.iloc[i]=np.nan
            else:
                result.iloc[i] = df_dt[df_dt.abs().idxmax()]  

    return result


def make_frequency_data_hdf(path_to_frequency_csv, tso_name, frequency_hdf_folder, start_time, end_time, time_zone, delete_existing_hdf=False):
    
    print('\nConverting frequency data to hdf ', tso_name, '...')
    
    year_index = pd.date_range(start=start_time, end=end_time, freq='Y')
    
    if not os.path.exists(frequency_hdf_folder):
        os.makedirs(frequency_hdf_folder)    
    
    hdf_file = glob(frequency_hdf_folder + 'cleansed*.h5')
    
    if not hdf_file or delete_existing_hdf:
        
        data = pd.Series(dtype=np.float)

        for year in year_index.year:
            print('Processing', year)
            path_to_tso_csv = path_to_frequency_csv + '{}_cleansed/{}/'.format(year, tso_name) + '{}.zip'.format(year)
            year_data = pd.read_csv(path_to_tso_csv, index_col=[0], header=None, squeeze=True)
            data = data.append(year_data)

        data.index = pd.to_datetime(data.index)
        data.index = data.index.tz_localize(time_zone, ambiguous='infer') 
        
        data_start = data.index[0].strftime('%Y-%m-%d')
        data_end = data.index[-1].strftime('%Y-%m-%d')
        
        if hdf_file and delete_existing_hdf:
            os.remove(hdf_file[0])
        
        hdf_file = frequency_hdf_folder + 'cleansed_{}_to_{}.h5'.format(data_start, data_end)
        
        data.to_hdf(hdf_file, key='df') 
        
    else:
        hdf_file = hdf_file[0]
        data_start = hdf_file[-27:-17]
        data_end = hdf_file[-13:-3]
        
        if start_time.strftime('%Y-%m-%d')!=data_start or end_time.strftime('%Y-%m-%d')!=data_end:
            print('Existing hdf file has deviating date range:', hdf_file)
    
    
        
    return hdf_file
        


