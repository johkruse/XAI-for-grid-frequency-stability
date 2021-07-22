import pandas as pd 
from matplotlib import pyplot as plt 
import os
import matplotlib
matplotlib.use('agg')

# Day-ahead available and actual (ex-post) features from ENTSO-E data
actual_cols = ['load', 'gen_biomass', 'gen_lignite', 'gen_coal_gas', 'gen_gas',
               'gen_hard_coal', 'gen_oil', 'gen_oil_shale', 'gen_fossil_peat',
               'gen_geothermal', 'gen_pumped_hydro', 'gen_run_off_hydro',
               'gen_reservoir_hydro', 'gen_marine', 'gen_nuclear', 'gen_other_renew',
               'gen_solar', 'gen_waste', 'gen_wind_off', 'gen_wind_on', 'gen_other']

forecast_cols = ['load_day_ahead', 'scheduled_gen_total','prices_day_ahead',
                 'solar_day_ahead','wind_off_day_ahead', 'wind_on_day_ahead']


# Areas inlcuding "country" areas
areas = ['CE', 'GB', 'Nordic', 'SE', 'CH', 'DE']

# Time zones of frequency recordings
tzs = {'GB':'GB', 'Nordic':'Europe/Helsinki', 'CE':'CET', 'DE':'CET', 'SE':'Europe/Helsinki', 'CH':'CET'}


for area in areas:

    print('Processing external features from', area)
    
    # Setup folder paths to raw input data 
    folder = './data/{}/'.format(area)
    doc_folder = folder + 'documentation_of_data_download/'
    if not os.path.exists(doc_folder):
        os.makedirs(doc_folder)

    # Load the pre-processed external features
    raw_input_data = pd.read_hdf(folder + 'raw_input_data.h5')

    # Inspection of data distribution as histograms
    fig,ax=plt.subplots(figsize=(20,20))
    raw_input_data.hist(log=True,ax=ax, bins=100)
    plt.tight_layout()
    plt.savefig(doc_folder+'raw_data_histograms.svg', bbox_inches='tight')
    plt.close()

    # Split into forecast and actual data and convert index to local timezone of frequency data
    input_forecast = raw_input_data.loc[:, raw_input_data.columns.intersection(forecast_cols)]
    input_actual = raw_input_data.loc[:, raw_input_data.columns.intersection(actual_cols)]
    input_actual.index = input_actual.index.tz_convert(tzs[area])
    input_forecast.index = input_forecast.index.tz_convert(tzs[area])

    ####  Additional engineered features ###
    
    # Time
    input_forecast['month'] = input_forecast.index.month
    input_forecast['weekday'] =  input_forecast.index.weekday
    input_forecast['hour'] =  input_forecast.index.hour

    # Total generation
    input_actual['total_gen'] = input_actual.filter(regex='^gen').sum(axis='columns')


    # Inertia proxy - Sum of all synchronous generation
    input_actual['synchronous_gen'] = input_actual.total_gen- input_actual.loc[:,['gen_solar',
                                                                                  'gen_wind_off',
                                                                                  'gen_wind_on']].sum(axis=1)

    # Ramps of load and total generation 
    input_forecast['load_ramp_day_ahead'] = input_forecast.load_day_ahead.diff()
    input_actual['load_ramp'] = input_actual.load.diff()
    input_forecast['total_gen_ramp_day_ahead'] = input_forecast.scheduled_gen_total.diff()
    input_actual['total_gen_ramp'] = input_actual.total_gen.diff()


    # Ramps of generaton types
    if 'wind_off_day_ahead' in input_forecast.columns:
        input_forecast['wind_off_ramp_day_ahead'] = input_forecast.wind_off_day_ahead.diff()
    input_forecast['wind_on_ramp_day_ahead'] = input_forecast.wind_on_day_ahead.diff()
    if 'solar_day_ahead' in input_forecast.columns:
        input_forecast['solar_ramp_day_ahead'] = input_forecast.solar_day_ahead.diff()
    gen_ramp_cols = input_actual.filter(regex='^gen').columns.str[4:] + '_ramp'
    input_actual[gen_ramp_cols] = input_actual.filter(regex='^gen').diff()

    # Price Ramps
    input_forecast['price_ramp_day_ahead'] = input_forecast.prices_day_ahead.diff()

    # Forecast errors
    input_actual['forecast_error_wind_on'] = input_forecast.wind_on_day_ahead - input_actual.gen_wind_on
    if 'wind_off_day_ahead' in input_forecast.columns:
        input_actual['forecast_error_wind_off'] = input_forecast.wind_off_day_ahead - input_actual.gen_wind_off
        input_actual['forecast_error_wind_off_ramp'] = input_forecast.wind_off_ramp_day_ahead - input_actual.wind_off_ramp
    input_actual['forecast_error_total_gen'] = input_forecast.scheduled_gen_total - input_actual.total_gen
    input_actual['forecast_error_load'] = input_forecast.load_day_ahead - input_actual.load
    input_actual['forecast_error_load_ramp'] = input_forecast.load_ramp_day_ahead - input_actual.load_ramp
    input_actual['forecast_error_total_gen_ramp'] = input_forecast.total_gen_ramp_day_ahead - input_actual.total_gen_ramp
    input_actual['forecast_error_wind_on_ramp'] = input_forecast.wind_on_ramp_day_ahead - input_actual.wind_on_ramp
    if 'solar_day_ahead' in input_forecast.columns:
        input_actual['forecast_error_solar_ramp'] = input_forecast.solar_ramp_day_ahead - input_actual.solar_ramp
        input_actual['forecast_error_solar'] = input_forecast.solar_day_ahead - input_actual.gen_solar


    # Save data
    input_actual.to_hdf(folder+'input_actual.h5',key='df')
    input_forecast.to_hdf(folder+'input_forecast.h5',key='df')





