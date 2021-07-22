import pandas as pd 
from sklearn import model_selection
import os

# Choose subset of targets if required
targets = ['f_integral', 'f_rocof', 'f_ext', 'f_msd']

# Areas inlcuding "country" areas
areas = ['CE', 'GB', 'Nordic', 'SE', 'CH', 'DE']


for area in areas:

   print('Processing external features from', area)
   
   # Setup folder for this specific version of train-test data
   folder = './data/{}/'.format(area)
   version_folder = folder + 'version_'+ '2021-07-01/' #pd.Timestamp("today").strftime("%Y-%m-%d") + '/'
   if not os.path.exists(version_folder):
      os.makedirs(version_folder)
   
   # Load actual and forecast (day-ahead available) data
   X_actual = pd.read_hdf(folder+'input_actual.h5')
   X_forecast = pd.read_hdf(folder + 'input_forecast.h5')
   y = pd.read_hdf(folder + 'outputs.h5').loc[:,targets]
      
   # Drop nan values
   valid_ind =  ~pd.concat([X_forecast, X_actual, y], axis=1).isnull().any(axis=1)
   X_forecast, X_actual, y = X_forecast[valid_ind], X_actual[valid_ind], y[valid_ind]

   # Join features for full model
   X_full = X_actual.join(X_forecast)

   # Train-test split
   X_train_full, X_test_full, y_train, y_test = model_selection.train_test_split(X_full, y, test_size=0.2, random_state=42)
   X_train_day_ahead = X_forecast.loc[X_train_full.index]
   X_test_day_ahead = X_forecast.loc[X_test_full.index]
   y_pred=pd.DataFrame(index=y_test.index)

   # Save data for full model and restricted (day-ahead) model
   X_train_full.to_hdf(version_folder+'X_train_full.h5',key='df')
   X_train_day_ahead.to_hdf(version_folder+'X_train_day_ahead.h5',key='df')
   y_train.to_hdf(version_folder+'y_train.h5',key='df')
   y_test.to_hdf(version_folder+'y_test.h5',key='df')
   y_pred.to_hdf(version_folder+'y_pred.h5',key='df')
   X_test_full.to_hdf(version_folder+'X_test_full.h5',key='df')
   X_test_day_ahead.to_hdf(version_folder+'X_test_day_ahead.h5',key='df')








