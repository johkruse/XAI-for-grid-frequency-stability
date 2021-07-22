import pandas as pd 
import numpy as np 
from sklearn.model_selection import GridSearchCV, train_test_split
import os
import time
import shap
import xgboost as xgb
from sklearn.metrics import r2_score

# Setup 
areas = ['DE','GB', 'CE', 'SE', 'CH', 'Nordic']  
data_version = '2021-07-01'
targets = ['f_integral', 'f_ext', 'f_msd', 'f_rocof']

start_time = time.time()

for area in areas:
    
    print('---------------------------- ', area, ' ------------------------------------')
    
    data_folder = './data/{}/version_{}/'.format(area,data_version)

    for target in targets: 
        
        print('-------- ', target, ' --------')
        
        # Result folder where prediction, SHAP values and CV results are saved
        res_folder = './results/model_fit/{}/version_{}/target_{}/'.format(area,data_version, target)

        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        
        # Load target data
        y_train = pd.read_hdf(data_folder+'y_train.h5').loc[:, target]
        y_test = pd.read_hdf(data_folder+'y_test.h5').loc[:, target]
        y_pred = pd.read_hdf(data_folder+'y_pred.h5') #contains only time index

        for model_type in ['_day_ahead','_full']:

            # Load feature data
            X_train = pd.read_hdf(data_folder+'X_train{}.h5'.format(model_type))
            X_test = pd.read_hdf(data_folder+'X_test{}.h5'.format(model_type)) 
          
            # Daily profile prediction
            daily_profile = y_train.groupby(X_train.index.time).mean()
            y_pred['daily_profile'] = [daily_profile[time] for time in X_test.index.time]

            # Mean predictor
            y_pred['mean_predictor'] = y_train.mean()

            #### Gradient boosting Regressor CV hyperparameter optimization ###

            # Split training set into (smaller) training set and validation set
            X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train,
                                                                                        test_size=0.2)
            # Parameters for hyper-parameter optimization
            params_grid = {
                'max_depth': [3,5,7,9,11],
                'learning_rate':[0.01,0.05,0.1],
                'subsample': [1,0.7,0.4],
                'min_child_weight':[1,5,10,30],
                'reg_lambda':[ 0.1, 1, 10]
            }
            fit_params = {
                'eval_set':[(X_train_train, y_train_train),(X_train_val, y_train_val)],
                'early_stopping_rounds':20, 
                'verbose':0
            }
            
            # Grid search for optimal hyper-parameters
            grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000,
                                                        verbosity=0, n_jobs=1, base_score = y_train.mean()),
                                    params_grid, verbose=1, n_jobs=25, cv=5)

            grid_search.fit(X_train_train, y_train_train, **fit_params)
            
            # Save CV results
            pd.DataFrame(grid_search.cv_results_).to_csv(res_folder+'cv_results_gtb{}.csv'.format(model_type))

            # Save best params (including n_estimators from early stopping on validation set)
            best_params = grid_search.best_estimator_.get_params()
            best_params['n_estimators'] = grid_search.best_estimator_.best_ntree_limit
            pd.DataFrame(best_params, index=[0]).to_csv(res_folder+'cv_best_params_gtb{}.csv'.format(model_type))

            # Gradient boosting regression best model evaluation on test set
            best_params = pd.read_csv(res_folder+'cv_best_params_gtb{}.csv'.format(model_type),
                                      usecols = list(params_grid.keys()) + ['n_estimators', 'base_score', 'objective'])
            best_params = best_params.to_dict('records')[0]
            best_params['n_jobs'] = 25
            print('Number of opt. boosting rounds:', best_params['n_estimators'] )
            
            # Train on whole training set (including validation set)
            model = xgb.XGBRegressor(**best_params)
            model.fit(X_train, y_train)

            # Calculate SHAP values on test set
            if area in ['CE', 'Nordic', 'GB']:
                if model_type=='_full':
                    shap_vals = shap.TreeExplainer(model).shap_values(X_test)
                    np.save(res_folder + 'shap_values_gtb{}.npy'.format(model_type), shap_vals)
                
                    shap_interact_vals = shap.TreeExplainer(model).shap_interaction_values(X_test)
                    np.save(res_folder + 'shap_interaction_values_gtb{}.npy'.format(model_type), shap_interact_vals)

            # Prediction on test set
            y_pred['gtb{}'.format(model_type)] = model.predict(X_test) 
            print(model_type[1:], 'Best performance: {}'.format(r2_score(y_test, y_pred['gtb{}'.format(model_type)])))   

        # Save prediction
        y_pred.to_hdf(res_folder+'y_pred.h5',key='df')
        

print("Execution time [h]: {}".format((time.time() - start_time)/3600.))

# %%
