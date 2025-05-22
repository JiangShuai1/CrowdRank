# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 14:26:18 2022

@author: 73243
"""

#import os
#os.getcwd()
import numpy as np
import pandas as pd
import datatable as dt
#import time
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, PredefinedSplit
from scipy.special import expit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.metrics import median_absolute_error, mean_absolute_percentage_error, explained_variance_score
#from sklearn.metrics import make_scorer


def LoadData(i,SelFeas):
    file_path = "{}{}{}".format("/home/data/jiangshuai/project/1-OpinionRank/data/Experiment-", i, "/ReportData.csv")
    data = dt.fread(file_path)
    data = data[:, SelFeas]
    data = data.to_pandas()
    cat_cols = [col for col in data.columns if str(data[col].dtype) == 'object' and col != 'Mask']
    num_cols = [col for col in data.columns if str(data[col].dtype) != 'object' and col != 'FErr' and col != 'APE']
    
    data_cat = data[cat_cols].to_numpy()
    enc = OneHotEncoder().fit(data_cat)
    data_cat = enc.transform(data_cat).toarray()
    #enc.categories_
    #data_cat = pd.DataFrame(data_cat)
    data_num = data[num_cols].to_numpy()
    data_num = winsorize(data_num, limits=[0.0001, 0.0001],axis=0)
    scaler = StandardScaler()
    data_num = scaler.fit_transform(data_num)
    data_new = np.concatenate([data_num, data_cat], axis=1)
    
    train_x = data_new[data['Mask'] == 'Train']
    train_val_x = data_new[data['Mask'] != 'Test']
    test_x = data_new[data['Mask'] == 'Test']
    
    train_val_y = data.loc[data['Mask'] != 'Test', 'APE'].to_numpy()
    test_y = data.loc[data['Mask'] == 'Test', 'APE'].to_numpy()
    
    test_fold = np.zeros(train_val_x.shape[0])   # Initialize all indices to 0, where 0 represents the validation set in the first round
    test_fold[:train_x.shape[0]] = -1            # Set indices for the training set to -1, indicating they will never be in the validation set

    return train_val_x, train_val_y, test_x, test_y, test_fold


  
def MyLoss(act, pred):
    ll = mean_squared_error(np.exp(act), np.exp(pred))
    return ll

#loss  = make_scorer(MyLoss, greater_is_better=False)
#score = make_scorer(logloss, greater_is_better=True)


def LinearReg(train_val_x, train_val_y, test_x, test_y):
    model = LinearRegression()
    model.fit(train_val_x, train_val_y)
    preds = model.predict(test_x)
    preds[preds >= 3 ] = 3
    preds[preds < 0] = 0

    MSE = mean_squared_error(test_y, preds)
    MAE = mean_absolute_error(test_y, preds)
    MSLE = mean_squared_log_error(test_y, preds)
    #MSLE = 0
    EVS = explained_variance_score(test_y, preds)
    R2Score = r2_score(test_y, preds)
    MDAE = median_absolute_error(test_y, preds)
    MAPE = mean_absolute_percentage_error(test_y, preds)

    info_per = dt.Frame({'MSE': MSE, 'MAE': MAE, 'MSLE': MSLE, 'EVS': EVS,
                         'R2Score': R2Score, 'MADE': MDAE, 'MAPE': MAPE})
    # info_per = np.round(info_per, 4)
    return info_per, preds


def RidgeReg(train_val_x, train_val_y, test_x, test_y, test_fold):
    model = Ridge()
    ps = PredefinedSplit(test_fold=test_fold)
    params_search = {'alpha': [0.001, 0.01, 0.1, 1.0, 5.0, 10]}
    grid_search_params = {'estimator': model,             # Target classifier
                          'param_grid': params_search,  # Parameter grid to optimize
                          'cv': ps,                     # Use the custom split validation strategy
                          'n_jobs': -1,                 # Number of parallel jobs, -1 means using all CPUs
                          'verbose': 100}               # Output information level, higher number means more output
    grsearch = GridSearchCV(**grid_search_params)
    grsearch.fit(train_val_x, train_val_y)
    bst_para = pd.DataFrame(grsearch.best_params_, index=[1])
    bst = grsearch.best_estimator_
    preds = bst.predict(test_x)
    preds[preds >= 3] = 3
    preds[preds < 0] = 0

    MSE = mean_squared_error(test_y, preds)
    MAE = mean_absolute_error(test_y, preds)
    MSLE = mean_squared_log_error(test_y, preds)
    #MSLE = 0
    EVS = explained_variance_score(test_y, preds)
    R2Score = r2_score(test_y, preds)
    MDAE = median_absolute_error(test_y, preds)
    MAPE = mean_absolute_percentage_error(test_y, preds)

    info_per = dt.Frame({'MSE': MSE, 'MAE': MAE, 'MSLE': MSLE, 'EVS': EVS,
                         'R2Score': R2Score, 'MADE': MDAE, 'MAPE': MAPE})
    # info_per = np.round(info_per, 4)
    return info_per, preds, bst_para



def LassoReg(train_val_x, train_val_y, test_x, test_y, test_fold):
    model = Lasso()
    ps = PredefinedSplit(test_fold=test_fold)
    params_search = {'alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]}
    grid_search_params = {'estimator': model,             # Target classifier
                          'param_grid': params_search,  # Parameter grid to optimize
                          'cv': ps,                     # Use the custom split validation strategy
                          'n_jobs': -1,                 # Number of parallel jobs, -1 means using all CPUs
                          'verbose': 32}                # Output information level, higher number means more output
    grsearch = GridSearchCV(**grid_search_params)
    grsearch.fit(train_val_x, train_val_y)
    bst_para = pd.DataFrame(grsearch.best_params_, index=[1])
    bst = grsearch.best_estimator_
    preds = bst.predict(test_x)
    preds[preds >= 3] = 3
    preds[preds < 0] = 0
    
    MSE = mean_squared_error(test_y, preds)
    MAE = mean_absolute_error(test_y, preds)
    MSLE = mean_squared_log_error(test_y, preds)
    #MSLE = 0
    EVS = explained_variance_score(test_y, preds)
    R2Score = r2_score(test_y, preds)
    MDAE = median_absolute_error(test_y, preds)
    MAPE = mean_absolute_percentage_error(test_y, preds)

    info_per = dt.Frame({'MSE': MSE, 'MAE': MAE, 'MSLE': MSLE, 'EVS': EVS,
                         'R2Score': R2Score, 'MADE': MDAE, 'MAPE': MAPE})
    # info_per = np.round(info_per, 4)
    return info_per, preds, bst_para



def ElasticNetReg(train_val_x, train_val_y, test_x, test_y, test_fold):
    model = ElasticNet()
    ps = PredefinedSplit(test_fold=test_fold)
    params_search = {'l1_ratio': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]}
    grid_search_params = {'estimator': model,             # Target classifier
                          'param_grid': params_search,  # Parameter grid to optimize
                          'cv': ps,                     # Use the custom split validation strategy
                          'n_jobs': -1,                 # Number of parallel jobs, -1 means using all CPUs
                          'verbose': 100}               # Output information level, higher number means more output
    grsearch = GridSearchCV(**grid_search_params)
    grsearch.fit(train_val_x, train_val_y)
    bst_para = pd.DataFrame(grsearch.best_params_, index=[1])
    bst = grsearch.best_estimator_
    preds = bst.predict(test_x)
    preds[preds >= 3] = 3
    preds[preds < 0] = 0
    
    MSE = mean_squared_error(test_y, preds)
    MAE = mean_absolute_error(test_y, preds)
    MSLE = mean_squared_log_error(test_y, preds)
    #MSLE = 0
    EVS = explained_variance_score(test_y, preds)
    R2Score = r2_score(test_y, preds)
    MDAE = median_absolute_error(test_y, preds)
    MAPE = mean_absolute_percentage_error(test_y, preds)

    info_per = dt.Frame({'MSE': MSE, 'MAE': MAE, 'MSLE': MSLE, 'EVS': EVS,
                         'R2Score': R2Score, 'MADE': MDAE, 'MAPE': MAPE})
    # info_per = np.round(info_per, 4)
    return info_per, preds, bst_para



def SVRReg(train_val_x, train_val_y, test_x, test_y, test_fold):
    model = LinearSVR(max_iter=5000, tol=0.001)
    ps = PredefinedSplit(test_fold=test_fold)
    params_search = {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]}
    grid_search_params = {'estimator': model,             # Target classifier
                          'param_grid': params_search,  # Parameter grid to optimize
                          'cv': ps,                     # Use the custom split validation strategy
                          'n_jobs': -1,                 # Number of parallel jobs, -1 means using all CPUs
                          'verbose': 100}               # Output information level, higher number means more output
    grsearch = GridSearchCV(**grid_search_params)
    grsearch.fit(train_val_x, train_val_y)
    bst_para = pd.DataFrame(grsearch.best_params_, index=[1])
    bst = grsearch.best_estimator_
    preds = bst.predict(test_x)
    preds[preds >= 3] = 3
    preds[preds < 0] = 0
    
    MSE = mean_squared_error(test_y, preds)
    MAE = mean_absolute_error(test_y, preds)
    MSLE = mean_squared_log_error(test_y, preds)
    #MSLE = 0
    EVS = explained_variance_score(test_y, preds)
    R2Score = r2_score(test_y, preds)
    MDAE = median_absolute_error(test_y, preds)
    MAPE = mean_absolute_percentage_error(test_y, preds)

    info_per = dt.Frame({'MSE': MSE, 'MAE': MAE, 'MSLE': MSLE, 'EVS': EVS,
                         'R2Score': R2Score, 'MADE': MDAE, 'MAPE': MAPE})
    # info_per = np.round(info_per, 4)
    return info_per, preds, bst_para



def DTreeReg(train_val_x, train_val_y, test_x, test_y):
    model = DecisionTreeRegressor()
    model.fit(train_val_x, train_val_y)
    preds = model.predict(test_x)
    preds[preds >= 3] = 3
    preds[preds < 0] = 0
    
    MSE = mean_squared_error(test_y, preds)
    MAE = mean_absolute_error(test_y, preds)
    MSLE = mean_squared_log_error(test_y, preds)
    #MSLE = 0
    EVS = explained_variance_score(test_y, preds)
    R2Score = r2_score(test_y, preds)
    MDAE = median_absolute_error(test_y, preds)
    MAPE = mean_absolute_percentage_error(test_y, preds)

    info_per = dt.Frame({'MSE': MSE, 'MAE': MAE, 'MSLE': MSLE, 'EVS': EVS,
                         'R2Score': R2Score, 'MADE': MDAE, 'MAPE': MAPE})
    # info_per = np.round(info_per, 4)
    return info_per, preds



def RFsReg(train_val_x, train_val_y, test_x, test_y, test_fold):
    model = RandomForestRegressor(n_jobs=-1)
    ps = PredefinedSplit(test_fold=test_fold)
    params_search = {'n_estimators': [50, 100, 200, 300],
                     'max_samples': [0.1, 0.2, 0.3, 0.4, 0.5],
                     'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
    grid_search_params = {'estimator': model,             # Target classifier
                          'param_distributions': params_search,  # Parameter grid to optimize
                          'cv': ps,                     # Use the custom split validation strategy
                          'n_iter': 15,  # Use the custom split validation strategy
                          'verbose': 100}                # Output information level, higher number means more output
    grsearch = RandomizedSearchCV(**grid_search_params)
    grsearch.fit(train_val_x, train_val_y)
    bst_para = pd.DataFrame(grsearch.best_params_, index=[1])
    bst = grsearch.best_estimator_
    preds = bst.predict(test_x)
    preds[preds >= 3] = 3
    preds[preds < 0] = 0
    
    MSE = mean_squared_error(test_y, preds)
    MAE = mean_absolute_error(test_y, preds)
    MSLE = mean_squared_log_error(test_y, preds)
    #MSLE = 0
    EVS = explained_variance_score(test_y, preds)
    R2Score = r2_score(test_y, preds)
    MDAE = median_absolute_error(test_y, preds)
    MAPE = mean_absolute_percentage_error(test_y, preds)

    info_per = dt.Frame({'MSE': MSE, 'MAE': MAE, 'MSLE': MSLE, 'EVS': EVS,
                         'R2Score': R2Score, 'MADE': MDAE, 'MAPE': MAPE})
    # info_per = np.round(info_per, 4)
    return info_per, preds, bst_para




def AdaBoostReg(train_val_x, train_val_y, test_x, test_y, test_fold):
    model = AdaBoostRegressor()
    ps = PredefinedSplit(test_fold=test_fold)
    params_search = {'n_estimators': [50, 100, 200, 300],
                     'learning_rate': [0.1, 0.5, 1.0, 2.0]}
    grid_search_params = {'estimator': model,             # Target classifier
                          'param_grid': params_search,  # Parameter grid to optimize
                          'cv': ps,                     # Use the custom split validation strategy
                          'n_jobs': -1,                 # Number of parallel jobs, -1 means using all CPUs
                          'verbose': 100}                # Output information level, higher number means more output
    grsearch = GridSearchCV(**grid_search_params)
    grsearch.fit(train_val_x, train_val_y)
    bst_para = pd.DataFrame(grsearch.best_params_, index=[1])
    bst = grsearch.best_estimator_
    preds = bst.predict(test_x)
    preds[preds >= 3] = 3
    preds[preds < 0] = 0
    
    MSE = mean_squared_error(test_y, preds)
    MAE = mean_absolute_error(test_y, preds)
    MSLE = mean_squared_log_error(test_y, preds)
    #MSLE = 0
    EVS = explained_variance_score(test_y, preds)
    R2Score = r2_score(test_y, preds)
    MDAE = median_absolute_error(test_y, preds)
    MAPE = mean_absolute_percentage_error(test_y, preds)

    info_per = dt.Frame({'MSE': MSE, 'MAE': MAE, 'MSLE': MSLE, 'EVS': EVS,
                         'R2Score': R2Score, 'MADE': MDAE, 'MAPE': MAPE})
    # info_per = np.round(info_per, 4)
    return info_per, preds, bst_para




def XGBReg(train_val_x, train_val_y, test_x, test_y, test_fold):
    model = XGBRegressor(n_jobs=15)
    ps = PredefinedSplit(test_fold=test_fold)
    params_search = {'n_estimators': [50, 100, 200, 300],
                     'learning_rate': [0.01, 0.05, 0.1, 1.0]}
    grid_search_params = {'estimator': model,             # Target classifier
                          'param_grid': params_search,  # Parameter grid to optimize
                          'cv': ps,                     # Use the custom split validation strategy
                          'n_jobs': 15,  # Number of parallel jobs, -1 means using all CPUs
                          'verbose': 100}                # Output information level, higher number means more output
    grsearch = GridSearchCV(**grid_search_params)
    grsearch.fit(train_val_x, train_val_y)
    bst_para = pd.DataFrame(grsearch.best_params_, index=[1])
    bst = grsearch.best_estimator_
    preds = bst.predict(test_x)
    preds[preds >= 3] = 3
    preds[preds < 0] = 0
    
    MSE = mean_squared_error(test_y, preds)
    MAE = mean_absolute_error(test_y, preds)
    MSLE = mean_squared_log_error(test_y, preds)
    #MSLE = 0
    EVS = explained_variance_score(test_y, preds)
    R2Score = r2_score(test_y, preds)
    MDAE = median_absolute_error(test_y, preds)
    MAPE = mean_absolute_percentage_error(test_y, preds)

    info_per = dt.Frame({'MSE': MSE, 'MAE': MAE, 'MSLE': MSLE, 'EVS': EVS,
                         'R2Score': R2Score, 'MADE': MDAE, 'MAPE': MAPE})
    # info_per = np.round(info_per, 4)
    return info_per, preds, bst_para



def LGBMReg(train_val_x, train_val_y, test_x, test_y, test_fold):
    model = LGBMRegressor(n_jobs=-1)
    ps = PredefinedSplit(test_fold=test_fold)
    params_search = {'n_estimators': [50, 100, 200, 300],
                     'learning_rate': [0.01, 0.05, 0.1, 1.0]}
    grid_search_params = {'estimator': model,             # Target classifier
                          'param_grid': params_search,  # Parameter grid to optimize
                          'cv': ps,                     # Use the custom split validation strategy
                          'verbose': 100}                # Output information level, higher number means more output
    grsearch = GridSearchCV(**grid_search_params)
    grsearch.fit(train_val_x, train_val_y)
    bst_para = pd.DataFrame(grsearch.best_params_, index=[1])
    bst = grsearch.best_estimator_
    preds = bst.predict(test_x)
    preds[preds >= 3] = 3
    preds[preds < 0] = 0
    
    MSE = mean_squared_error(test_y, preds)
    MAE = mean_absolute_error(test_y, preds)
    MSLE = mean_squared_log_error(test_y, preds)
    #MSLE = 0
    EVS = explained_variance_score(test_y, preds)
    R2Score = r2_score(test_y, preds)
    MDAE = median_absolute_error(test_y, preds)
    MAPE = mean_absolute_percentage_error(test_y, preds)

    info_per = dt.Frame({'MSE': MSE, 'MAE': MAE, 'MSLE': MSLE, 'EVS': EVS,
                         'R2Score': R2Score, 'MADE': MDAE, 'MAPE': MAPE})
    # info_per = np.round(info_per, 4)
    return info_per, preds, bst_para



def MLPReg(train_val_x, train_val_y, test_x, test_y, test_fold):
    model = MLPRegressor(hidden_layer_sizes=(128, 128, 128, 128), max_iter=15, batch_size=5000,
                         tol=0.005, early_stopping=True)
    ps = PredefinedSplit(test_fold=test_fold)
    params_search = {'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
                     'alpha': [0.01, 0.1, 0.5, 1.0]}
    grid_search_params = {'estimator': model,             # Target classifier
                          'param_grid': params_search,  # Parameter grid to optimize
                          'cv': ps,                     # Use the custom split validation strategy
                          'n_jobs': -1,                 # Number of parallel jobs, -1 means using all CPUs
                          'verbose': 100}                # Output information level, higher number means more output
    grsearch = GridSearchCV(**grid_search_params)
    grsearch.fit(train_val_x, train_val_y)
    bst_para = pd.DataFrame(grsearch.best_params_, index=[1])
    bst = grsearch.best_estimator_
    preds = bst.predict(test_x)
    preds[preds >= 3] = 3
    preds[preds < 0] = 0
    
    MSE = mean_squared_error(test_y, preds)
    MAE = mean_absolute_error(test_y, preds)
    MSLE = mean_squared_log_error(test_y, preds)
    #MSLE = 0
    EVS = explained_variance_score(test_y, preds)
    R2Score = r2_score(test_y, preds)
    MDAE = median_absolute_error(test_y, preds)
    MAPE = mean_absolute_percentage_error(test_y, preds)

    info_per = dt.Frame({'MSE': MSE, 'MAE': MAE, 'MSLE': MSLE, 'EVS': EVS,
                         'R2Score': R2Score, 'MADE': MDAE, 'MAPE': MAPE})
    # info_per = np.round(info_per, 4)
    return info_per, preds, bst_para