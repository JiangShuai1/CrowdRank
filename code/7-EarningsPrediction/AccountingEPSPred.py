# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 14:26:18 2022

@author: 73243
"""


import os
import numpy as np
import pandas as pd
import time
from progress.bar import IncrementalBar
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
#from sklearn.metrics import make_scorer


def LoadData(i,SelFeas):
    file_path = f"./data/Experiment-{i}/ReportData.csv"
    data = pd.read_csv(file_path)

    agg_data = data.groupby('GroupID')['AFErr'].agg(['max', 'min', 'mean']).reset_index()  

    data = data.loc[data.groupby('GroupID')['ReportDate'].idxmax()]

    # 将统计结果与原始数据合并  
    merged_data = pd.merge(data, agg_data, on='GroupID', how='left')  

    # 更改列名使之更清晰 (可选)  
    merged_data = merged_data.rename(columns={'max': 'AFErr_max', 'min': 'AFErr_min', 'mean': 'AFErr_mean'})
    merged_data = merged_data[merged_data['Mask'] == 'Test']
    merged_data['AFE_diff'] = merged_data['AFErr_max'] - merged_data['AFErr_min']
    merged_data['AFE_diff'] = merged_data['AFE_diff'].clip(lower=0.001)
    data = data[SelFeas]


    cat_cols = [col for col in data.columns if str(data[col].dtype) == 'object' and col != 'Mask']
    num_cols = [col for col in data.columns if str(data[col].dtype) != 'object' and col != 'AEPS']
    
    data_cat = data[cat_cols].to_numpy()
    enc = OneHotEncoder().fit(data_cat)
    data_cat = enc.transform(data_cat).toarray()
    #enc.categories_
    #data_cat = pd.DataFrame(data_cat)
    data_num = data[num_cols].to_numpy()
    data_num = winsorize(data_num, limits=[0.001, 0.001],axis=0)
    #scaler = StandardScaler()
    #data_num = scaler.fit_transform(data_num)
    data_new = np.concatenate([data_num, data_cat], axis=1)
    
    train_x = data_new[data['Mask'] == 'Train']
    train_val_x = data_new[data['Mask'] != 'Test']
    test_x = data_new[data['Mask'] == 'Test']
    
    train_val_y = data.loc[data['Mask'] != 'Test', 'AEPS'].to_numpy()
    test_y = data.loc[data['Mask'] == 'Test', 'AEPS'].to_numpy()
    
    test_fold = np.zeros(train_val_x.shape[0])   # 将所有index初始化为0,0表示第一轮的验证集
    test_fold[:train_x.shape[0]] = -1            # 将训练集对应的index设为-1，表示永远不划分到验证集中

    return train_val_x, train_val_y, test_x, test_y, test_fold, merged_data



def XGBReg(train_val_x, train_val_y, test_x, test_fold, test_data):
    model = XGBRegressor(n_jobs=15)
    ps = PredefinedSplit(test_fold=test_fold)
    params_search = {'n_estimators': [300, 400, 500],
                     'learning_rate': [0.01, 0.05, 0.1]}
    grid_search_params = {'estimator': model,             # 目标分类器
                          'param_grid': params_search,  # 前面定义的我们想要优化的参数
                          'cv': ps,                     # 使用前面自定义的split验证策略
                          'n_jobs': 15,  # 并行运行的任务数，-1表示使用所有CPU
                          'verbose': 100}                # 输出信息，数字越大输出信息越多
    grsearch = GridSearchCV(**grid_search_params)
    grsearch.fit(train_val_x, train_val_y)
    bst_para = pd.DataFrame(grsearch.best_params_, index=[1])
    bst = grsearch.best_estimator_
    preds = bst.predict(test_x)
    merged_data = test_data.copy()
    merged_data['Preds'] = preds
    merged_data = merged_data[merged_data["GroupSize"]>=5]
    merged_data['AFE'] = abs(merged_data['Preds'] - merged_data['AEPS'])
    merged_data['APE'] = merged_data['AFE'] / abs(merged_data['AEPS'])
    merged_data['FACC'] = (merged_data['AFErr_max'] - merged_data['AFE'])/ merged_data['AFE_diff']
    merged_data['PMAFE'] = (merged_data['AFErr_mean'] - merged_data['AFE'])/ merged_data['AFErr_mean']
    merged_data = merged_data.replace([np.inf, -np.inf], np.nan)
    merged_data = merged_data[['APE', 'FACC', 'PMAFE']]
    # 计算每列的平均值，忽略 NaN
    mean_values = merged_data.mean(skipna=True)

    # 创建新的 DataFrame per，只有一行，包含计算出的平均值
    per = pd.DataFrame([mean_values])

    # 确保新 DataFrame 的列名正确
    per.columns = ['APE', 'FACC', 'PMAFE']
    # info_per = np.round(info_per, 4)
    return per, preds, bst_para



def LGBMReg(train_val_x, train_val_y, test_x, test_fold, test_data):
    model = LGBMRegressor(n_jobs=20)
    ps = PredefinedSplit(test_fold=test_fold)
    params_search = {'n_estimators': [300, 400, 500],
                     'learning_rate': [0.01, 0.05, 0.1]}
    grid_search_params = {'estimator': model,             # 目标分类器
                          'param_grid': params_search,  # 前面定义的我们想要优化的参数
                          'cv': ps,                     # 使用前面自定义的split验证策略
                          'verbose': 100}                # 输出信息，数字越大输出信息越多
    grsearch = GridSearchCV(**grid_search_params)
    grsearch.fit(train_val_x, train_val_y)
    bst_para = pd.DataFrame(grsearch.best_params_, index=[1])
    bst = grsearch.best_estimator_
    preds = bst.predict(test_x)

    merged_data = test_data.copy()
    merged_data['Preds'] = preds
    merged_data = merged_data[merged_data["GroupSize"]>=5]
    # 接下来进行赋值操作
    merged_data['AFE'] = abs(merged_data['Preds'] - merged_data['AEPS'])
    merged_data['APE'] = merged_data['AFE'] / abs(merged_data['AEPS'])
    merged_data['FACC'] = (merged_data['AFErr_max'] - merged_data['AFE'])/ merged_data['AFE_diff']
    merged_data['PMAFE'] = (merged_data['AFErr_mean'] - merged_data['AFE'])/ merged_data['AFErr_mean']
    merged_data = merged_data.replace([np.inf, -np.inf], np.nan)
    merged_data = merged_data[['APE', 'FACC', 'PMAFE']]

    # 计算每列的平均值，忽略 NaN
    mean_values = merged_data.mean(skipna=True)

    # 创建新的 DataFrame per，只有一行，包含计算出的平均值
    per = pd.DataFrame([mean_values])

    # 确保新 DataFrame 的列名正确
    per.columns = ['APE', 'FACC', 'PMAFE']
    # info_per = np.round(info_per, 4)
    return per, preds, bst_para




SelFeas = [
    "IndustryID","ListYear","Udwnm","PE","PB","PS","Turnover","Liquidility","CircuMarketValue",
    "Volatility","Beta","Corr","NonSysRisk","StockPrice","ChangeRatio","CAR5","CAR20","CAR60",
    "PrevEPS","AEPS","Mask"]


method_list = ['XGBoost', 'LGBM']
algorithms = pd.DataFrame({"Methods": method_list})


EXPIDs = [1,2,3]
bar = IncrementalBar('Countdown', max = len(EXPIDs))
All_Start_Time = time.time()
for i in EXPIDs:
    print(("Starting Experiment, Current Data Set ID is: %.d"%(i)).center(200 // 2,"-"))
    This_Exp_Start = time.time()
    ###############################   Loading Data
    train_val_x, train_val_y, test_x, test_y, test_fold, merged_data = LoadData(i, SelFeas)
    print('Data Successfully Loaded!')  # 1secs
    ###############################   Runing Models
    
    # XGBoost Regression
    start = time.time()
    xgb_per, xgb_preds, xgb_bst_para = XGBReg(train_val_x, train_val_y, test_x, test_fold, merged_data)
    print('Model-1-XGBoost: time of training is %.4f seconds.'%(time.time()-start))#1secs
    
    # LGBM Regression
    start = time.time()
    lgbm_per, lgbm_preds, lgbm_bst_para = LGBMReg(train_val_x, train_val_y, test_x, test_fold, merged_data)
    print('Model-2-LGBM: time of training is %.4f seconds.'%(time.time()-start))#1secs
    
    ############ Results
    # Performance
    info_per = pd.concat([xgb_per, lgbm_per], axis=0)
    algorithms_reset = algorithms.reset_index(drop=True)
    info_per_reset = info_per.reset_index(drop=True)

    # 然后进行连接
    info_per = pd.concat([algorithms_reset, info_per_reset], axis=1)
    info_per.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/AddedACWPer/EPSPred_Performance.csv"))
    # Prediction
    all_preds = pd.DataFrame(np.column_stack((xgb_preds, lgbm_preds)))
    all_preds.columns = method_list
    all_preds.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/AddedACWPer/EarningsPrediction.csv"))
    # Best hyper-parameters
    # os.makedirs("./output/table/Experiment-{}/HyperP".format(i), exist_ok=True)
    xgb_bst_para.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/HyperP/EPSPred_XGBoost.csv"))
    lgbm_bst_para.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/HyperP/EPSPred_LGBM.csv"))

    print('time of experiments on this data set is %.4f seconds.'%(time.time()-This_Exp_Start))  #1secs
    bar.next()
    time.sleep(1)

bar.finish()
print('time of experiments on the whole data set is %.4f seconds.'%(time.time()-All_Start_Time)) #1secs