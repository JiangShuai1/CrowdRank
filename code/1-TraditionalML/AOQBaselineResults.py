# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 15:26:00 2022

@author: shuai jiang
"""


import os
import numpy as np
import pandas as pd
import time
from datatable import dt
import sys
from progress.bar import IncrementalBar
import AOQBaselineModels as BSM 


SelFeas = [
    "AnaNum","Gender","Degree","StarTimes","AnaReportNum","AnaIndustryNum","AnaStockNum",
    "Per_ME_Err","Per_SD_Err","Per_ME_APE","Per_SD_APE","BrokerName","ActiveAnaNum",
    "BroReportNum","BroIndustryNum","BroStockNum","BroP_ME_APE","BroP_SD_APE",
    "IndustryID","ListYear","Udwnm","PE","PB","PS","Turnover","Liquidility","CircuMarketValue",
    "Volatility","Beta","Corr","NonSysRisk","StockPrice","ChangeRatio","CAR5","CAR20","CAR60",
    "StdRank","RankChan","FEPS","RevHorizon","ForHorizon","PrevEPS","AnaStockReportNum","AnaStockIndustryNum",
    "StockPer_ME_Err","StockPer_SD_Err","StockPer_ME_APE","StockPer_SD_APE","PreForNum","Boldness",
    "FErr","APE","Mask"]

method_list = ['LR', 'Ridge', 'Lasso', 'ElasticNet', 'SVR', 'DT','RFs', 'Adaboost', 'XGBoost', 'LGBM']
algorithms = dt.Frame({"Methods": method_list})

EXPIDs = [1, 2, 3]
bar = IncrementalBar('Countdown', max = len(EXPIDs))
All_Start_Time = time.time()
for i in EXPIDs:
    print(("Starting Experiment, Current Data Set ID is: %.d"%(i)).center(200 // 2,"-"))
    This_Exp_Start = time.time()
    ###############################   Loading Data
    train_val_x, train_val_y, test_x, test_y, test_fold = BSM.LoadData(i, SelFeas)
    print('Data Successfully Loaded!')  # 1secs
    ###############################   Runing Models
    
    # Linear Regression
    start = time.time()
    lr_per, lr_preds = BSM.LinearReg(train_val_x, train_val_y, test_x, test_y)
    print('Model-1-LR: time of training is %.4f seconds.'%(time.time()-start))#1secs
    
    # Ridge Regression
    start = time.time()
    ridge_per, ridge_preds, ridge_bst_para = BSM.RidgeReg(train_val_x, train_val_y, test_x, test_y, test_fold)
    print('Model-2-Ridge: time of training is %.4f seconds.'%(time.time()-start))#1secs
    
    # Lasso Regression
    start = time.time()
    lasso_per, lasso_preds, lasso_bst_para = BSM.LassoReg(train_val_x, train_val_y, test_x, test_y, test_fold)
    print('Model-3-Lasso: time of training is %.4f seconds.'%(time.time()-start))#1secs
    
    # ElasticNet
    start = time.time()
    elanet_per, elanet_preds, elanet_bst_para = BSM.ElasticNetReg(train_val_x, train_val_y, test_x, test_y, test_fold)
    print('Model-4-ElasticNet: time of training is %.4f seconds.'%(time.time()-start))#1secs
    
    # Support Vector Regression
    start = time.time()
    svr_per, svr_preds, svr_bst_para = BSM.SVRReg(train_val_x, train_val_y, test_x, test_y, test_fold)
    print('Model-5-SVR: time of training is %.4f seconds.'%(time.time()-start))#1secs
    
    # Decision Tree
    start = time.time()
    dt_per, dt_preds = BSM.DTreeReg(train_val_x, train_val_y, test_x, test_y)
    print('Model-6-DT: time of training is %.4f seconds.'%(time.time()-start))#1secs
    
    # Random Foreasts Regression
    start = time.time()
    rfs_per, rfs_preds, rfs_bst_para = BSM.RFsReg(train_val_x, train_val_y, test_x, test_y, test_fold)
    print('Model-7-RFs: time of training is %.4f seconds.'%(time.time()-start))#1secs
    
    # Adaboost Regression
    start = time.time()
    ada_per, ada_preds, ada_bst_para = BSM.AdaBoostReg(train_val_x, train_val_y, test_x, test_y, test_fold)
    print('Model-8-Adaboost: time of training is %.4f seconds.'%(time.time()-start))#1secs
    
    # XGBoost Regression
    start = time.time()
    xgb_per, xgb_preds, xgb_bst_para = BSM.XGBReg(train_val_x, train_val_y, test_x, test_y, test_fold)
    print('Model-9-XGBoost: time of training is %.4f seconds.'%(time.time()-start))#1secs
    
    # LGBM Regression
    start = time.time()
    lgbm_per, lgbm_preds, lgbm_bst_para = BSM.LGBMReg(train_val_x, train_val_y, test_x, test_y, test_fold)
    print('Model-10-LGBM: time of training is %.4f seconds.'%(time.time()-start))#1secs
    
    # MLP Regression
    #start = time.time()
    #mlp_per, mlp_preds, mlp_bst_para = BSM.MLPReg(train_val_x, train_val_ty, test_x, test_y, test_fold)
    #print('Model-11-MLP: time of training is %.4f seconds.'%(time.time()-start))#1secs
    
    ############ Results
    # Performance
    info_per = dt.rbind(lr_per, ridge_per, lasso_per, elanet_per, svr_per, dt_per,
                        rfs_per, ada_per, xgb_per, lgbm_per)
    info_per = dt.cbind(algorithms, info_per)
    info_per.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/BaselinePerformance.csv"))
    # Prediction
    all_preds = dt.Frame(np.column_stack((lr_preds, ridge_preds, lasso_preds, elanet_preds, svr_preds, dt_preds,
                        rfs_preds, ada_preds, xgb_preds, lgbm_preds)))
    all_preds.names = method_list
    all_preds.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/BaselinePrediction.csv"))
    # Best hyper-parameters
    os.makedirs("./output/table/Experiment-{}/HyperP".format(i), exist_ok=True)
    ridge_bst_para.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/HyperP/Ridge.csv"))
    lasso_bst_para.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/HyperP/Lasso.csv"))
    elanet_bst_para.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/HyperP/ElaNet.csv"))
    svr_bst_para.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/HyperP/SVR.csv"))
    rfs_bst_para.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/HyperP/RFs.csv"))
    ada_bst_para.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/HyperP/AdaBoost.csv"))
    xgb_bst_para.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/HyperP/XGBoost.csv"))
    lgbm_bst_para.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/HyperP/LGBM.csv"))
    #mlp_bst_para.to_csv("{}{}{}".format("./output/table/Experiment-", i, "/HyperP/MLP.csv"))

    print('time of experiments on this data set is %.4f seconds.'%(time.time()-This_Exp_Start))  #1secs
    bar.next()
    time.sleep(1)

bar.finish()
print('time of experiments on the whole data set is %.4f seconds.'%(time.time()-All_Start_Time)) #1secs