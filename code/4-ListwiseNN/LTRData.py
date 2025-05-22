# -*- coding: utf-8 -*-
"""
Created on Tus Oct 12 15:26:00 2024

@author: shuai jiang
"""


import os
import sys
import platform
    
import numpy as np
import pandas as pd
import time
import sys
from progress.bar import IncrementalBar
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats.mstats import winsorize
from scipy.stats import rankdata
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def LoadData(EXPID, SelFeas, min_size=5):
    file_path = f"./data/Experiment-{EXPID}/ReportData.csv"
    data = pd.read_csv(file_path)
        
    # Filter out groups with size less than min_size
    data = data[data['GroupSize'] >= min_size]
    test_id_info = data.loc[data['Mask'] == 'Test', ['GroupID','ReportID']]
    test_id_info.to_csv(f"./output/table/Experiment-{EXPID}/L2RData/TestID.csv", index=False)
        
    # Convert APE to rank within each group
    data['Rank'] = data.groupby('GroupID')['APE'].transform(lambda x: rankdata(-x, method='ordinal').astype(int))
    data['Rank'] = data['Rank'] - 1
    # 转换GroupID为整数
    data['GroupID'] = pd.factorize(data['GroupID'])[0] + 1
    
    data = data[SelFeas + ['Rank', 'GroupID']]
    
    cat_cols = [col for col in data.columns if str(data[col].dtype) == 'object' and col != 'Mask']
    num_cols = [col for col in data.columns if str(data[col].dtype) != 'object' and col != 'ReportID' and col != 'APE']
    
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
    val_x = data_new[data['Mask'] == 'Val']
    test_x = data_new[data['Mask'] == 'Test']
    
    
    train_y = data.loc[data['Mask'] == 'Train', 'Rank'].to_numpy()
    val_y = data.loc[data['Mask'] == 'Val', 'Rank'].to_numpy()
    test_y = data.loc[data['Mask'] == 'Test', 'Rank'].to_numpy()
    
    train_qid = data.loc[data['Mask'] == 'Train', 'GroupID'].to_numpy()
    val_qid = data.loc[data['Mask'] == 'Val', 'GroupID'].to_numpy()
    test_qid = data.loc[data['Mask'] == 'Test', 'GroupID'].to_numpy()

    return train_x, train_y, val_x, val_y, test_x, test_y, train_qid, val_qid, test_qid



def convert_to_svm_light_format(x, y, qid, output_file):
    with open(output_file, 'w') as f:
        for i in range(len(y)):
            features = ' '.join([f"{j+1}:{x[i][j]}" for j in range(x.shape[1])])
            line = f"{y[i]} qid:{qid[i]} {features}\n"
            f.write(line)


SelFeas = [
    "AnaNum","Gender","Degree","StarTimes","AnaReportNum","AnaIndustryNum","AnaStockNum",
    "Per_ME_Err","Per_SD_Err","Per_ME_APE","Per_SD_APE","BrokerName","ActiveAnaNum",
    "BroReportNum","BroIndustryNum","BroStockNum","BroP_ME_APE","BroP_SD_APE",
    "IndustryID","ListYear","Udwnm","PE","PB","PS","Turnover","Liquidility","CircuMarketValue",
    "Volatility","Beta","Corr","NonSysRisk","StockPrice","ChangeRatio","CAR5","CAR20","CAR60",
    "StdRank","RankChan","FEPS","RevHorizon","ForHorizon","PrevEPS","AnaStockReportNum","AnaStockIndustryNum",
    "StockPer_ME_Err","StockPer_SD_Err","StockPer_ME_APE","StockPer_SD_APE","PreForNum","Boldness",
    "APE","Mask"]


min_size=5
EXPIDS = [1, 2, 3]

for EXPID in EXPIDS:
    print(f"Experiment {EXPID}")
    train_x, train_y, val_x, val_y, test_x, test_y, train_qid, val_qid, test_qid = LoadData(EXPID, SelFeas)
    convert_to_svm_light_format(train_x, train_y, train_qid, f"./output/table/Experiment-{EXPID}/L2RData/train.txt")
    convert_to_svm_light_format(val_x, val_y, val_qid, f"./output/table/Experiment-{EXPID}/L2RData/val.txt")
    convert_to_svm_light_format(test_x, test_y, test_qid, f"./output/table/Experiment-{EXPID}/L2RData/test.txt")


