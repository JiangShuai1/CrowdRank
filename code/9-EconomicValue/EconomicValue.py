# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:44:58 2024

@author: jiangshuai
"""


import os
import sys
import platform    
import time
import shutil
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import EPSPredFunc as EPSPF


code_test = 0
EXP_IDs = [1,2,3]
SelFeas = ['ReportID', 'GroupID', 'StockID', 'ReportDate', 'FEndDate', 'FEPS', 'PrevEPS', 'AEPS', 'StockPrice', 'PE', 'PB', 'Volatility', 'ForHorizon', 'CAR20', 'FErr', 'AFErr', 'APE', 'GroupSize']
models = ['rs3', 'DCN_score', 'RankEWDNN_score_ep', 'RankAutoIntNet_score_ep', 'BORN_score_ep']
models_name = ['RS(pve)', 'DCN', 'RankNet', 'RN-AutoInt', 'CrowdRank']
directions = ['pos', 'neg', 'pos', 'pos', 'pos']
days = 251
strategy = 'TopKSoftmax'
top_rates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.45]

print("Get price data")
start_time = time.time()
price_path = "./data/股票历史行情信息表-前复权"
price_df, index_df = EPSPF.GetPriceData(price_path, start_date=pd.to_datetime('2018-01-01'), end_date=pd.to_datetime('2023-12-31'))
print("running time of GetPriceData is：", time.time() - start_time)
# 打印price_df的日期范围
print("date range of price_df is: ", price_df['TradingDate'].min(), price_df['TradingDate'].max())
del price_path, start_time


for EXP_ID in EXP_IDs:
    print("{}{}{}".format("Experiment-", EXP_ID, " is beginning now!"))
    # 在output/table/Experiment-EXP_ID文件夹下创建一个EconomicValue的文件夹,如果已经存在则覆盖
    output_path = "./output/table/Experiment-{}/EconomicValue".format(EXP_ID)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    # 加载数据
    data, rank = EPSPF.LoadData(EXP_ID, SelFeas)
    # 打印data的列名
    print(data.columns)
    print("length of group rank in experiment-{} is: ".format(EXP_ID), len(rank))

    if code_test:
        # code test
        group_rank = rank[0]
        group_rank = pd.DataFrame(group_rank)   # 将字典group_rank转换为DataFrame
        print(group_rank.head())   # 打印group_rank的前5行
        print(group_rank.columns)  # 打印group_rank的列名
        start_time = time.time()
        tar_report, stock_croi, stock_car = EPSPF.GetEPSPred(0, rank, data, models, directions, price_df, index_df, strategy=strategy, days=days)  # 计算group_rank的第10个元素的tar_report, stock_croi, stock_car
        print("running time of GetEPSPred is：", time.time() - start_time)
        print(tar_report.head())
        print("length of stock_croi: ", len(stock_croi), "class of stock_croi: ", type(stock_croi))
        print("length of stock_car: ", len(stock_car), "class of stock_car: ", type(stock_car))

    else:
        # 记录时间
        start_time = time.time()
        exp_tar_report, exp_stock_roi, exp_stock_ar, exp_stock_croi, exp_stock_car = EPSPF.GetAllEPSPred(rank, data, models, directions, price_df, index_df, strategy=strategy, days=days)
        print("running time of GetAllEPSPred is：", time.time() - start_time)
        # 将tar_report (dataframe), stock_croi (numpy array), stock_car (numpy array)分别保存为csv与npy文件
        exp_tar_report.to_csv(output_path + "/tar_report.csv", index=False)
        np.save(output_path + "/stock_croi.npy", exp_stock_croi)
        np.save(output_path + "/stock_car.npy", exp_stock_car)

        exp_tar_report, exp_stock_croi, exp_stock_car = EPSPF.FilterTarReport(exp_tar_report, exp_stock_croi, exp_stock_car)

        #将每次实验的tar_report, stock_croi, stock_car合并
        if EXP_ID == 1:
            tar_report = exp_tar_report
            stock_roi = exp_stock_roi
            stock_ar = exp_stock_ar
            stock_croi = exp_stock_croi
            stock_car = exp_stock_car
        else:
            tar_report = pd.concat([tar_report, exp_tar_report], ignore_index=True)
            stock_roi = np.concatenate([stock_roi, exp_stock_roi], axis=0)
            stock_ar = np.concatenate([stock_ar, exp_stock_ar], axis=0)
            stock_croi = np.concatenate([stock_croi, exp_stock_croi], axis=0)
            stock_car = np.concatenate([stock_car, exp_stock_car], axis=0)

    print("Experiment-{} is finished!".format(EXP_ID))


# 计算average_croi_df与average_car_df
output_path = "./output/table/EconomicValue"
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)
for top_rate in top_rates:
    print("Top rate is: {}".format(top_rate))
    average_croi_df, average_car_df = EPSPF.ReturnValidation(tar_report, stock_croi, stock_car, models, top_rate, days)
    average_croi_df, average_car_df = average_croi_df.T, average_car_df.T
    average_croi_df.columns = models_name
    average_car_df.columns = models_name
    average_croi_df.to_csv(output_path + "/ave_croi_{}.csv".format(int(top_rate*100)), index=True)  
    average_car_df.to_csv(output_path + "/ave_car_{}.csv".format(int(top_rate*100)), index=True)

