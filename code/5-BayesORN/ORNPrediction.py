
# This code is used to predict the opinion rank of the test data using the learned model.

import os
import sys
import platform
    
import time
import pandas as pd
import datatable as dt
import optuna
import timeit
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import BaseFunction as BSF
import BayesianORNModels as RANKNN
from Early_Stopping import EarlyStopping
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


EXP_IDs = [1, 2, 3]
test_attn_layers = 3
NNs = ["BayesianORN"]
sample_num = 20

SelFeas = [
    "AnaNum","Gender","Degree","StarTimes","AnaReportNum","AnaIndustryNum","AnaStockNum",
    "Per_ME_Err","Per_SD_Err","Per_ME_APE","Per_SD_APE","BrokerName","ActiveAnaNum",
    "BroReportNum","BroIndustryNum","BroStockNum","BroP_ME_APE","BroP_SD_APE",
    "IndustryID","ListYear","Udwnm","PE","PB","PS","Turnover","Liquidility","CircuMarketValue",
    "Volatility","Beta","Corr","NonSysRisk","StockPrice","ChangeRatio","CAR5","CAR20","CAR60",
    "StdRank","RankChan","FEPS","RevHorizon","ForHorizon","PrevEPS","AnaStockReportNum","AnaStockIndustryNum",
    "StockPer_ME_Err","StockPer_SD_Err","StockPer_ME_APE","StockPer_SD_APE","PreForNum","Boldness"]



class SimulatedOptimizationTrial:
    def __init__(self, params):
        self.params = params
        
    def suggest_int(self, name, low, high, step=1, log=False):
        if name in self.params and (self.params[name] >= low and self.params[name] <= high and (self.params[name] - low) % step == 0):
            return self.params[name]
        else:
            raise ValueError(f"Invalid value for parameter {name}. Expected an integer within range [{low}, {high}] with step {step}.")
    
    def suggest_float(self, name, low, high, step=0.0, log=False):
        if name in self.params and low <= self.params[name] <= high:
            # 更正步长检查逻辑
            if step > 0 and not np.isclose((self.params[name] - low) / step, round((self.params[name] - low) / step), atol=1e-5):
                raise ValueError(f"Invalid value for parameter {name}. Expected a float within range [{low}, {high}] stepping by {step}.")
            return self.params[name]
        else:
            raise ValueError(f"Invalid value for parameter {name}. Expected a float within range [{low}, {high}].")


for EXP_ID in EXP_IDs:
    print("{}{}{}".format("Experiment-", EXP_ID, " is beginning now!"))
    # training parameters setting
    
    # experiment data loading
    start_time = timeit.default_timer()
    _, _, _, _, _, _, size_tuple_list = BSF.LoadTrainData(EXP_ID, SelFeas) 
    end_time = timeit.default_timer()
    print(f"Time of data loading and preprocessing is : {end_time - start_time} secs.")
    del (start_time, end_time)

    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        mydevice = torch.device('cuda:0')
    else:
        mydevice = torch.device('cpu')


    # 实例化优化器
    optuna_study =pd.read_csv("{}{}{}".format("./output/table/Experiment-", EXP_ID, "/Pairwise/BayesianORN/OptunaStudy.csv"))
    best_trial_index = optuna_study['value'].idxmin()
    # 获取最佳试验的所有信息
    best_trial_data = optuna_study.iloc[best_trial_index]
    best_params = {
        'dropout_rate': best_trial_data['params_dropout_rate'],
        'embedding_size': int(best_trial_data['params_embedding_size']),
        'hidden_dims': int(best_trial_data['params_hidden_dims']), 
        'hidden_nums': int(best_trial_data['params_hidden_nums'])}
    # print(best_trial)
    # print(best_trial.suggest_int("hidden_dims", 128, 1024, step=128))

    # 使用之前提取的最佳参数实例化模拟的Trial对象
    best_trial = SimulatedOptimizationTrial(best_params)
    #print(best_trial.suggest_float("dropout_rate", 0.3, 0.7, step=0.05))

    # 实例化模型
    model_save_path = "{}{}{}".format("./output/table/Experiment-", EXP_ID, "/Pairwise/BayesianORN")  # key path
    learned_model_path = os.path.join(model_save_path, "{}{}".format(best_trial_index, "_TrainedModel.pth"))
    learned_model = RANKNN.OpinionRankNet(best_trial, size_tuple_list, attn_layers=test_attn_layers).to(mydevice)
    learned_model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(learned_model_path).items()})
    print("Successfully loaded the learned model.")


    for Test_ID in range(5):
        test_X_1, test_X_2, test_Y = BSF.LoadTestData(EXP_ID, Test_ID+1, SelFeas)
        test_dataset = BSF.PairwiseDataset(test_X_1, test_X_2, test_Y)

        # non bayesian ranknet predictions
        nb_test_preds = BSF.RankNetPreds(model=learned_model, test_dataset=test_dataset, mydevice=mydevice,
                                      batch_size=5000, num_outputs=1)
        if Test_ID == 0:
            nb_all_test_y = test_Y
            nb_all_preds = nb_test_preds
        else:
            nb_all_test_y = np.hstack((nb_all_test_y, test_Y))
            nb_all_preds = np.hstack((nb_all_preds, nb_test_preds))            


        # bayesian ranknet predictions
        test_preds, test_preds_sample = BSF.BayesianRankNetPreds(model=learned_model, test_dataset=test_dataset, mydevice=mydevice,
                                            batch_size=10000, sample_num=sample_num)
        np.save(os.path.join(model_save_path, "Test_Preds_{}.npy".format(Test_ID+1)), test_preds_sample)
        if Test_ID == 0:
            all_test_y = test_Y
            all_preds = test_preds
        else:
            all_test_y = np.hstack((all_test_y, test_Y))  # [dataset_size]
            all_preds = np.vstack((all_preds, test_preds))  # [dataset_size, 2]
            
        del(test_X_1, test_X_2, test_Y, test_preds, test_dataset)


    # bayesian                                                                                                      
    test_per, all_preds = BSF.GetBayesianRankPer(all_test_y, all_preds)
    # key path
    all_preds.to_csv(os.path.join(model_save_path, "Predictions.csv"))
    test_per.to_csv(os.path.join(model_save_path, "Performance.csv"))

    # non bayesian
    nb_test_per = BSF.GetRankPerformance(nb_all_test_y, nb_all_preds)
    nb_all_preds = dt.Frame(nb_all_preds)
    # predictions.names = ['Predictions', 'Var']
    nb_all_preds.names = ['Predictions']
    # key path
    nb_all_preds.to_csv(os.path.join("{}{}{}".format("./output/table/Experiment-", EXP_ID, "/Pairwise/NBORN"), "Predictions.csv"))
    nb_test_per.to_csv(os.path.join("{}{}{}".format("./output/table/Experiment-", EXP_ID, "/Pairwise/NBORN"), "Performance.csv"))

    del(best_trial_data, best_trial_index, best_params, model_save_path, learned_model_path, learned_model)
    del(nb_all_test_y, nb_all_preds, nb_test_per, all_preds, all_test_y, test_per)
    torch.cuda.empty_cache()
    print("{}{}{}".format("Experiment-", EXP_ID, " has done!"))    # 1secs


    








                        