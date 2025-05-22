# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:16:23 2024

@author: jiangshuai
"""


import os
import sys
import platform
import gc
    
import time
import pandas as pd
import optuna
import timeit
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import DeepListwiseFunc as DLF
from Early_Stopping import EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

EXP_IDs = [1,2,3]
min_size = 5
NNs = ["ListNet", "ListGATNet"]
SelFeas = [
    "AnaNum","Gender","Degree","StarTimes","AnaReportNum","AnaIndustryNum","AnaStockNum",
    "Per_ME_Err","Per_SD_Err","Per_ME_APE","Per_SD_APE","BrokerName","ActiveAnaNum",
    "BroReportNum","BroIndustryNum","BroStockNum","BroP_ME_APE","BroP_SD_APE",
    "IndustryID","ListYear","Udwnm","PE","PB","PS","Turnover","Liquidility","CircuMarketValue",
    "Volatility","Beta","Corr","NonSysRisk","StockPrice","ChangeRatio","CAR5","CAR20","CAR60",
    "StdRank","RankChan","FEPS","RevHorizon","ForHorizon","PrevEPS","AnaStockReportNum","AnaStockIndustryNum",
    "StockPer_ME_Err","StockPer_SD_Err","StockPer_ME_APE","StockPer_SD_APE","PreForNum","Boldness"]


num_outputs = 1
es_patience = 5
min_epoch = 1
epoch_num = 100
trial_num = 30



for EXP_ID in EXP_IDs:
    print("{}{}{}".format("Experiment-", EXP_ID, " is beginning now!"))
    # training parameters setting
    
    # experiment data loading
    start_time = timeit.default_timer()
    qid, report_id, X, Y, Mask, size_tuple_list = DLF.LoadData(EXP_ID, SelFeas, min_size)
    end_time = timeit.default_timer()
    print(f"Time of data loading and preprocessing is : {end_time - start_time} secs.")
    del (start_time, end_time)
    # 创建训练集和验证集的Dataset实例
    train_dataset = DLF.ListwiseCustomDataset(qid, report_id, X, Y, Mask, size_tuple_list, mode='Train')
    val_dataset = DLF.ListwiseCustomDataset(qid, report_id, X, Y, Mask, size_tuple_list, mode='Val')
     
    """
        check if use cuda
    """
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        mydevice = torch.device('cuda:0')
    else:
        mydevice = torch.device('cpu')
    
    for nn_type in NNs:
        print("{}{}{}".format("NN type-", nn_type, " is beginning now!"))
        clip_grad = 0.1 if nn_type in ["RankDCN","RankAutoIntNet","OpinionRankNet"] else 0.1  # gradient clip value 0.1 for other models, 0.01 for AutoIntNet
        # Define the objective function for Optuna.
        def objective(trial):
            # Generate the model.
            if nn_type == "ListNet":
                model = DLF.ListNet(trial, size_tuple_list)
            elif nn_type == "ListGATNet":
                model = DLF.ListGATNet(trial, size_tuple_list)
                    
                    
            # GPU Parallel
            if torch.cuda.device_count() > 1:
                print("Use", torch.cuda.device_count(), 'GPUs')
                model = nn.DataParallel(model, device_ids=[0,1]).cuda()
            
            # Generate the optimizers.
            lr = trial.suggest_float("lr", 1e-4, 1e-2, step=1e-4)
            wd = trial.suggest_float("wd", 0.1, 1.5, step=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=3, verbose=True)
            batch_size = 64
            # batch_size = trial.suggest_int("batch_size", 6000, 14000, step=2000)
            # Generate data loader.
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=DLF.collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=DLF.collate_fn)

            # Optimization Target
            train_criterion = DLF.ListNetLoss()
            val_criterion = DLF.ListNetLoss()
            
            train_losses = []
            val_losses = []
            save_path = "{}{}{}{}".format("./output/table/Experiment-", EXP_ID, "/Listwise/", nn_type)  # key path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            early_stopping = EarlyStopping(save_path=save_path, patience=es_patience, trial=trial, verbose=True)
            # Training of the model.
            best_val_loss = float('inf')
            train_start_time = time.time()
            for epoch in range(epoch_num):
                epoch_train_time = time.time()
                train_loss = 0
                model.train()
                for i, data in enumerate(train_loader):
                    X = data['features']
                    Y = data['ranks']
                    doc_counts = data['doc_counts']
                    X = X.to(device=mydevice, dtype=torch.float)
                    Y = Y.to(device=mydevice, dtype=torch.float)
                    doc_counts = doc_counts.to(device=mydevice, dtype=torch.int32)
                    outputs = model(X)
                    loss = train_criterion(outputs, Y, doc_counts)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                    optimizer.step()
                    train_loss += loss.item()

                train_losses.append(train_loss / len(train_loader))
                print("epoch：%d" % epoch, "loss: %f" % (train_loss / len(train_loader)),
                    "epoch training: %.4f seconds" % (time.time() - epoch_train_time))

                # Validation of the model.
                epoch_val_time = time.time()
                epoch_val_losses = []
                model.eval()
                with torch.no_grad():
                    for i, val_data in enumerate(val_loader):
                        val_X = val_data['features']
                        val_Y = val_data['ranks']
                        val_doc_counts = val_data['doc_counts']
                        val_X = val_X.to(device=mydevice, dtype=torch.float)
                        val_Y = val_Y.to(device=mydevice, dtype=torch.float)   # (batch_size, max_doc_nums)
                        val_doc_counts = val_doc_counts.to(device=mydevice, dtype=torch.int32)
                        batch_val_preds=model(val_X)
                        batch_loss = val_criterion(batch_val_preds, val_Y, val_doc_counts).item()
                        epoch_val_losses.append(batch_loss)

                        
                    epoch_val_loss = np.mean(epoch_val_losses)
                    val_losses.append(epoch_val_loss)
                    print('time of epoch validation is %.4f seconds.' % (time.time() - epoch_val_time))
                    # scheduler.step(epoch_valid_loss)
                    
                    if epoch_val_loss < best_val_loss:
                        best_val_loss = epoch_val_loss

                    if epoch + 1 >= min_epoch:
                        early_stopping(epoch_val_loss, model)
                        if early_stopping.early_stop:
                            print("Early Stopping Now！")
                            break
                
                #trial.report(best_val_loss, trial.number)
                trial.report(epoch_val_loss, epoch)
                '''  
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                '''
            

            print("Model Learning is Completed with Time %.4f Seconds!" % (time.time() - train_start_time))
            info_loss = pd.DataFrame({'Train': train_losses, 'Validation': val_losses})
            info_loss.to_csv(os.path.join(save_path, "{}{}".format(trial.number, "_InfoLoss.csv")))

            # trained_model = model.load_state_dict(torch.load(os.path.join(save_path, 'trained_network.pth')))
            # path = os.path.join(save_path, "{}{}".format(trial.number, "_TrainedModel.pth"))
            # torch.save(model.state_dict(), path)
            return best_val_loss

        # Optuna study.
        All_Start_Time = time.time()
        # Optuna setting
        study_name = "ListwiseNN_{}_{}".format(nn_type, EXP_ID)
        study = optuna.create_study(study_name=study_name, direction="minimize")
        study.optimize(objective, n_trials=trial_num)
        print("{}{}{}".format("NN type-", nn_type, " is finished!"))
        print('NNModel: time of training is %.4f seconds.' % (time.time() - All_Start_Time))  # 1secs
        
        trial = study.best_trial
        study_df = study.trials_dataframe()
        model_save_path = "{}{}{}{}".format("./output/table/Experiment-", EXP_ID, "/Listwise/",nn_type)  # key path
        learned_model_path = os.path.join(model_save_path, "{}{}".format(trial.number, "_TrainedModel.pth"))
        
        
        if nn_type == "ListNet":
            learned_model = DLF.ListNet(trial, size_tuple_list).to(mydevice)
        elif nn_type == "ListGATNet":
            learned_model = DLF.ListGATNet(trial, size_tuple_list).to(mydevice)

        
        learned_model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(learned_model_path).items()})
        test_dataset = DLF.ListwiseCustomDataset(qid, report_id, X, Y, Mask, size_tuple_list, mode='Test')
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=20, collate_fn=DLF.collate_fn)
        test_preds = DLF.ListwisePreds(learned_model, test_loader, mydevice)
        # test_preds.names = ['ReportID', 'predictions']
        test_preds.to_csv(os.path.join(model_save_path, "predictions.csv"))
        study_df.to_csv(os.path.join(model_save_path, "OptunaStudy.csv"))

        del (study, trial, study_df, learned_model, test_preds)
        torch.cuda.empty_cache()
    
    del(train_dataset, val_dataset, test_dataset, test_loader)
    print("{}{}{}".format("Experiment-", EXP_ID, " has done!"))    # 1secs




