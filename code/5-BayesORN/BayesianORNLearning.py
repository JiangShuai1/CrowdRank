# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:26:35 2024

@author: jiangshuai
"""

import os
import sys
import platform
    
import time
from datatable import dt
import optuna
import timeit
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import BaseFunction as BSF
import BayesianORNModels as BORN
from Early_Stopping import EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"



EXP_IDs = [1,2,3]
out_idx = "-6"
NNs = ["BayesianORN"]
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
# runs_num = 20
epoch_num = 100
trial_num = 50
test_attn_layers = 3
sample_num = 20


for EXP_ID in EXP_IDs:
    print("{}{}{}".format("Experiment-", EXP_ID, " is beginning now!"))
    # training parameters setting
    
    # experiment data loading
    start_time = timeit.default_timer()
    train_X_1, train_X_2, train_Y, val_X_1, val_X_2, val_Y, size_tuple_list = BSF.LoadTrainData(EXP_ID, SelFeas) 
    end_time = timeit.default_timer()
    print(f"Time of data loading and preprocessing is : {end_time - start_time} secs.")
    del (start_time, end_time)
    # Create Dataset instances for training and validation sets
    train_dataset = BSF.PairwiseDataset(train_X_1, train_X_2, train_Y)
    val_dataset = BSF.PairwiseDataset(val_X_1, val_X_2, val_Y) 
    '''
    # data split
    train_indices = [i for i, m in enumerate(Mask) if m == 'Train']
    val_indices = [i for i, m in enumerate(Mask) if m == 'Val']
    test_indices = [i for i, m in enumerate(Mask) if m == 'Test']

    train_mask = [Mask[i] for i in train_indices]
    val_mask = [Mask[i] for i in val_indices]
    test_mask = [Mask[i] for i in test_indices]
    # Create Dataset instances for training, validation and test sets
    train_dataset = BSF.PairwiseDataset(X_1[train_indices], X_2[train_indices], Y[train_indices], train_mask)
    val_dataset = BSF.PairwiseDataset(X_1[val_indices], X_2[val_indices], Y[val_indices], val_mask)
    test_dataset = BSF.PairwiseDataset(X_1[test_indices], X_2[test_indices], Y[test_indices], test_mask)
    '''
     
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
        clip_grad = 0.1 if nn_type in ["RankDCN","RankAutoIntNet","OpinionRankNet", "BayesianORN"] else 0.1  # gradient clip value 0.1 for other models, 0.01 for AutoIntNet
        # Define the objective function for Optuna.
        def objective(trial):
            # Generate the model.
            if nn_type == "BayesianORN":
                model = BORN.BayesianORN(trial, size_tuple_list, attn_layers=test_attn_layers)
                    
                    
            # GPU Parallel
            if torch.cuda.device_count() > 1:
                print("Use", torch.cuda.device_count(), 'GPUs')
                model = nn.DataParallel(model, device_ids=[0,1]).cuda()
            
            # Generate the optimizers.
            lr = trial.suggest_float("lr", 1e-4, 1e-2, step=1e-4)
            wd = trial.suggest_float("wd", 0.1, 1.5, step=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=3, verbose=True)
            batch_size = 10000   # trial.suggest_int("batch_size", 6000, 14000, step=2000)
            # batch_size = trial.suggest_int("batch_size", 6000, 14000, step=2000)
            # Generate data loader.
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Optimization Target
            train_criterion = nn.BCELoss()
            val_criterion = nn.BCELoss()
            
            train_losses = []
            val_losses = []
            save_path = "{}{}{}{}{}".format("./output/table/Experiment-", EXP_ID, "/Pairwise/", nn_type, out_idx)  # key path
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
                    X_1, X_2, Y = data
                    X_1 = X_1.to(device=mydevice, dtype=torch.float)
                    X_2 = X_2.to(device=mydevice, dtype=torch.float)
                    Y = Y.to(device=mydevice, dtype=torch.float)
                    outputs = model(X_1, X_2)
                    loss = train_criterion(outputs, Y)
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
                val_preds = []
                all_val_Y = []
                model.eval()
                BSF.EnableDropout(model)
                with torch.no_grad():
                    for i, val_data in enumerate(val_loader):
                        val_X_1, val_X_2, val_Y = val_data
                        val_X_1 = val_X_1.to(device=mydevice, dtype=torch.float)
                        val_X_2 = val_X_2.to(device=mydevice, dtype=torch.float)
                        val_Y = val_Y.to(device=mydevice, dtype=torch.float)
                        all_val_Y.append(val_Y)
                        batch_val_preds = []

                        for j in range(sample_num):
                            batch_val_preds.append(model(val_X_1, val_X_2).unsqueeze(1))

                        batch_val_preds = torch.cat(batch_val_preds, dim=1)   # [batch_size, sample_num]


                        batch_val_preds_mean = torch.mean(batch_val_preds, dim=1) # batchsize
                        batch_val_preds_var =   torch.var(batch_val_preds, dim=1) # batchsize
                        batch_val_preds = torch.cat([batch_val_preds_mean.unsqueeze(dim=1),
                                                 batch_val_preds_var.unsqueeze(dim=1)], dim=1) # [batch_size, 2]
                        val_preds.append(batch_val_preds)  # [len(val_loader), batch_size, 2]
                        
                    val_preds = torch.cat(val_preds, dim=0)  # [dataset_size, 2]
                    # print(torch.isnan(val_preds).any(),torch.isnan(val_preds).all())
                    all_val_Y = torch.cat(all_val_Y, dim=0) # [dataset_size]
                    epoch_valid_loss = val_criterion(val_preds[:, 0], all_val_Y).item()
                    val_losses.append(epoch_valid_loss)
                    print('time of epoch validation is %.4f seconds.' % (time.time() - epoch_val_time))
                    # scheduler.step(epoch_valid_loss)
                    
                    if epoch_valid_loss < best_val_loss:
                        best_val_loss = epoch_valid_loss

                    if epoch + 1 >= min_epoch:
                        early_stopping(epoch_valid_loss, model)
                        if early_stopping.early_stop:
                            print("Early Stopping Now！")
                            break
                
                #trial.report(best_val_loss, trial.number)
                trial.report(epoch_valid_loss, epoch)
                '''  
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                '''
            

            print("Model Learning is Completed with Time %.4f Seconds!" % (time.time() - train_start_time))
            info_loss = dt.Frame({'Train': train_losses, 'Validation': val_losses})
            info_loss.to_csv(os.path.join(save_path, "{}{}".format(trial.number, "_InfoLoss.csv")))

            # trained_model = model.load_state_dict(torch.load(os.path.join(save_path, 'trained_network.pth')))
            # path = os.path.join(save_path, "{}{}".format(trial.number, "_TrainedModel.pth"))
            # torch.save(model.state_dict(), path)
            return best_val_loss

        # Optuna study.
        All_Start_Time = time.time()
        # Optuna setting
        study_name = "PairwiseNN_{}_{}".format(nn_type, EXP_ID)
        study = optuna.create_study(study_name=study_name, direction="minimize")
        study.optimize(objective, n_trials=trial_num)
        print("{}{}{}".format("NN type-", nn_type, " is finished!"))
        print('NNModel: time of training is %.4f seconds.' % (time.time() - All_Start_Time))  # 1secs
        
        trial = study.best_trial
        study_df = study.trials_dataframe()
        model_save_path = "{}{}{}{}{}".format("./output/table/Experiment-", EXP_ID, "/Pairwise/", nn_type, out_idx)  # key path
        learned_model_path = os.path.join(model_save_path, "{}{}".format(trial.number, "_TrainedModel.pth"))
        
        
        if nn_type == "BayesianORN":
            learned_model = BORN.BayesianORN(trial, size_tuple_list, attn_layers=test_attn_layers).to(mydevice)
        
        learned_model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(learned_model_path).items()})
        
        for Test_ID in range(5):
            test_X_1, test_X_2, test_Y = BSF.LoadTestData(EXP_ID, Test_ID+1, SelFeas)
            test_dataset = BSF.PairwiseDataset(test_X_1, test_X_2, test_Y)
            print("Test ID: ", Test_ID+1)
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

        test_per, all_preds = BSF.GetBayesianRankPer(all_test_y, all_preds)
        # key path
        all_preds.to_csv(os.path.join(model_save_path, "Predictions.csv"))
        test_per.to_csv(os.path.join(model_save_path, "Performance.csv"))
        study_df.to_csv(os.path.join(model_save_path, "OptunaStudy.csv"))

        del (study, trial, study_df, learned_model, all_preds, test_per)
        torch.cuda.empty_cache()
    
    del(train_dataset, val_dataset)
    print("{}{}{}".format("Experiment-", EXP_ID, " has done!"))    # 1secs

