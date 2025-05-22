# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:46:55 2021

@author: 73243
"""
import json
import numpy as np
import pandas as pd
from datatable import dt
from collections import namedtuple
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.metrics import median_absolute_error, mean_absolute_percentage_error, explained_variance_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss


def LoadData(EXP_ID, SelFeas):
    file_path = "{}{}{}".format("./data/Experiment-",EXP_ID,"/ReportData.csv")
    data = dt.fread(file_path).to_pandas()
    Mask = data["Mask"].tolist()
    Y = np.array(data["APE"])
    data = data[SelFeas]
    num_cols = [col for col in data.columns if str(data[col].dtype) != 'object']
    size_tuple = namedtuple('size_tuple', ('name', 'vocab_size'))  # Returns a new subclass of tuple with named fields
    size_tuple_list = []
    for col in data.columns:
        if col in num_cols:
            v_size = 1
            data[col] = winsorize(data[col], limits=[0.0001, 0.0001])
            # normalize
            col_mean = data[col].mean()
            col_std = data[col].std()
            data[col] = (data[col] - col_mean) / col_std
        elif col!='Mask':
            data.loc[pd.isnull(data[col]), col]='N/A'
            le = LabelEncoder()
            le.fit(data[col])
            v_size = len(le.classes_)
            data[col] = le.transform(data[col])  # 对目标标签进行编码，值在0到n_class -1之间
            del(le)

        size_tuple_list.append(size_tuple(name=col, vocab_size=v_size))
    
    X = np.array(data)
    return X, Y, Mask, size_tuple_list




class MyDataset(Dataset):
    def __init__(self, X, Y, Mask):
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(Y).float()
        self.Mask = Mask

    def __len__(self):
        return len(self.Mask)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X[idx]
        y = self.Y[idx]
        #mask = self.Mask[idx]

        return x, y


def Get_Performance(targets, preds):
    MSE = mean_squared_error(targets, preds)
    MAE = mean_absolute_error(targets, preds)
    MSLE = mean_squared_log_error(targets, preds)
    EVS = explained_variance_score(targets, preds)
    R2Score = r2_score(targets, preds)
    MDAE = median_absolute_error(targets, preds)
    MAPE = mean_absolute_percentage_error(targets, preds)

    info_per = dt.Frame({'MSE': MSE, 'MAE': MAE, 'MSLE': MSLE, 'EVS': EVS,
                         'R2Score': R2Score, 'MADE': MDAE, 'MAPE': MAPE})
    return info_per



def Evaluate(model, test_dataset, mydevice, batch_size, num_outputs=1):
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=20)
    model.eval()
    preds = []
    all_test_Y = []
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_X, test_Y = test_data
            test_X = test_X.to(device=mydevice, dtype=torch.float)
            test_Y = test_Y.to(device=mydevice, dtype=torch.float)
            all_test_Y.append(test_Y)
            batch_preds = model(test_X)
            preds.append(batch_preds)    

            print(f'Test Batch: {i} out of {len(test_loader)}')
        
 
    preds = torch.cat(preds, dim=0)
    all_test_Y = torch.cat(all_test_Y, dim=0)
    preds = preds.to(device=torch.device('cpu')).numpy()
    all_test_Y = all_test_Y.to(device=torch.device('cpu')).numpy()
        
    info_per = Get_Performance(all_test_Y, preds)
        
    preds = dt.Frame(preds)
    # predictions.names = ['Predictions', 'Var']
    preds.names = ['Predictions']

    return preds, info_per



def LoadJsonData(data_path, SelFeas):
    X_1 = []  # Opinion 1
    X_2 = []  # Opinion 2
    Y = []    # RankLabel
    
    with open(data_path, 'r') as data_file:
        data = json.load(data_file)
        
    for sample in data.values():
        sample_X_1 = []
        sample_X_2 = []
    
        for feature in SelFeas:
            sample_X_1.append(sample['Opinion_1'][feature])
    
        for feature in SelFeas:
            sample_X_2.append(sample['Opinion_2'][feature])
    
        X_1.append(sample_X_1)
        X_2.append(sample_X_2)
        Y.append(sample['RankLabel'])
        del(sample_X_1, sample_X_2)
        # if len(Mask) % 10000 == 0:
        #     print(f"已处理样本数据: {len(Mask)} 条")
    del(data, data_path, data_file, sample)
    df_X_1 = pd.DataFrame(X_1)
    df_X_2 = pd.DataFrame(X_2)
    df_X_1 = df_X_1.set_axis(SelFeas, axis=1)
    df_X_2 = df_X_2.set_axis(SelFeas, axis=1)
    Y = np.array(Y)
    del(X_1, X_2)
    return df_X_1, df_X_2, Y
    



def LoadTrainData(EXP_ID, SelFeas):
    
    ### Training Data
    train_path = "{}{}{}".format("./data/Experiment-", EXP_ID, "/Train.json")
    train_X_1, train_X_2, train_Y = LoadJsonData(train_path, SelFeas)
     
    #### Validation Data
    val_path = "{}{}{}".format("./data/Experiment-", EXP_ID, "/Val.json")
    val_X_1, val_X_2, val_Y = LoadJsonData(val_path, SelFeas)
    
    
    del(train_path, val_path)
    
    ### Feature Preprocess
    file_path = "{}{}{}".format("./data/Experiment-",EXP_ID,"/ReportData.csv")
    # file_path = "./data/DataReport230914.csv"
    data = dt.fread(file_path).to_pandas()
    data = data[SelFeas]
    num_cols = [col for col in data.columns if str(data[col].dtype) != 'object']
    size_tuple = namedtuple('size_tuple', ('name', 'vocab_size'))  # Returns a new subclass of tuple with named fields
    size_tuple_list = []
    for col in data.columns:
        if col in num_cols:
            v_size = 1
            col_mean = data[col].mean()
            col_std = data[col].std()
            train_X_1[col] = winsorize(train_X_1[col], limits=[0.0001, 0.0001])
            train_X_2[col] = winsorize(train_X_2[col], limits=[0.0001, 0.0001])
            train_X_1[col] = (train_X_1[col] - col_mean) / col_std
            train_X_2[col] = (train_X_2[col] - col_mean) / col_std
            
            val_X_1[col] = winsorize(val_X_1[col], limits=[0.0001, 0.0001])
            val_X_2[col] = winsorize(val_X_2[col], limits=[0.0001, 0.0001])
            val_X_1[col] = (val_X_1[col] - col_mean) / col_std
            val_X_2[col] = (train_X_2[col] - col_mean) / col_std
        elif col!='Mask':
            data.loc[pd.isnull(data[col]), col]='N/A'
            train_X_1.loc[pd.isnull(train_X_1[col]), col]='N/A'
            train_X_2.loc[pd.isnull(train_X_2[col]), col]='N/A'
            val_X_1.loc[pd.isnull(val_X_1[col]), col]='N/A'
            val_X_2.loc[pd.isnull(val_X_2[col]), col]='N/A'
            train_X_1.loc[train_X_1[col]=='NA',col] = 'N/A'
            train_X_2.loc[train_X_2[col]=='NA',col] = 'N/A'
            val_X_1.loc[val_X_1[col]=='NA',col] = 'N/A'
            val_X_2.loc[val_X_2[col]=='NA',col] = 'N/A'
            le = LabelEncoder()
            le.fit(data[col])
            v_size = len(le.classes_)
            train_X_1[col] = le.transform(train_X_1[col])  # Encode target labels with values between 0 and n_class-1
            train_X_2[col] = le.transform(train_X_2[col])
            val_X_1[col] = le.transform(val_X_1[col])
            val_X_2[col] = le.transform(val_X_2[col])
            del(le)
    
        size_tuple_list.append(size_tuple(name=col, vocab_size=v_size))


    train_X_1 = np.array(train_X_1)
    train_X_2 = np.array(train_X_2)
    val_X_1 = np.array(val_X_1)
    val_X_2 = np.array(val_X_2)
        
    return train_X_1, train_X_2, train_Y, val_X_1, val_X_2, val_Y, size_tuple_list


def LoadTestData(EXP_ID, Test_ID, SelFeas):
    #### Test Data
    test_path = "{}{}{}{}{}".format("./data/Experiment-", EXP_ID, "/Test-", Test_ID,".json")
    test_X_1, test_X_2, test_Y = LoadJsonData(test_path, SelFeas)
    ### Feature Preprocess
    file_path = "{}{}{}".format("./data/Experiment-",EXP_ID,"/ReportData.csv")
    data = dt.fread(file_path).to_pandas()
    data = data[SelFeas]
    num_cols = [col for col in data.columns if str(data[col].dtype) != 'object']
    for col in data.columns:
        if col in num_cols:
            test_X_1[col] = winsorize(test_X_1[col], limits=[0.0001, 0.0001])
            test_X_2[col] = winsorize(test_X_2[col], limits=[0.0001, 0.0001])
            col_mean = data[col].mean()
            col_std = data[col].std()
            test_X_1[col] = (test_X_1[col] - col_mean) / col_std
            test_X_2[col] = (test_X_2[col] - col_mean) / col_std
        elif col!='Mask':
            data.loc[pd.isnull(data[col]), col]='N/A'
            test_X_1.loc[pd.isnull(test_X_1[col]), col]='N/A'
            test_X_2.loc[pd.isnull(test_X_2[col]), col]='N/A'
            test_X_1.loc[test_X_1[col]=='NA',col] = 'N/A'
            test_X_2.loc[test_X_2[col]=='NA',col] = 'N/A'
            le = LabelEncoder()
            le.fit(data[col])
            test_X_1[col] = le.transform(test_X_1[col])  # Encode target labels with values between 0 and n_class-1
            test_X_2[col] = le.transform(test_X_2[col])
            del(le)

    test_X_1 = np.array(test_X_1)
    test_X_2 = np.array(test_X_2)
    
    return test_X_1, test_X_2, test_Y
    
    

class PairwiseDataset(Dataset):
    def __init__(self, X_1, X_2, Y):
        self.X_1 = torch.tensor(X_1).float()
        self.X_2 = torch.tensor(X_2).float()
        self.Y = torch.tensor(Y).float()
        # self.Mask = Mask

    def __len__(self):
        return (self.Y).size()[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_1 = self.X_1[idx]
        x_2 = self.X_2[idx]
        y = self.Y[idx]
        # mask = self.Mask[idx]

        return x_1, x_2, y
    
    

def GetRankPerformance(targets, preds):
    pcls = (preds > 0.5)
    targets = (targets > 0.5)
    # preds = preds.to(device=torch.device('cpu')).numpy()
    # pcls = pcls.to(device=torch.device('cpu')).numpy()
    acc = accuracy_score(targets, pcls)
    precision = precision_score(targets, pcls, zero_division=0)
    recall = recall_score(targets, pcls, zero_division=0)
    f1 = f1_score(targets, pcls, zero_division=0)
    auc = roc_auc_score(targets, preds)
    logloss = -1 * log_loss(targets, preds, eps=1e-5)
    info_per = dt.Frame({'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'AUC': auc, 'Log-Loss': logloss})
    return info_per



def RankNetPreds(model, test_dataset, mydevice, batch_size, num_outputs=1):
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=20)
    model.eval()
    preds = []
    # all_test_Y = []
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_X_1, test_X_2, test_Y = test_data
            test_X_1 = test_X_1.to(device=mydevice, dtype=torch.float)
            test_X_2 = test_X_2.to(device=mydevice, dtype=torch.float)
            # test_Y = test_Y.to(device=mydevice, dtype=torch.float)
            # all_test_Y.append(test_Y)
            batch_preds = model(test_X_1, test_X_2)
            preds.append(batch_preds)    
            print(f'Test Batch: {i} out of {len(test_loader)}')
        
 
    preds = torch.cat(preds, dim=0)
    #all_test_Y = torch.cat(all_test_Y, dim=0)
    preds = preds.to(device=torch.device('cpu')).numpy()
    #all_test_Y = all_test_Y.to(device=torch.device('cpu')).numpy()
    '''    
    info_per = GetRankPerformance(all_test_Y, preds)
        
    preds = dt.Frame(preds)
    # predictions.names = ['Predictions', 'Var']
    preds.names = ['Predictions']
    '''
    return preds


def EnableDropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()



def BayesianRankNetPreds(model, test_dataset, mydevice, batch_size, sample_num=30):
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=20)
    model.eval()
    EnableDropout(model)
    preds = []
    preds_samples = []
    # all_test_Y = []
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_X_1, test_X_2, _ = test_data
            test_X_1 = test_X_1.to(device=mydevice, dtype=torch.float)
            test_X_2 = test_X_2.to(device=mydevice, dtype=torch.float)
            batch_test_preds = []

            for j in range(sample_num):
                batch_test_preds.append(model(test_X_1, test_X_2).unsqueeze(1))

            batch_test_preds = torch.cat(batch_test_preds, dim=1)   # [batch_size, sample_num]
            preds_samples.append(batch_test_preds)
            batch_test_preds_mean = torch.mean(batch_test_preds, dim=1) # batchsize
            batch_test_preds_var =   torch.var(batch_test_preds, dim=1) # batchsize
            batch_test_preds = torch.cat([batch_test_preds_mean.unsqueeze(dim=1),
                                         batch_test_preds_var.unsqueeze(dim=1)], dim=1) # [batch_size, 2]
            preds.append(batch_test_preds)  # [len(test_loader), batch_size, 2]
            print(f'Test Batch: {i} out of {len(test_loader)}')
        
 
    preds = torch.cat(preds, dim=0)  # [dataset_size, 2]
    preds_samples = torch.cat(preds_samples, dim=0)  # [dataset_size, sample_num]
    #all_test_Y = torch.cat(all_test_Y, dim=0)
    preds = preds.to(device=torch.device('cpu')).numpy()  # [dataset_size, 2]
    preds_samples = preds_samples.to(device=torch.device('cpu')).numpy()  # [dataset_size, sample_num]
    #all_test_Y = all_test_Y.to(device=torch.device('cpu')).numpy()
    '''    
    info_per = GetRankPerformance(all_test_Y, preds)
        
    preds = dt.Frame(preds)
    # predictions.names = ['Predictions', 'Var']
    preds.names = ['Predictions']
    '''
    return preds, preds_samples


def GetBayesianRankPer(targets, preds):
    # preds = preds.to(device=torch.device('cpu')).numpy()
    # pcls = pcls.to(device=torch.device('cpu')).numpy()

    q_series = 1 - np.linspace(0.00, 0.95, num=20)
    for q in q_series:
        var_q = np.quantile(preds[:, 1], q)
        preds_var, test_y_var = preds[preds[:,1] <= var_q, 0], targets[preds[:,1] <= var_q]
        per_var = GetRankPerformance(test_y_var, preds_var)
        if q == q_series[0]:
            info_per = per_var
        else:
            info_per = dt.rbind(info_per, per_var)

    preds = dt.Frame(preds)
    preds.names = ['Preds', 'Var']

    return info_per, preds


# 定义一个模拟 trial 对象的类
class MockTrial:
    def __init__(self, params):
        self.params = params

    def suggest_int(self, name, low, high, step=1):
        return self.params.get(name, None)

    def suggest_float(self, name, low, high, step=None):
        return self.params.get(name, None)