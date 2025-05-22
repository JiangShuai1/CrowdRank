# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:46:55 2021

@author: 73243
"""

import os
import json
import numpy as np
import pandas as pd
import math
from datatable import dt
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score
from multiprocessing import Pool
from functools import partial
import scipy.stats as stats
from scipy.linalg import eig
import choix
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import scipy.sparse
import scipy.sparse.linalg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
# from joblib import parallel_backend
import pickle
from scipy.special import softmax




def ScoreToRank(scores, direct="neg"):
    sorted_index = sorted(range(len(scores)), key=lambda x: scores[x])
    if direct != "neg":
        rank = [sorted_index.index(i) + 1 for i in range(len(scores))]   ## score越大， rank分数越大
    else:
        rank = [len(scores) - sorted_index.index(i) for i in range(len(scores))]
    return rank


def GetPreds(EXP_ID, ListwiseTree, ListwiseDeep):
    data = pd.read_csv(f"./data/Experiment-{EXP_ID}/ReportData.csv")
    test_ids = pd.read_csv(f"./output/table/Experiment-{EXP_ID}/L2RData/TestID.csv")
    data = test_ids.merge(data, on=['GroupID', 'ReportID'], how='left')
    data.reset_index(drop=True, inplace=True)
        
    # listwise
    for model in ListwiseTree:
        tree_preds_path = f"./output/table/Experiment-{EXP_ID}/Listwise/{model}/predictions.csv"
        if model == ListwiseTree[0] :
            tree_preds = pd.read_csv(tree_preds_path)['prediction']               
        else:
            tree_preds = pd.concat([tree_preds, pd.read_csv(tree_preds_path)['prediction']], axis=1)
    
    tree_preds.columns = ListwiseTree
    data = pd.concat([data, tree_preds], axis=1)

    # deep
    for model in ListwiseDeep:
        deep_preds_path = f"./output/table/Experiment-{EXP_ID}/Listwise/{model}/predictions.csv"
        if model == ListwiseDeep[0] :
            deep_preds = pd.read_csv(deep_preds_path)[['ReportID','prediction']]               
        else:
            deep_preds = deep_preds.merge(pd.read_csv(deep_preds_path)[['ReportID','prediction']], on='ReportID', how='left')
    
    deep_preds.columns = ['ReportID'] + ListwiseDeep

    data = data.merge(deep_preds, on='ReportID', how='left')        
    return data



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
    Y = pd.DataFrame(Y)
    del(X_1, X_2)
    return df_X_1, df_X_2, Y




def PairPreds(EXP_ID, Test_IDs, SelFeas, RankNNs):
    for Test_ID in Test_IDs:
        test_path = "{}{}{}{}{}".format("./data/Experiment-", EXP_ID, "/Test-", Test_ID,".json")
        test_X_1, test_X_2, test_Y = LoadJsonData(test_path, SelFeas)
        if Test_ID == 1:
            opinion_1 = test_X_1
            opinion_2 = test_X_2
            truth = test_Y
        else:
            opinion_1 = pd.concat([opinion_1, test_X_1], axis=0)
            opinion_2 = pd.concat([opinion_2, test_X_2], axis=0)
            truth = pd.concat((truth, test_Y), axis=0)
        del(test_X_1, test_X_2, test_Y, test_path)    
    
    opinion_1.columns = ['GroupID', 'OID_1', 'APE_1']
    opinion_2.columns = ['GroupID', 'OID_2', 'APE_2']
    truth.columns = ['Truth']
    pairs = pd.concat([opinion_1, opinion_2[['OID_2', 'APE_2']], truth], axis=1)
    pairs.reset_index(drop=True, inplace=True)
    del(opinion_1, opinion_2, truth)
    
    for k in range(len(RankNNs)):
        pred_path = "{}{}{}{}{}".format("./output/table/Experiment-", EXP_ID, "/Pairwise/", RankNNs[k],"/Predictions.csv")
        temp_pred = dt.fread(pred_path).to_pandas()
        if k == 0 :
            preds = temp_pred
        else:
            preds = pd.concat([preds, temp_pred], axis=1)
        
        del(pred_path, temp_pred)
        
    preds.columns = RankNNs
    preds.reset_index(drop=True, inplace=True)
    pairs_preds = pd.concat([pairs, preds], axis=1)
    group_id_set = pairs_preds['GroupID'].unique()
    del(pairs, preds)
    return pairs_preds, group_id_set


def ReadBORNPred(EXP_ID, Test_IDs, pairs_preds):
    model_save_path = "{}{}{}".format("./output/table/Experiment-", EXP_ID, "/Pairwise/BayesianORN") 
    for Test_ID in Test_IDs:
        temp_preds = np.load(os.path.join(model_save_path, "Test_Preds_{}.npy".format(Test_ID)))
        if Test_IDs.index(Test_ID)==0:
            model_preds = temp_preds
        else:
            model_preds = np.vstack((model_preds, temp_preds))
        
        del(temp_preds)
        
    model_preds = pd.DataFrame(model_preds)
    model_preds.columns = ["BORN_s_" + str(x+1) for x in range(model_preds.shape[-1])]
    pairs_preds = pd.concat([pairs_preds, model_preds], axis=1)    
    return pairs_preds


def GetComMatrix(values):
    # 计算对象数量n
    n = int((1 + np.sqrt(1 + 8 * len(values))) / 2)
    # 初始化矩阵
    compare_matrix = np.zeros((n, n))
    
    # 还原矩阵
    index = 0
    for i in range(n):
        for j in range(i+1, n):
            compare_matrix[i, j] = values[index]
            compare_matrix[j, i] = 1 - compare_matrix[i, j]
            index += 1

    # 处理对角线元素为0.5
    np.fill_diagonal(compare_matrix, 0.5)
    return compare_matrix




def PairsCompMat(group_id, RankNN, pairs_preds):
    tar_data = pairs_preds[pairs_preds['GroupID'] == group_id]
    compare_matrix = GetComMatrix(list(tar_data[RankNN]))
    report_id = []
    for x in list(tar_data['OID_1']) + list(tar_data['OID_2']):
        if x not in report_id:
            report_id.append(x)
            
    return report_id, compare_matrix



#### Pairwise Comparison to Listwise Rank

def BordaCount(compare_matrix):
    borda_score = []
    for row in compare_matrix:
        count = sum(1 for elem in row if elem > 0.5)
        borda_score.append(count/compare_matrix.shape[1])
        borda_rank = ScoreToRank(borda_score, direct="pos")
    return borda_score, borda_rank



## Evaluation Metrics

def MRR(true_rank, pred_rank):
    true_rank = [len(true_rank)-x+1 for x in true_rank]
    pred_rank = [len(pred_rank)-x+1 for x in pred_rank]
    mrr_score = 1/pred_rank[true_rank.index(1)]
    return mrr_score


def GSMRR(true_rank, pred_rank):
    true_rank = [len(true_rank)-x+1 for x in true_rank]
    pred_rank = [len(pred_rank)-x+1 for x in pred_rank]
    gs_mrr_score = np.log(len(true_rank))/pred_rank[true_rank.index(1)]
    return gs_mrr_score


def PairComMatToList(compare_matrix):
    n_items = compare_matrix.shape[0]
    comp_list = []
    for i in range(n_items-1):
        for j in range(i+1, n_items):
            if compare_matrix[i, j]>0.5:
                comp_list.append([i,j])
            else:
                comp_list.append([j,i])
    
    return comp_list, n_items



def PairRankToListRank(method, compare_matrix, alpha=0.01):
    comp_list, n_items = PairComMatToList(compare_matrix)
    if method == "broda":
        temp_score, temp_rank = BordaCount(compare_matrix)
    elif method == "lsr":
        temp_score = choix.lsr_pairwise_dense(compare_matrix, alpha=alpha)
        temp_rank = ScoreToRank(temp_score, direct="pos")
    elif method == "ep":
        temp_score = choix.ep_pairwise(n_items, comp_list, alpha=alpha)[0]
        temp_rank = ScoreToRank(temp_score, direct="pos")

    return temp_score, temp_rank
    


def PairRankToListRankBORN(method, n_items, compare_matrix, comp_list, alpha=0.01):
    if method == "broda":
        temp_score, temp_rank = BordaCount(compare_matrix)
    elif method == "lsr":
        temp_score = choix.lsr_pairwise(n_items, comp_list, alpha=alpha)
        temp_rank = ScoreToRank(temp_score, direct="pos")
    elif method == "ep":
        temp_score = choix.ep_pairwise(n_items, comp_list, alpha=alpha)[0]
        temp_rank = ScoreToRank(temp_score, direct="pos")
        
    return temp_score, temp_rank



def GetOpinionRank(i, group_id_set, data, ListModels, pairs_preds, RankNNs, RankAggre, alpha=0.01):
    
    ### ground truth and analyst reliability
    tar_data = data[data['GroupID']==group_id_set[i]]
    tar_report_id = list(tar_data['ReportID'])
    
    group = {"group_id": group_id_set[i],
             "report_id": tar_report_id,
             "afe": list(tar_data['AFErr']),
             "ape": list(tar_data['APE']),
             "true_rank": ScoreToRank(list(tar_data['AFErr']), direct="neg"),
             }
    
    ### listwise rank prediction
    for model in ListModels:
        temp_score = {model+"_score":list(tar_data[model])}
        temp_rank = {model+"_rank":ScoreToRank(list(tar_data[model]), direct="pos")}
        group = dict(group, **temp_score)
        group = dict(group, **temp_rank)
        
    del(temp_score, temp_rank)    
        
    ### pairwise quality comparison
    for RankNN in RankNNs:
        report_id, compare_matrix = PairsCompMat(group_id_set[i], RankNN, pairs_preds)
        # group = dict(group, **{RankNN+"_report_id":report_id})
        for method in RankAggre:
            temp_score, temp_rank = PairRankToListRank(method, compare_matrix, alpha=alpha)    
            temp_score = [temp_score[k] for k in [report_id.index(i) for i in tar_report_id]]
            temp_rank = [temp_rank[k] for k in [report_id.index(i) for i in tar_report_id]]
            group = dict(group, **{RankNN+"_score_"+method:temp_score})
            group = dict(group, **{RankNN+"_rank_"+method:temp_rank})
            del(temp_score, temp_rank)
            
    ### BayesianORN
    born_report_id, _ = PairsCompMat(group_id_set[i], "BORN_s_1", pairs_preds)
    # group = dict(group, **{"BORN_report_id":born_report_id})
    born_cols = [col for col in pairs_preds.columns if "BORN" in col]    

    for method in RankAggre:
        for BORN in born_cols:
            _, temp_compare_matrix = PairsCompMat(group_id_set[i], BORN, pairs_preds)
            comp_list_temp, n_items = PairComMatToList(compare_matrix)
            temp_compare_matrix = np.where(temp_compare_matrix > 0.5, 1, 0).astype(np.float64)
            if BORN == born_cols[0]:
                compare_matrix = temp_compare_matrix   ###############
                comp_list = comp_list_temp
            else:
                compare_matrix += temp_compare_matrix
                comp_list += comp_list_temp

            del(temp_compare_matrix, comp_list_temp)

        compare_matrix = compare_matrix/len(born_cols)
        np.fill_diagonal(compare_matrix, 0.5)   
        model_score, model_rank = PairRankToListRankBORN(method, n_items, compare_matrix, comp_list, alpha) 
        model_score = [model_score[k] for k in [born_report_id.index(i) for i in tar_report_id]]
        model_rank = [model_rank[k] for k in [born_report_id.index(i) for i in tar_report_id]]
        group = dict(group, **{"BORN_score_" + method: model_score})
        group = dict(group, **{"BORN_rank_" + method: model_rank})   
                        
    return group



def GetRankPerformance(i, group_id_set, data, ListModels, pairs_preds, RankNNs, RankAggre, alpha=0.01):
    group = GetOpinionRank(i, group_id_set, data, ListModels, pairs_preds, RankNNs, RankAggre, alpha)

    results = {'group_id':group['group_id']}
    for model in ListModels + RankNNs + ["BORN"]:
        if model in ListModels:
            temp_ndcg = ndcg_score([group['true_rank']], [group[model+'_rank']])
            temp_ndcg_k = ndcg_score([group['true_rank']], [group[model+'_rank']], k=5)
            temp_mrr = MRR(group['true_rank'], group[model+'_rank'])
            temp_gsmrr = GSMRR(group['true_rank'], group[model+'_rank'])
            temp_kendall = kendalltau(group['true_rank'], group[model+'_rank'])[0]
            
            results = dict(results, **{model+"_ndcg":temp_ndcg})
            results = dict(results, **{model+"_ndcg_k":temp_ndcg_k})
            results = dict(results, **{model+"_mrr":temp_mrr})
            results = dict(results, **{model+"_gsmrr":temp_gsmrr})
            results = dict(results, **{model+"_kendall":temp_kendall})
        else:        
            for method in RankAggre:
                temp_ndcg = ndcg_score([group['true_rank']], [group[model+'_rank_'+method]])
                temp_ndcg_k = ndcg_score([group['true_rank']], [group[model+'_rank_'+method]], k=5)
                temp_mrr = MRR(group['true_rank'], group[model+'_rank_'+method])
                temp_gsmrr = GSMRR(group['true_rank'], group[model+'_rank_'+method])
                temp_kendall = kendalltau(group['true_rank'], group[model+'_rank_'+method])[0]
                
                results = dict(results, **{model+"_"+method+"_ndcg":temp_ndcg})
                results = dict(results, **{model+"_"+method+"_ndcg_k":temp_ndcg_k})
                results = dict(results, **{model+"_"+method+"_mrr":temp_mrr})
                results = dict(results, **{model+"_"+method+"_gsmrr":temp_gsmrr})
                results = dict(results, **{model+"_"+method+"_kendall":temp_kendall})

        
    results = pd.DataFrame(results, index=[0])
    return group, results



def work(i, group_id_set, data, ListModels, pairs_preds, RankNNs, RankAggre, alpha=0.01):
    temp_group, temp_per = GetRankPerformance(i, group_id_set, data, ListModels, pairs_preds, RankNNs, RankAggre, alpha)
    return temp_group, temp_per



def parallel_processing(group_id_set, data, ListModels, pairs_preds, RankNNs, RankAggre, alpha=0.01, cores=20):
    results = Parallel(n_jobs=cores, verbose=10, backend='loky')(
        delayed(GetRankPerformance)(i, group_id_set, data, ListModels, pairs_preds, RankNNs, RankAggre, alpha) for i in range(len(group_id_set))
    )
            
    group_rank = []
    rank_per = []
    
    for result in results:
        temp_group, temp_per = result
        group_rank.append(temp_group)
        rank_per.append(temp_per) 

    per = pd.concat(rank_per, axis=0)    
    return  group_rank, per



def ListwisePerResult(per, RankMetrics, ListModels, RankNNs, RankAggre):
    average_per = {'Models':ListModels + [x+'_'+y for x in RankNNs + ['BORN'] for y in RankAggre]}
    for metric in RankMetrics:
        metric_per = []
        metric_per_sig = []
        for model in ListModels + RankNNs + ['BORN']:
            if model in ListModels:
                metric_per.append(per[model + '_' + metric].mean())
                statistic, p_value = stats.ttest_rel(per['BORN_ep_' + metric].tolist(), per[model + '_' + metric].tolist())
                metric_per_sig.append(p_value)
            else:
                for method in RankAggre:
                    metric_per.append(per[model + '_' + method + '_' + metric].mean())
                    if model!='BORN' and method!='ep':
                        statistic, p_value = stats.ttest_rel(per['BORN_ep_' + metric].tolist(), per[model + '_' + method + '_' + metric].tolist())
                    else:
                        p_value = 0
                    
                    metric_per_sig.append(p_value)
            
        
        per_improve = [(metric_per[-1] - x) / abs(x) * 100 for x in metric_per]
        metric_per = [round(x,4) for x in metric_per]
        per_improve = [round(x,4) for x in per_improve]
        metric_per_sig = [round(x,4) for x in metric_per_sig]
        average_per = dict(average_per, **{metric:metric_per})
        average_per = dict(average_per, **{metric+'_improve':per_improve})
        average_per = dict(average_per, **{metric+'_imp_sig':metric_per_sig})
        del(metric_per, metric_per_sig, statistic, p_value, per_improve)
        
    average_per = pd.DataFrame(average_per)
    return average_per




##################################################################################################################
def LoadData(EXP_ID, SelFeas):
    data_path = "{}{}{}".format("./data/Experiment-",EXP_ID,"/ReportData.csv")
    data = dt.fread(data_path).to_pandas()
    data = data[data['Mask']=='Test']
    data.reset_index(drop=True, inplace=True)
    data = data[SelFeas]
    
    # 找出每个类别A对应的B列最小值
    max_by_group = data.groupby('GroupID')['APE'].max() 
    # 找出每个类别A对应的B列最小值
    min_by_group = data.groupby('GroupID')['APE'].min() 
    # 找出每个类别A对应的B列均值  
    mean_by_group = data.groupby('GroupID')['APE'].mean()
    
    data['Max_APE'] = data['GroupID'].map(max_by_group)
    data['Min_APE'] = data['GroupID'].map(min_by_group)
    data['Mean_APE'] = data['GroupID'].map(mean_by_group)
    
    # data = data[data['Max_APE']>data['Min_APE']]
    # 打开pkl文件
    with open("{}{}{}".format("./output/table/Experiment-",EXP_ID,"/AddedRankPer/GroupRank.pkl"), 'rb') as f:  
        rank = pickle.load(f)
    
    return data, rank


def RankFilter(rank, min_size):
    select_rank = []
    for i in range(len(rank)):
        if len(rank[i]["report_id"]) >= min_size:
            select_rank.append(rank[i])
    return select_rank
    


# 聚合策略一:简单平均
def Concensus(feps):
    return sum(feps) / len(feps)
    


# 聚合策略二:选择排名最高的 
def TopOne(feps, opinion_rank):
    aggre_eps = feps[opinion_rank.index(max(opinion_rank))]
    return aggre_eps
    


def TopKAvg(feps, opinion_rank, k=5):
    sorted_rank = sorted(opinion_rank)
    topk_rank = sorted_rank[-1*k:]
    topk_eps = [feps[opinion_rank.index(r)] for r in topk_rank]
    return np.mean(topk_eps)




# 聚合策略四:选择排名最靠前20%取平均  
def TOPPercAvg(feps, opinion_rank, percent=0.2):
    num = int(len(opinion_rank) * percent)
    top_rank = sorted(opinion_rank)[-1*num:]   
    top_eps = [feps[opinion_rank.index(r)] for r in top_rank]
    return np.mean(top_eps)




# 聚合策略五:使用softmax将分数转为权重,计算加权平均   
def ScoreSoftmax(feps, opinion_score, direct="pos"):
    if direct=="pos":
        weights = softmax(opinion_score)
    else:
        inverse_score = [-1*x for x in opinion_score]
        weights = softmax(inverse_score)
    return sum(feps * weights) / sum(weights)




# 聚合策略六:先选择排名靠前的k个,再使用softmax将分数转为权重,计算加权平均
def TopKSoftmax(feps, opinion_rank, opinion_score, direct="pos", k=5):
    topk_rank = sorted(opinion_rank)[-1*k:]
    topk_eps = [feps[opinion_rank.index(r)] for r in topk_rank]
    topk_score = [opinion_score[opinion_rank.index(r)] for r in topk_rank]    
    if direct=="pos":
        weights = softmax(topk_score)
    else:
        inverse_score = [-1*x for x in topk_score]    
        weights = softmax(inverse_score)
    return sum(topk_eps * weights) / sum(weights)                                



def EPSAggre(feps, opinion_score, opinion_rank, strategy, direct="pos", k=5, percent=0.5):
    if strategy == "TopOne":
        aggre_eps = TopOne(feps, opinion_rank)
    elif strategy == "TopKAve":
        aggre_eps = TopKAvg(feps, opinion_rank, k)
    elif strategy == "TopPercAve":
        aggre_eps = TOPPercAvg(feps, opinion_rank, percent)
    elif strategy == "ScoreSoftmax":
        aggre_eps = ScoreSoftmax(feps, opinion_score, direct)
    elif strategy == "TopKSoftmax":
        aggre_eps = TopKSoftmax(feps, opinion_rank, opinion_score, direct, k)
    return aggre_eps



def ForecastAcc(aggre_eps, true_eps, max_ape, min_ape, mean_ape, metric):
    abs_err = abs(aggre_eps - true_eps)
    if metric == "APE":
        acc = abs_err/abs(true_eps)
        acc = acc.item()
    elif metric == "FACC":
        acc = (max_ape - abs_err/abs(true_eps))/max((max_ape - min_ape), 0.001)
        acc = min(acc.item(),100)
    elif metric == "PMAFE":
        acc = (mean_ape - abs_err/abs(true_eps))/mean_ape
        acc = acc.item()
    return acc



def SingleAggregation(i, rank, data, ListModels, RankNNs, RankAggre, aggre_strategy, eps_metrics):
    
    group_rank = rank[i]
    report_list =  group_rank['report_id']
    group_report = data.query('ReportID in @report_list')
    group_report = group_report.set_index('ReportID')
    group_report = group_report.loc[report_list]
    
    feps = list(group_report['FEPS'])
    true_eps = group_report['AEPS'].unique()
    ape = [abs(x - true_eps)/abs(true_eps) for x in feps]
    max_ape = max(ape)
    min_ape = min(ape)
    mean_ape = np.mean(ape)
        
    
    models = ["Concensus"] + ListModels
    for model in RankNNs + ["BORN"]:
        for aggre in RankAggre:
            models.append(model+"-"+aggre)
    
    all_per = {"Models": models}
    del(model, models)
    
    for metric in eps_metrics:
        for strategy in aggre_strategy:
            per = []
            concensus=Concensus(feps)
            per.append(ForecastAcc(concensus, true_eps, max_ape, min_ape, mean_ape, metric))
            for model in ListModels:                
                opinion_rank =  group_rank[model + '_rank']
                opinion_score =  group_rank[model + '_score']
                aggre_eps = EPSAggre(feps, opinion_score, opinion_rank, strategy, direct="pos", k=5, percent=0.5)
                temp_per = ForecastAcc(aggre_eps, true_eps, max_ape, min_ape, mean_ape, metric)
                per.append(temp_per)
                del(temp_per)

            
            for model in RankNNs + ["BORN"]:
                for aggre in RankAggre:
                    opinion_rank =  group_rank[model + '_rank_' + aggre]
                    opinion_score =  group_rank[model + '_score_' + aggre]
                    aggre_eps = EPSAggre(feps, opinion_score, opinion_rank, strategy, direct="pos", k=5, percent=0.5)
                    temp_per = ForecastAcc(aggre_eps, true_eps, max_ape, min_ape, mean_ape, metric)
                    per.append(temp_per)
                    del(temp_per)
            
            per = {metric + "-" + strategy: per}        
            all_per = dict(all_per, **per)
            del(per)
    all_per = pd.DataFrame(all_per)
    all_per.set_index('Models', inplace=True)
    return all_per 



def parallel_processing_eps(rank, data, ListModels, RankNNs, RankAggre, aggre_strategy, eps_metrics, cores=20):
    results = Parallel(n_jobs=cores, verbose=10, backend='loky', timeout=1000000)(
        delayed(SingleAggregation)(i, rank, data, ListModels, RankNNs, RankAggre, aggre_strategy, eps_metrics) for i in range(len(rank))
    )
            
    return  results 



def GroupSizeFilter(results, rank, min_size):
    tar_results = []
    for i in range(len(rank)):
        if len(rank[i]["report_id"]) >= min_size and np.std(rank[i]["afe"]) > 0 and np.mean(rank[i]["ape"]) != 0:
            tar_results.append(results[i])
    
    ave_per = sum(tar_results)/len(tar_results)
    return ave_per



def GetPerImp(ave_per):
    origin_cols = ave_per.columns
    for col in origin_cols:
        #获取该列所有元素列表
        col_values = ave_per[col].tolist() 
        #获取最后一个元素
        last = col_values[-1]
        #计算其他元素与最后一个元素的增长率
        rates = [(last-x)/x for x in col_values[:-1]]
        rates.append(0)
        #添加新列
        ave_per[col+'_Imp'] = rates
        
    cols = []
    for col in origin_cols:
        cols.append(col)
        if col+'_Imp' in ave_per.columns:
            cols.append(col+'_Imp')
        
    ave_per = ave_per[cols].round(4) * 100    
    
    return ave_per