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


def ScoreToRank(scores, direct="neg"):
    sorted_index = sorted(range(len(scores)), key=lambda x: scores[x])
    if direct != "neg":
        rank = [sorted_index.index(i) + 1 for i in range(len(scores))]   ## score越大， rank分数越大
    else:
        rank = [len(scores) - sorted_index.index(i) for i in range(len(scores))]
    return rank


def GetPreds(EXP_ID, NNs):
    data_path = "{}{}{}".format("./output/table/Experiment-",EXP_ID,"/ReportData.csv")
    data = dt.fread(data_path).to_pandas()
    data = data[data['Mask']=='Test']
    data.reset_index(drop=True, inplace=True)
    
    ml_preds_path = "{}{}{}".format("./output/table/Experiment-",EXP_ID,"/BaselinePrediction.csv")
    ml_preds = dt.fread(ml_preds_path).to_pandas()
    
    for i in range(len(NNs)):
        nn_preds_path = "{}{}{}{}{}".format("./output/table/Experiment-",EXP_ID,"/Pointwise/",NNs[i],"/Predictions.csv")
        if i == 0 :
            nn_preds = dt.fread(nn_preds_path).to_pandas()
        else:
            nn_preds = pd.concat([nn_preds, dt.fread(nn_preds_path).to_pandas()], axis=1)
    
    nn_preds.columns = NNs
    preds = pd.concat([ml_preds, nn_preds], axis=1)      
    data = pd.concat([data, preds], axis=1)
    point_models = list(ml_preds.columns) + NNs
    
    return data, point_models



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
    elif method == "serial":
        temp_score = SerialRank(compare_matrix)
        temp_rank = ScoreToRank(temp_score, direct="pos")
    elif method == "spring":
        temp_score = SpringRank(compare_matrix, alpha=alpha)
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
    elif method == "serial":
        temp_score = SerialRank(compare_matrix)
        temp_rank = ScoreToRank(temp_score, direct="pos")
    elif method == "spring":
        temp_score = SpringRank(compare_matrix, alpha=alpha)
        temp_rank = ScoreToRank(temp_score, direct="pos")
    elif method == "ep":
        temp_score = choix.ep_pairwise(n_items, comp_list, alpha=alpha)[0]
        temp_rank = ScoreToRank(temp_score, direct="pos")
        
    return temp_score, temp_rank




def GetOpinionRank(i, group_id_set, data, point_models, pairs_preds, RankNNs, RankAggre, alpha=0.01):
    
    ### ground truth and analyst reliability
    tar_data = data[data['GroupID']==group_id_set[i]]
    tar_report_id = list(tar_data['ReportID'])
    
    group = {"group_id": group_id_set[i],
             "report_id": tar_report_id,
             "afe": list(tar_data['AFErr']),
             "ape": list(tar_data['APE']),
             "true_rank": ScoreToRank(list(tar_data['AFErr']), direct="neg"),
             }
    
    ### pointwise quality prediction
    for model in point_models:
        temp_score = {model+"_score":list(tar_data[model])}
        temp_rank = {model+"_rank":ScoreToRank(list(tar_data[model]), direct="neg")}
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
        group = dict(group, **{"BORN_score_"+method: model_score})
        group = dict(group, **{"BORN_rank_"+method: model_rank})   
                        
    return group




def GetRankPerformance(i, group_id_set, data, point_models, pairs_preds, RankNNs, AnaRel, RankAggre, alpha=0.01):
    group = GetOpinionRank(i, group_id_set, data, point_models, pairs_preds, RankNNs, RankAggre, alpha)
    results = {'group_id':group['group_id']}
    for model in AnaRel + point_models + RankNNs + ["BORN"]:
        if model in AnaRel + point_models:
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




def work(i, group_id_set, data, point_models, pairs_preds, RankNNs, AnaRel, RankAggre, alpha=0.01):
    temp_group, temp_per = GetRankPerformance(i, group_id_set, data, point_models, pairs_preds, RankNNs, AnaRel, RankAggre, alpha)
    return temp_group, temp_per



def parallel_processing(group_id_set, data, point_models, pairs_preds, RankNNs, AnaRel, RankAggre, alpha=0.01, cores=20):
    results = Parallel(n_jobs=cores, verbose=10, backend='loky')(
        delayed(GetRankPerformance)(i, group_id_set, data, point_models, pairs_preds, RankNNs, AnaRel, RankAggre, alpha) for i in range(len(group_id_set))
    )
            
    group_rank = []
    rank_per = []
    
    for result in results:
        temp_group, temp_per = result
        group_rank.append(temp_group)
        rank_per.append(temp_per) 

    per = pd.concat(rank_per, axis=0)    
    return  group_rank, per




################
def ListwisePerResult(per, RankMetrics, AnaRel, point_models, RankNNs, RankAggre):
    average_per = {'Models':AnaRel + point_models + [x+'_'+y for x in RankNNs + ['BORN'] for y in RankAggre]}
    for metric in RankMetrics:
        metric_per = []
        metric_per_sig = []
        for model in AnaRel + point_models + RankNNs+['BORN']:
            if model in AnaRel + point_models:
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



#####################################################################################################

def SerialRank(A):
    N = A.shape[0]
    Q = np.copy(A)
    M = A + A.T
    Q[A != 0] = A[A != 0] / M[A != 0]
    Q[M == 0] = 1 / 2

    S = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            indicator = M[i, :] * M[j, :]
            S[i, j] = S[i, j] + np.sum(indicator == 0) / 2
            nonzero = np.where(indicator > 0)[0]
            if len(nonzero) > 0:
                S[i, j] = S[i, j] + len(nonzero) - np.sum(np.abs(Q[i, nonzero] - Q[j, nonzero])) / 2
    S = S + S.T
    L = np.diag(np.sum(S, axis=1)) - S
    _, serr = np.linalg.eig(L)
    serr = serr[:, 1]
    return serr




def build_from_dense(A, alpha, l0, l1):
    """
    Given as input a 2d numpy array, build the matrices A and B to feed to the linear system solver for SpringRank.
    """
    n = A.shape[0]
    k_in = np.sum(A, 0)
    k_out = np.sum(A, 1)

    D1 = k_in + k_out           # to be seen as diagonal matrix, stored as 1d array
    D2 = l1 * (k_out - k_in)    # to be seen as diagonal matrix, stored as 1d array

    if alpha != 0.:
        B = np.ones(n) * (alpha * l0) + D2
        A = - (A + A.T)
        A[np.arange(n), np.arange(n)] = alpha + D1 + np.diagonal(A)
    else:
        last_row_plus_col = (A[n - 1, :] + A[:, n - 1]).reshape((1, n))
        A = A + A.T
        A += last_row_plus_col

        A[np.arange(n), np.arange(n)] = A.diagonal() + D1
        D3 = np.ones(n) * (l1 * (k_out[n - 1] - k_in[n - 1]))  # to be seen as diagonal matrix, stored as 1d array
        B = D2 + D3

    return scipy.sparse.csr_matrix(A), B




def solve_linear_system(A, B, solver, verbose):
    if solver not in ['spsolve', 'bicgstab']:
        warnings.warn('Unknown parameter {solver} for argument solver. Setting solver = "bicgstab"'.format(solver=solver))
        solver = 'bicgstab'

    if verbose:
        print('Using scipy.sparse.linalg.{solver}(A,B)'.format(solver=solver))

    if solver == 'spsolve':
        sol = scipy.sparse.linalg.spsolve(A, B)
    elif solver == 'bicgstab':
        sol = scipy.sparse.linalg.bicgstab(A, B)[0]

    return sol.reshape((-1,))





def SpringRank(A, alpha=0., l0=1., l1=1., solver='bicgstab', verbose=False):
    """
        Main routine to calculate SpringRank by a solving linear system.

        Parameters
        ----------
        A : numpy.ndarray or scipy.sparse.spmatrix
            Has tobe  2 dimensional and with same dimensions.
        alpha, l0, l1: float
            Defined as in the SpringRank paper
            https://arxiv.org/abs/1709.09002
        solver: str
            One between 'spsolve' (direct, slower) and 'bicgstab' (iterative, faster).
            The solver to be used for the linear system returning the ranks.
        verbose: bool
        force_dense: bool
            By default A is converted to a sparse matrix scipy.sparse.csr, if it is not already sparse.
            If force_dense is set to True and a dense ndarray A is input, then it is not converted to sparse.

        Returns
        -------
        rank
            numpy.ndarray of ranks. Indices represent the nodes' indices used in the matrix A.

    """

    # check if input is sparse or can be converted to sparse.

    # build array to feed linear system solver
    A, B = build_from_dense(A, alpha, l0, l1)

    rank = solve_linear_system(A, B, solver, verbose)

    return rank    






    
###########################################################################################################
def PairwisePredPointModel(i, pairs_preds, data, point_models, PointNNs):
    data_1 = data[data['ReportID']==pairs_preds.at[i,'OID_1']]
    data_2 = data[data['ReportID']==pairs_preds.at[i,'OID_2']]
    pred = {"Truth": pairs_preds.at[i,'Truth'],
            "rs1": int(data_1['RS1_APE'].iloc[0] < data_2['RS1_APE'].iloc[0]),
            "rs2": int(data_1['RS2_APE'].iloc[0] < data_2['RS2_APE'].iloc[0]),
            "rs3": int(data_1['RS3_APE'].iloc[0] < data_2['RS3_APE'].iloc[0])}
    for model in point_models+PointNNs:
        pred = dict(pred, **{model: int(data_1[model].iloc[0] < data_2[model].iloc[0])})
    
    pred = pd.DataFrame([pred])
    return pred
        



def parallel_point(pairs_preds, data, point_models, PointNNs, cores=20):
    results = Parallel(n_jobs=cores, verbose=10, backend='loky')(
        delayed(PairwisePredPointModel)(i, pairs_preds, data, point_models, PointNNs) for i in range(1000)             # pairs_preds.shape[0]
    )
            
    pair_preds_from_point = []
    for result in results:
        pair_preds_from_point.append(result)
        
    pair_preds_from_point = pd.concat(pair_preds_from_point, axis=0)
    return  pair_preds_from_point
        
        
 
def PairwisePerPointModel(pair_preds_from_point, AnaRel, point_models, PointNNs):
    targets = pair_preds_from_point['Truth'].values > 0.5
    pair_per_from_point = []
    for model in AnaRel + point_models + PointNNs:
        preds = pair_preds_from_point[model].values
        pcls = preds > 0.5
        acc = accuracy_score(targets, pcls)
        precision = precision_score(targets, pcls, zero_division=0)
        recall = recall_score(targets, pcls, zero_division=0)
        f1 = f1_score(targets, pcls, zero_division=0)
        auc = roc_auc_score(targets, preds)
        logloss = -1 * log_loss(targets, preds, eps=1e-5)
        temp_per = pd.DataFrame([{'Models': model,
                                  'Accuracy': acc,
                                  'Precision': precision,
                                  'Recall': recall,
                                  'F1-Score': f1,
                                  'AUC': auc,
                                  'Log-Loss': logloss}])
        pair_per_from_point.append(temp_per)
        
    pair_per_from_point = pd.concat(pair_per_from_point, axis=0)
    return pair_per_from_point
    



def PairwisePerResult(EXP_ID, RankNNs, pair_per_from_point):
    models = pd.DataFrame({'Models':  list(pair_per_from_point['Models']) + RankNNs + 
                           ['BORN_'+str(x) for x in [100-5*y for y in range(20)]]})
    for model in RankNNs:
        tar_per = pd.read_csv("{}{}{}{}{}".format("./output/table/Experiment-",EXP_ID,"/Pairwise/",model,"/Performance.csv"))
        if model == RankNNs[0]:
            pair_per = tar_per
        else:
            pair_per = pd.concat([pair_per, tar_per], axis=0)
            
    born_per = pd.read_csv("{}{}{}".format("./output/table/Experiment-",EXP_ID,"/Pairwise/BayesianORN/Performance.csv"))
    pair_per = pd.concat([pair_per, born_per], axis=0)
    pair_per = pd.concat([pair_per_from_point[pair_per.columns], pair_per], axis=0)
        
    new_df = pair_per.copy()  # 创建一个副本以免修改原始数据
    column_order = []
    for col in new_df.columns:
        new_df[col] = new_df[col].round(4)  # 将数据类型转为float
        last_value = new_df[col].iloc[-1]  # 获取每列的最后一个元素
        percentage_growth = (last_value/new_df[col]  - 1) * 100  # 计算百分比增长率
        col_name = col + '_Improve'
        new_df[col_name] = percentage_growth.round(4)  # 将结果添加到原数据框中
        column_order.append(col)
        column_order.append(col_name)
    # 调整列顺序
    new_df = new_df[column_order]
    new_df['Log-Loss_Improve'] = -1*new_df['Log-Loss_Improve']
    new_df['Log-Loss'] = -1*new_df['Log-Loss']
    models.reset_index(drop=True, inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    pair_per = pd.concat([models, new_df], axis=1)
    return pair_per    