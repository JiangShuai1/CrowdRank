# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:26:35 2024

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
import EvaluationFunc as EF

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

### Settting
EXP_IDs = [1, 2, 3]
AnaRel = ['rs1','rs2','rs3']
PointNNs = ["EWDNN", "ResNet", "WideAndDeep", "FM", "DeepFM", "DCN", "AutoIntNet"]
RankNNs = ['RankEWDNN', 'RankResNet', 'RankWAD', 'RankFM', 'RankDeepFM', 'RankDCN', 'RankAutoIntNet', 'NBORN']
RankMetrics = ['ndcg', 'ndcg_k', 'mrr', 'gsmrr', 'kendall']
SelFeas = ['GroupID', 'ReportID', 'APE']
RankAggre = ['broda', 'serial', 'lsr', 'spring', 'ep']
Test_IDs = [i+1 for i in range(5)]
alphas = [0.01]


for EXP_ID in EXP_IDs: 
    print("{}{}{}".format("Experiment-", EXP_ID, " is beginning now!"))

    ## Create output directory
    per_path = "./output/table/Experiment-{}/RankPer/".format(EXP_ID)
    if os.path.exists(per_path):
        shutil.rmtree(per_path)
    # os.rmdir(per_path) if os.path.exists(per_path) else None
    os.makedirs(per_path, exist_ok=True)
    ## Loading Key Data
    data, point_models = EF.GetPreds(EXP_ID=EXP_ID, NNs=PointNNs)
    print("Start getting pairs' predictions...")
    start_time = time.time()
    pairs_preds, group_id_set = EF.PairPreds(EXP_ID, Test_IDs, SelFeas, RankNNs)
    print("running time is：", time.time() - start_time)
    pairs_preds = EF.ReadBORNPred(EXP_ID, Test_IDs, pairs_preds)
    
    print("Start getting pairwise ranking performance...")
    start_time = time.time()
    pair_preds_from_point = EF.parallel_point(pairs_preds, data, point_models, PointNNs, cores=50)   
    pair_per_from_point = EF.PairwisePerPointModel(pair_preds_from_point, AnaRel, point_models, PointNNs)
    pair_per = EF.PairwisePerResult(EXP_ID, RankNNs, pair_per_from_point)
    pair_per.to_csv("{}{}{}".format("./output/table/Experiment-", EXP_ID, "/RankPer/PairRankPerforamnce.csv"), mode='w')
    del(pair_preds_from_point, pair_per_from_point, pair_per)
    print("running time of pairwise ranking performance is：", time.time() - start_time)
    
    
    for alpha in alphas:
        print("alpha is:", alpha)
        # 调用函数进行并行处理
        start_time = time.time()
        print("Start parallel processing...")
        group_rank, per = EF.parallel_processing(group_id_set, data, point_models, pairs_preds, RankNNs, AnaRel, RankAggre, alpha, cores=50)
        print("running time is：", time.time() - start_time)
        with open("{}{}{}".format("./output/table/Experiment-",EXP_ID,"/RankPer/GroupRank.pkl"), 'wb') as file:
            pickle.dump(group_rank, file)
            
        list_per = EF.ListwisePerResult(per, RankMetrics, AnaRel, point_models, RankNNs, RankAggre)    
        
        list_per.to_csv("{}{}{}".format("./output/table/Experiment-", EXP_ID, "/RankPer/ListRankPerforamnce.csv"), mode='w')
        del(group_rank, list_per, per)

     
    del(data, point_models, pairs_preds, group_id_set)
    print("{}{}{}".format("Experiment-", EXP_ID, " has done!"))    # 1sec