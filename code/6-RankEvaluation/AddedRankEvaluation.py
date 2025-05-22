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
import AddedRankEvaluationFunc as EF

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

### Settting
EXP_IDs = [1, 2, 3]
ListwiseTree = ['LambdaMART', 'DART']
ListwiseDeep = ['ListNet', 'ListGATNet']
ListModels = ListwiseTree + ListwiseDeep
RankNNs = ['RankAutoIntNet', 'NBORN']
RankMetrics = ['ndcg', 'ndcg_k', 'mrr', 'gsmrr', 'kendall']
SelFeas = ['GroupID', 'ReportID', 'APE']
RankAggre = ['broda', 'ep']
Test_IDs = [i+1 for i in range(5)]
alphas = [0.01]


for EXP_ID in EXP_IDs: 
    print("{}{}{}".format("Experiment-", EXP_ID, " is beginning now!"))

    ## Create output directory
    per_path = "./output/table/Experiment-{}/AddedRankPer/".format(EXP_ID)
    if os.path.exists(per_path):
        shutil.rmtree(per_path)
    # os.rmdir(per_path) if os.path.exists(per_path) else None
    os.makedirs(per_path, exist_ok=True)
    ## Loading Key Data
    data = EF.GetPreds(EXP_ID, ListwiseTree, ListwiseDeep)
    print("Start getting pairs' predictions...")
    start_time = time.time()
    pairs_preds, group_id_set = EF.PairPreds(EXP_ID, Test_IDs, SelFeas, RankNNs)
    print("running time is：", time.time() - start_time)
    pairs_preds = EF.ReadBORNPred(EXP_ID, Test_IDs, pairs_preds)
    
    
    for alpha in alphas:
        print("alpha is:", alpha)
        # 调用函数进行并行处理
        start_time = time.time()
        print("Start parallel processing...")
        group_rank, per = EF.parallel_processing(group_id_set, data, ListModels, pairs_preds, RankNNs, RankAggre, alpha, cores=50)
        print("running time is：", time.time() - start_time)
        with open(f"{per_path}/GroupRank.pkl", 'wb') as file:
            pickle.dump(group_rank, file)
            
        list_per = EF.ListwisePerResult(per, RankMetrics, ListModels, RankNNs, RankAggre)    
        
        list_per.to_csv(f"{per_path}/ListRankPerforamnce.csv", mode='w')
        del(group_rank, list_per, per)

     
    del(data, pairs_preds, group_id_set)
    print("{}{}{}".format("Experiment-", EXP_ID, " has done!"))    # 1sec