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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

EXP_IDs = [1, 2, 3]
SelFeas = ['ReportID', 'GroupID', 'ReportDate', 'FEndDate', 'FEPS', 'PrevEPS', 'AEPS', 'FErr', 'AFErr', 'APE', 'GroupSize']
AnaRel = ['rs1','rs2','rs3']
PointNNs = ["EWDNN", "ResNet", "WideAndDeep", "FM", "DeepFM", "DCN", "AutoIntNet"]
RankNNs = ['RankEWDNN', 'RankResNet', 'RankWAD', 'RankFM', 'RankDeepFM', 'RankDCN', 'RankAutoIntNet', 'NBORN']
RankAggre = ['broda', 'ahp', 'serial', 'lsr', 'ilsr', 'spring', 'rcen', 'ep']
aggre_strategy = ['TopOne', 'TopKAve', 'TopPercAve', 'TopKSoftmax', 'ScoreSoftmax']
eps_metrics = ['APE', 'FACC', 'PMAFE']
min_sizes = [5,10,15,20,25,30,35,40,45,50]


for EXP_ID in EXP_IDs: 
    print("{}{}{}".format("Experiment-", EXP_ID, " is beginning now!"))
    data, rank = EPSPF.LoadData(EXP_ID, SelFeas)
    per_path = "./output/table/Experiment-{}/ACWPer/".format(EXP_ID)
    if os.path.exists(per_path):
        shutil.rmtree(per_path)
        # os.rmdir(per_path) if os.path.exists(per_path) else None
    os.makedirs(per_path, exist_ok=True)
    print("perforamnce folder {} is created!".format(EXP_ID))    
    print("Start calculating performance of ACW...")
    # temp = EPSPF.SingleAggregation(0, rank, data, AnaRel, PointNNs, RankNNs, RankAggre, aggre_strategy, eps_metrics)
    print("Start getting average performance of ACW...")
    start_time = time.time()
    results = EPSPF.parallel_processing(rank, data, AnaRel, PointNNs, RankNNs, RankAggre, aggre_strategy, eps_metrics, cores=50)
    print("running time of ACW performance is：", time.time() - start_time)

    for min_size in min_sizes:
        print("Start filtering data with min_size =", min_size)
        ave_per = EPSPF.GroupSizeFilter(results, rank, min_size)
        ave_per = EPSPF.GetPerImp(ave_per)        
        ave_per.to_csv("{}{}{}{}{}".format("./output/table/Experiment-", EXP_ID, "/ACWPer/Perforamnce-",min_size, ".csv"), mode='w')
        print("The average performance of ACW with min_size =", min_size, "is calculated!")
    del(data, rank, results, ave_per)
    print("{}{}{}".format("Experiment-", EXP_ID, " has done!"))    # 1sec