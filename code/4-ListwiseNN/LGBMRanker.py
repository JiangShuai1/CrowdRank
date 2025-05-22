# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:42:43 2024

@author: jiangshuai
"""


import os
import time
import sys
import platform
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import LTRBenchFunc as LTRF

EXPIDs = [1, 2, 3]

boosting_types = ['gbdt', 'dart']
 
param_grid = {
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [100, 200, 300]
}

start_time = time.time()
for EXPID in EXPIDs:
    print(f"Processing Experiment ID: {EXPID}")
    exp_start_time = time.time()
    
    X_train, y_train, qid_train, group_train, X_val, y_val, qid_val, group_val, \
        X_test, y_test, qid_test, group_test = LTRF.load_data(EXPID)
        
    max_gain = max(group_train.max(), group_val.max(), group_test.max())
    
    for boosting_type in boosting_types:
        print(f"Starting grid search for boosting type: {boosting_type}")
        bt_start_time = time.time()
            
        model_name, best_model, best_params, best_ndcg = \
             LTRF.grid_search_lgbm(X_train, y_train, group_train, X_val, y_val, qid_val, boosting_type, \
                                   param_grid, max_gain)
        
        bt_end_time = time.time()
        bt_run_time = (bt_end_time - bt_start_time) / 60.0
        
        model_dir = f"./output/table/Experiment-{EXPID}/Listwise/{model_name}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # 保存最佳参数
        best_params = pd.DataFrame([best_params], index=[0])
        best_params.to_csv(f"{model_dir}/best_params.csv", index=False)
        print(f"Best parameters saved to: {model_dir}/best_params.csv")
        
        # 保存模型
        best_model.booster_.save_model(f"{model_dir}/best_model.txt")
        print(f"Best model saved to: {model_dir}/best_model.txt")
    
        test_preds = []
        for group in np.unique(qid_test):
            preds = best_model.predict(X_test[qid_test == group])
            test_preds.extend(preds)
    
        # 保存预测结果
        predictions_df = pd.DataFrame({'qid': qid_test, 'prediction': test_preds})
        predictions_df.to_csv(f"./output/table/Experiment-{EXPID}/Listwise/{model_name}/predictions.csv", index=False)
        print(f"Test set predictions saved to: {model_dir}/predictions.csv")
        print(f"Grid search for boosting type {boosting_type} completed. Run time: {bt_run_time:.2f} mins")
    
    exp_end_time = time.time()
    exp_run_time = (exp_end_time - exp_start_time) / 60.0
    print(f"Experiments for {EXPID} completed. Run time: {exp_run_time:.2f} mins")
    
end_time = time.time()
run_time = (end_time - start_time) / 60.0
print(f"Total execution time: {run_time:.2f} mins")