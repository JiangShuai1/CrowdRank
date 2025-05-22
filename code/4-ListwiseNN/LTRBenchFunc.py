# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:45:35 2024

@author: jiangshuai
"""

import subprocess
import os
import time
import sys
import platform
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from scipy.stats import rankdata

import itertools
from itertools import product
import shutil
from sklearn.datasets import load_svmlight_file
from lightgbm import LGBMRanker
 
if platform.system().lower() == 'windows':
    print("windows")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.chdir('E:/project/12-OpinionRank')
    sys.path.append(r"E:/project/12-OpinionRank/code/Py")
elif platform.system().lower() == 'linux':
    print("linux")
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"   # set GPU device number 0, 1, 2, etc.
    os.chdir('/home/data/jiangshuai/project/1-OpinionRank')
    sys.path.append(r"/home/data/jiangshuai/project/1-OpinionRank/code/Py")


# 训练模型
def train_model(RANKLIB_JAR, train_file, model_file, ranker, params, metric='NDCG@5'):
    
    if ranker != 7:
        command = [
            'java', '-jar', RANKLIB_JAR,
            '-train', train_file,
            '-ranker', str(ranker),
            '-metric2t', metric,
            '-save', model_file
        ]
    else:
        command = [
            'java', '-jar', RANKLIB_JAR,
            '-train', train_file,
            '-ranker', str(ranker),
            '-save', model_file
        ]
        
    command.extend(params)
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        
        

# 生成预测结果
def generate_predictions(RANKLIB_JAR, model_file, test_file, ranker, params, output_file='Predictions.txt'):
    command = [
        'java', '-jar', RANKLIB_JAR,
        '-load', model_file,
        '-rank', test_file,
        '-ranker', str(ranker),
        '-score', output_file
    ]
    command.extend(params)
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")




# 读取预测结果
def read_predictions(pred_file):
    predictions = []
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid = int(parts[0])
            doc_id = int(parts[1])
            score = float(parts[2])
            predictions.append((qid, doc_id, score))
    return pd.DataFrame(predictions, columns=['qid', 'doc_id', 'score'])



# 读取测试集的真实标签
def read_test_labels(test_file):
    labels = []
    with open(test_file, 'r') as f:
        lines = f.readlines()
        current_qid = None
        current_doc_id = 0
        for line in lines:
            parts = line.strip().split()
            qid = int(parts[1].split(':')[1])  # 假设qid格式为 qid:123
            label = int(parts[0])
            if qid != current_qid:
                current_qid = qid
                current_doc_id = 0
            labels.append((qid, current_doc_id, label))
            current_doc_id += 1
    return pd.DataFrame(labels, columns=['qid', 'doc_id', 'label'])




def evaluate_performance(true_labels, predictions):
    # 确保两个数据框都按qid和doc_id排序
    true_labels = true_labels.sort_values(by=['qid', 'doc_id']).reset_index(drop=True)
    predictions = predictions.sort_values(by=['qid', 'doc_id']).reset_index(drop=True)
    
    # 获取唯一的qid列表
    qids = true_labels['qid'].unique()
    
    # 初始化NDCG值列表
    ndcg_scores = []
    
    for qid in qids:
        # 获取当前qid的真实标签和预测分数
        true_subset = true_labels[true_labels['qid'] == qid]['label']
        pred_subset = predictions[predictions['qid'] == qid]['rank']
        
        # 计算NDCG
        ndcg = ndcg_score([true_subset], [pred_subset])
        ndcg_scores.append(ndcg)
    
    # 计算所有qid的平均NDCG
    mean_ndcg = np.mean(ndcg_scores)
    
    return mean_ndcg
    



def grid_search_hyperparameter_tuning(EXPID, RANKLIB_JAR, MODELS):
    
    ### data path
    train_file =  f"./output/table/Experiment-{EXPID}/L2RData/train.txt"
    val_file =  f"./output/table/Experiment-{EXPID}/L2RData/val.txt"
    test_file =  f"./output/table/Experiment-{EXPID}/L2RData/test.txt"


    # 遍历每个模型
    for model_name, model_info in MODELS.items():
        #path to save
        model_dir = f"./output/table/Experiment-{EXPID}/Listwise/{model_name}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # 初始化最佳模型和最佳NDCG
        best_model = f"{model_dir}/best_model.txt"
        best_params = {}
        best_ndcg = -1.0
        
        
        # 模型选择与产参数
        ranker = model_info['ranker']
        params_grid = model_info['params_grid']
        

        # 生成所有可能的参数组合
        param_combinations = list(itertools.product(*params_grid.values()))
        
        # 记录开始时间
        start_time = time.time()        

        # 遍历每个参数组合
        for params in param_combinations:
            # 构建当前参数列表
            current_params = []
            for i, (param_key, _) in enumerate(params_grid.items()):
                current_params.append(param_key)
                current_params.append(str(params[i]))
                
            # 记录单个模型训练的开始时间
            single_start_time = time.time()

            # 训练模型
            model_file = f"{model_dir}/{model_name}_model.txt"
            print(f"Training {model_name} with params: {current_params}")
            train_model(RANKLIB_JAR, train_file, model_file, ranker, current_params)

            # 生成验证集的预测结果
            val_pred_file = f"{model_dir}/{model_name}_predictions_val.txt"
            generate_predictions(RANKLIB_JAR, model_file, val_file, ranker, current_params, val_pred_file)

            # 读取预测结果和真实标签
            val_predictions = read_predictions(val_pred_file)
            val_predictions['rank'] = val_predictions.groupby('qid')['score'].transform(lambda x: rankdata(x, method='max').astype(int))
            true_labels = read_test_labels(val_file)

            # 评估性能
            ndcg = evaluate_performance(true_labels, val_predictions)

            # 更新最佳模型
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                shutil.move(model_file, best_model)
                # 保存最佳参数到数据框
                best_params = {k: v for k, v in zip(params_grid.keys(), params)}
                best_params_df = pd.DataFrame([best_params])
                best_params_df.to_csv(f"{model_dir}/best_params.csv", index=False)

            # 记录单个模型训练的结束时间并打印
            single_end_time = time.time()
            single_training_time = (single_end_time - single_start_time)/60
            print(f"Model: {model_name}, Params: {current_params}, NDCG: {ndcg:.4f}, Training Time: {single_training_time:.2f} mins")
            
            
        # 最佳模型在测试集上的预测
        best_pred_file = f"{model_dir}/best_predictions_test.txt"
        best_params_list = [item for k, v in best_params.items() for item in (k, str(v))]
        generate_predictions(RANKLIB_JAR, best_model, test_file, ranker, best_params_list, best_pred_file)
        predictions = read_predictions(best_pred_file)
        predictions = predictions[['qid', 'score']].rename(columns={'score': 'prediction'})  
        predictions['rank'] = predictions.groupby('qid')['prediction'].transform(lambda x: rankdata(x, method='max').astype(int))
        predictions.to_csv(f"{model_dir}/Predictions.csv")
        # 记录结束时间
        end_time = time.time()
        print(f"Grid search for {model_name} completed in {end_time - start_time:.2f} seconds.")

    return predictions






######################################################################################################################

def ScoreToRank(scores, direct="neg"):
    # 确保 scores 是一维数组
    scores = np.array(scores).ravel()
    
    sorted_index = sorted(range(len(scores)), key=lambda x: scores[x])
    if direct != "neg":
        rank = [sorted_index.index(i) + 1 for i in range(len(scores))]   # score越大，rank分数越大
    else:
        rank = [len(scores) - sorted_index.index(i) for i in range(len(scores))]
    return rank


def evaluate_ndcg(y_true, y_pred, qid):
    unique_qids = np.unique(qid)
    ndcg_scores = []
    
    for q in unique_qids:
        mask = qid == q
        # 确保 y_true 和 y_pred 是一维数组
        y_true_q = y_true[mask].ravel() + 1
        y_pred_q = y_pred[mask].ravel()
        
        # 将预测分数转换为排名
        y_pred_rank = ScoreToRank(y_pred_q, 'pos')
        
        # 计算 NDCG 分数
        ndcg = ndcg_score([y_true_q], [y_pred_rank])
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)


def load_data(EXPID):
    X_train, y_train, qid_train = load_svmlight_file(f"./output/table/Experiment-{EXPID}/L2RData/train.txt", query_id=True)
    X_val, y_val, qid_val = load_svmlight_file(f"./output/table/Experiment-{EXPID}/L2RData/val.txt", query_id=True)
    X_test, y_test, qid_test = load_svmlight_file(f"./output/table/Experiment-{EXPID}/L2RData/test.txt", query_id=True)

    _, group_train = np.unique(qid_train, return_counts=True)
    _, group_val = np.unique(qid_val, return_counts=True)
    _, group_test = np.unique(qid_test, return_counts=True)

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    return X_train, y_train, qid_train, group_train, X_val, y_val, qid_val, group_val, X_test, y_test, qid_test, group_test



def grid_search_lgbm(X_train, y_train, group_train, X_val, y_val, qid_val, boosting_type, param_grid, max_gain):
    best_params = None
    best_ndcg = -np.inf
    best_model = None
    model_name = f"{'DART' if boosting_type == 'dart' else 'LambdaMART'}"

    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        
        gbm = LGBMRanker(
            boosting_type = boosting_type,
            objective='lambdarank',
            label_gain=[i + 1 for i in range(max_gain)],
            n_jobs=8,
            num_leaves=param_dict['num_leaves'],
            max_depth=param_dict['max_depth'],
            learning_rate=param_dict['learning_rate'],
            n_estimators=param_dict['n_estimators'],
            deterministic=True,
            force_row_wise=True
        )

        start_time = time.time()
        gbm.fit(X_train, y_train, group=group_train)
        end_time = time.time()
        train_time = (end_time - start_time)/60

        val_preds = []
        for group in np.unique(qid_val):
            preds = gbm.predict(X_val[qid_val == group])
            val_preds.extend(preds)
        
        # 确保 val_preds 是一维数组
        val_preds = np.array(val_preds).ravel()
        ndcg = evaluate_ndcg(y_val, val_preds, qid_val)

        print(f"Model: {model_name}, Params: {param_dict}, NDCG: {ndcg:.4f}, Train Time: {train_time:.2f} mins")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_params = param_dict
            best_model = gbm

    return model_name, best_model, best_params, best_ndcg
