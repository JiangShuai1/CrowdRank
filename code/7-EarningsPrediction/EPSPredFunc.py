# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:44:18 2024

@author: jiangshuai
"""

import os
import json
import numpy as np
import pandas as pd
from datatable import dt
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score
from multiprocessing import Pool
from functools import partial
import scipy.stats as stats
from collections import namedtuple
import pickle
from scipy.special import softmax
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import glob


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
    with open("{}{}{}".format("./output/table/Experiment-",EXP_ID,"/RankPer/GroupRank.pkl"), 'rb') as f:  
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



def SingleAggregation(i, rank, data, AnaRel, PointNNs, RankNNs, RankAggre, aggre_strategy, eps_metrics):
    
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
        
    
    models = ["Concensus"] + AnaRel + PointNNs
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
            for model in AnaRel:                
                opinion_rank =  group_rank[model + '_rank']
                opinion_score =  group_rank[model]
                aggre_eps = EPSAggre(feps, opinion_score, opinion_rank, strategy, direct="pos", k=5, percent=0.5)
                temp_per = ForecastAcc(aggre_eps, true_eps, max_ape, min_ape, mean_ape, metric)
                per.append(temp_per)
                del(temp_per)
            
            for model in PointNNs:
                opinion_rank =  group_rank[model + '_rank']
                opinion_score =  group_rank[model + '_score']
                aggre_eps = EPSAggre(feps, opinion_score, opinion_rank, strategy, direct="neg", k=5, percent=0.5)
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



def parallel_processing(rank, data, AnaRel, PointNNs, RankNNs, RankAggre, aggre_strategy, eps_metrics, cores=20):
    results = Parallel(n_jobs=cores, verbose=10, backend='loky', timeout=1000000)(
        delayed(SingleAggregation)(i, rank, data, AnaRel, PointNNs, RankNNs, RankAggre, aggre_strategy, eps_metrics) for i in range(len(rank))
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

    
    
    
########### 
def SingleAggregation1(i, rank, data, AnaRel, PointNNs, RankNNs, RankAggre):
    
    group_rank = rank[i]
    report_list =  group_rank['report_id']
    group_report = data.query('ReportID in @report_list')
    group_report = group_report.set_index('ReportID')
    group_report = group_report.loc[report_list]
    
    for model in AnaRel:
        group_report[model + '_rank'] =  group_rank[model + '_rank']
        group_report[model + '_score'] =  group_rank[model]
    
    for model in PointNNs:
        group_report[model + '_rank'] =  group_rank[model + '_rank']
        group_report[model + '_score'] =  group_rank[model + '_score']
        
    
    for model in RankNNs + ["BORN"]:
        for aggre in RankAggre:
            temp = pd.DataFrame({k: group_rank[k] for k in [model + '_rank_' + aggre, model + '_score_' + aggre]})
            temp.index = report_list
            group_report = pd.concat([group_report, temp], axis=1)
    
    return group_report


######################################## Added in Revision 2 ########################################
# 将score转为rank
def ScoreToRank(scores, direct="neg"):
    sorted_index = sorted(range(len(scores)), key=lambda x: scores[x])
    if direct != "neg":
        rank = [sorted_index.index(i) + 1 for i in range(len(scores))]   ## score越大， rank分数越大
    else:
        rank = [len(scores) - sorted_index.index(i) for i in range(len(scores))]  ## score越小， rank分数越大
    return rank


def GetPriceData(price_path, start_date=pd.to_datetime('2010-01-01'), end_date=pd.to_datetime('2022-12-31')):
    """
    自动化读取指定目录及其所有子目录下的CSV文件，并将它们按列合并。
    
    参数:
    - directory: str, 根目录路径
    
    返回:
    - merged_df: DataFrame, 按列合并后的数据框
    """
    # 获取大盘指数的价格数据
    index_df = pd.read_csv("data/股票历史行情信息表-前复权/指数文件/TRD_Index.csv")
    index_df['Trddt'] = pd.to_datetime(index_df['Trddt'])
    index_df = index_df[(index_df['Trddt'] >= start_date) & (index_df['Trddt'] <= end_date)]
    index_df = index_df.sort_values(by=['Indexcd', 'Trddt'], ascending=[True, True])

    # 存储所有数据框的列表
    dfs = []
    # 获取price_path路径下的所有文件
    search_pattern = os.path.join(price_path, '**', '*.csv')
    files = glob.glob(search_pattern, recursive=True)

    # 去除files中包含index的文件
    files = [file for file in files if 'Index' not in file]
    # 将所有文件的日期转为datetime
    for file in files:
        df = pd.read_csv(file)
        # 将df中的TradingDate列转为日期格式
        df['TradingDate'] = pd.to_datetime(df['TradingDate'])
        dfs.append(df)

    # 假设所有csv文件具有相同的列名，我们直接进行纵向（行方向）拼接
    if dfs:
        price_df = pd.concat(dfs, axis=0, ignore_index=True)  # 纵向拼接，忽略原来的索引
        # 将merged_df中的TradingDate列转为日期格式
        price_df['TradingDate'] = pd.to_datetime(price_df['TradingDate'])
        # 选择start_date到end_date之间的数据
        price_df = price_df[(price_df['TradingDate'] >= start_date) & (price_df['TradingDate'] <= end_date)]
        # 按照Symbol（升序）和TradingDate（升序）进行排序
        price_df = price_df.sort_values(by=['Symbol', 'TradingDate'], ascending=[True, True])
        return price_df, index_df
    else:
        print("没有找到任何CSV文件")
        return None
        


def GetStockReturn(stock_code, start_date, price_df, index_df, days=251):
    # 将start_date转化为pd.to_datetime
    start_date = pd.to_datetime(start_date)
    # 获取沪深300指数的数据
    index_data = index_df[index_df['Indexcd'] == 300].copy()
    index_data = index_data.sort_values(by='Trddt', ascending=True)
    if start_date in index_data['Trddt'].tolist():
        index_data = index_data[index_data['Trddt'] >= start_date]
    else:
        temp_date = index_data[index_data['Trddt'] < start_date].iloc[-1]['Trddt']
        index_data = index_data[index_data['Trddt'] >= temp_date]
    
    # 选择index_data中前days行
    index_data = index_data.head(days)
    # 计算沪深300指数的累计收益率
    index_data['ROI'] = index_data['Clsindex'].pct_change().fillna(0)
    index_data['CROI'] = (1 + index_data['ROI']).cumprod() - 1
    index_roi = index_data['ROI'].tolist()
    index_croi = index_data['CROI'].tolist()
    # 如果index_croi的长度小于days，则使用最后一个值填充至days
    if len(index_croi) < days:
        index_croi = index_croi + [index_croi[-1]] * (days - len(index_croi))


    # 获取目标股票的数据
    stock_data = price_df[price_df['Symbol'] == stock_code].copy()
    stock_data = stock_data.sort_values(by='TradingDate', ascending=True)
    if start_date in stock_data['TradingDate'].tolist():
        stock_data = stock_data[stock_data['TradingDate'] >= start_date]
    else:
        temp_date = stock_data[stock_data['TradingDate'] < start_date].iloc[-1]['TradingDate']
        stock_data = stock_data[stock_data['TradingDate'] >= temp_date]
    
    # 选择stock_data中前days行
    stock_data = stock_data.head(days)
    # 计算目标股票的累计收益率
    stock_data['ROI'] = stock_data['ClosePrice'].pct_change().fillna(0)
    stock_data['CROI'] = (1 + stock_data['ROI']).cumprod() - 1
    stock_roi = stock_data['ROI'].tolist()
    stock_croi = stock_data['CROI'].tolist()
    # 如果stock_croi的长度小于days，则使用最后一个值填充至days
    if len(stock_croi) < days:
        stock_croi = stock_croi + [stock_croi[-1]] * (days - len(stock_croi))

    # 计算超额收益率
    stock_ar = [stock_roi[i] - index_roi[i] for i in range(len(stock_roi))]
    stock_car = [stock_croi[i] - index_croi[i] for i in range(len(stock_croi))]


    return stock_roi, stock_ar, stock_croi, stock_car
    


def calculate_performance_metrics(returns, cumulative_returns):
    """
    输入：
        returns: list of daily returns (e.g., [0.01, -0.005, ...])
        cumulative_returns: list of cumulative excess returns (already minus 1)
    
    输出：
        pd.DataFrame: 一行，包含所有绩效指标
    """
    # 转换为 numpy 数组便于计算
    r = np.array(returns)
    cr = np.array(cumulative_returns)

    # 常量定义
    days_per_year = 252
    total_days = len(r)

    if total_days < 2:
        return pd.DataFrame([{}])  # 返回空数据框以防计算错误

    # 最大累计收益
    max_cr = np.max(cr)
    # 总收益
    total_return = cr[-1]

    # 年化收益率
    annualized_return = (1 + total_return) ** (days_per_year / total_days) - 1 if total_days > 0 else np.nan

    # 日度波动率
    volatility = np.std(r, ddof=1) if len(r) > 1 else np.nan

    # 年化波动率
    annualized_volatility = volatility * np.sqrt(days_per_year) if not np.isnan(volatility) else np.nan

    # 夏普比率（默认无风险利率为0）
    sharpe_ratio = annualized_return / annualized_volatility if not np.isnan(annualized_volatility) and annualized_volatility != 0 else np.nan

    # 最大回撤
    drawdowns = 1 - (1 + cr) / (1 + np.maximum.accumulate(cr))
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else np.nan

    # Calmar比率
    calmar_ratio = annualized_return / max_drawdown if not np.isnan(max_drawdown) and max_drawdown != 0 else np.nan

    # Sortino比率（下行波动率）
    downside_returns = r[r < 0]
    downside_deviation = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else np.nan
    annualized_downside_deviation = downside_deviation * np.sqrt(days_per_year) if not np.isnan(downside_deviation) else np.nan
    sortino_ratio = annualized_return / annualized_downside_deviation if not np.isnan(annualized_downside_deviation) and annualized_downside_deviation != 0 else np.nan

    # 胜率
    win_rate = np.mean(r > 0) if len(r) > 0 else np.nan

    # 盈亏比
    gains = r[r > 0]
    losses = -r[r < 0]
    profit_factor = np.sum(gains) / np.sum(losses) if len(losses) > 0 and np.sum(losses) != 0 else np.inf if len(gains) > 0 else np.nan

    # 其他统计
    avg_return = np.mean(r) if len(r) > 0 else np.nan
    avg_up_return = np.mean(gains) if len(gains) > 0 else np.nan
    avg_down_return = np.mean(losses) if len(losses) > 0 else np.nan
    max_up_return = np.max(r) if len(r) > 0 else np.nan
    max_down_return = np.min(r) if len(r) > 0 else np.nan

    # 构建结果字典
    metrics = {
        'Max_CR': max_cr,
        'Total_Return': total_return,
        'Annualized_Return': annualized_return,
        'Volatility': volatility,
        'Annualized_Volatility': annualized_volatility,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown,
        'Calmar_Ratio': calmar_ratio,
        'Sortino_Ratio': sortino_ratio,
        'Win_Rate': win_rate,
        'Profit_Factor': profit_factor,
        'Avg_Return': avg_return,
        'Avg_Up_Return': avg_up_return,
        'Avg_Down_Return': avg_down_return,
        'Max_Up_Return': max_up_return,
        'Max_Down_Return': max_down_return
    }

    # 转换为DataFrame
    df_metrics = pd.DataFrame([metrics])

    return df_metrics



# 计算
def GetEPSPred(i, rank, data, models, directions, price_df, index_df, strategy='TopKAve', days=251):
    # 获取group_rank
    group_rank = rank[i]
    # 获取report_list
    report_list =  group_rank['report_id']
    # 获取group_report
    group_report = data.query('ReportID in @report_list')
    # 设置ReportI
    group_report = group_report.set_index('ReportID')
    # 根据report_list获取group_report
    group_report = group_report.loc[report_list]
    feps = list(group_report['FEPS'])
        
    all_aggregated_feps = []
    # 计算concensus
    concensus=Concensus(feps)
    all_aggregated_feps.append(concensus)
    # 计算其他模型
    for j in range(len(models)):
        model = models[j]
        direct = directions[j]
        opinion_score =  group_rank[model]
        opinion_rank = ScoreToRank(opinion_score, direct)
        aggre_eps = EPSAggre(feps, opinion_score, opinion_rank, strategy, direct=direct, k=5, percent=0.5)
        all_aggregated_feps.append(aggre_eps)

    # 以dataframe形式记录all_aggregated_feps,其形状应该为一行，列名为models
    model_names = ['Concensus'] + models
    all_aggregated_feps = pd.DataFrame([all_aggregated_feps], columns=model_names)

    # 需要将group_report中的ReportDate转为datetime
    group_report['ReportDate'] = pd.to_datetime(group_report['ReportDate'])
    group_report = group_report.sort_values(by='ReportDate', ascending=False)
    # 选取group_report中ReportDate最大的一行,作为一个单独的只有一行的dataframe
    tar_report = group_report.head(1)

    # 合并 A 的第一行和 B, 其中A的列名是group_report的列名，B的列名是all_aggregated_feps的列名, 得到一个一行的dataframe
    tar_report = pd.concat([tar_report.reset_index(drop=True), all_aggregated_feps.reset_index(drop=True)], axis=1)
    # print(tar_report)
    
    try:
        # 根据tar_report中的ReportDate、StockID与ForHorizon,获取股票的收益率, 得到4个list
        stock_roi, stock_ar, stock_croi, stock_car = GetStockReturn(tar_report['StockID'].iloc[0], tar_report['ReportDate'].iloc[0], price_df, index_df, days)

        # 计算股票的收益表现指标
        stock_metrics_roi = calculate_performance_metrics(stock_roi, stock_croi)
        stock_metrics_ar = calculate_performance_metrics(stock_ar, stock_car)

        # 如果长度不足days,使用最后一个值将stock_roi, stock_ar, stock_croi, stock_car进行补齐，使其长度统一为days
        if len(stock_roi) < days:
            stock_roi = stock_roi + [stock_roi[-1]] * (days - len(stock_roi))
        if len(stock_ar) < days:
            stock_ar = stock_ar + [stock_ar[-1]] * (days - len(stock_ar))
        if len(stock_croi) < days:
            stock_croi = stock_croi + [stock_croi[-1]] * (days - len(stock_croi))
        if len(stock_car) < days:
            stock_car = stock_car + [stock_car[-1]] * (days - len(stock_car))


    except Exception as e:
        print(f"获取股票收益率时出错: {e}")
        # 返回空列表作为默认值
        stock_roi, stock_ar, stock_croi, stock_car = [0] * days, [0] * days, [0] * days, [0] * days
        stock_metrics_roi, stock_metrics_ar = pd.DataFrame(), pd.DataFrame()
    return tar_report, stock_roi, stock_ar, stock_croi, stock_car, stock_metrics_roi, stock_metrics_ar



# 采用并行计算，计算所有group的tar_report, stock_croi, stock_car
def GetAllEPSPred(rank, data, models, directions, price_df, index_df, strategy='TopKAve', days=251):

    # 使用循环
    results = []
    # 使用tqdm显示进度与耗时
    for i in tqdm(range(len(rank)), desc="Processing groups"):
        result = GetEPSPred(i, rank, data, models, directions, price_df, index_df, strategy, days)
        results.append(result)

    # 将results中的tar_report(dataframe), stock_croi(list), stock_car(list)分别合并
    # tar_report合并为一个dataframe, stock_croi与stock_car分别合并为一个numpy数组
    tar_report = pd.concat([result[0] for result in results], axis=0)
    stock_roi = [result[1] for result in results]
    stock_ar = [result[2] for result in results]
    stock_croi = [result[3] for result in results]
    stock_car = [result[4] for result in results]
    stock_metrics_roi = pd.concat([result[5] for result in results], axis=0)
    stock_metrics_ar = pd.concat([result[6] for result in results], axis=0)
    stock_roi = np.array(stock_roi)
    stock_ar = np.array(stock_ar)
    stock_croi = np.array(stock_croi)
    stock_car = np.array(stock_car)
    # 确保tar_report, stock_metrics_roi, stock_metrics_ar的index从0开始
    tar_report.index = range(len(tar_report))
    stock_metrics_roi.index = range(len(stock_metrics_roi))
    stock_metrics_ar.index = range(len(stock_metrics_ar))
    return tar_report, stock_roi, stock_ar, stock_croi, stock_car, stock_metrics_roi, stock_metrics_ar
    


def FilterTarReport(tar_report, stock_croi, stock_car, stock_metrics_roi, stock_metrics_ar):
    temp_tar_report = tar_report[(tar_report['PE'] < 50) & (tar_report['GroupSize'] >= 5)]
    temp_stock_croi = stock_croi[(tar_report['PE'] < 50) & (tar_report['GroupSize'] >= 5), :]
    temp_stock_car = stock_car[(tar_report['PE'] < 50) & (tar_report['GroupSize'] >= 5), :]
    temp_stock_metrics_roi = stock_metrics_roi[(tar_report['PE'] < 50) & (tar_report['GroupSize'] >= 5)]
    temp_stock_metrics_ar = stock_metrics_ar[(tar_report['PE'] < 50) & (tar_report['GroupSize'] >= 5)]
    # 重置index
    temp_tar_report.index = range(len(temp_tar_report))
    temp_stock_metrics_roi.index = range(len(temp_stock_metrics_roi))
    temp_stock_metrics_ar.index = range(len(temp_stock_metrics_ar))
    return temp_tar_report, temp_stock_croi, temp_stock_car, temp_stock_metrics_roi, temp_stock_metrics_ar



def ReturnValidation(tar_report, stock_croi, stock_car, stock_metrics_roi, stock_metrics_ar, models, top_rate=0.01, days=251):
    # 遍历models，计算ExpectedReturn, 其值为PE*FEPS/StockPrice - 1
    average_croi = []
    average_car = []
    average_metrics_roi = []
    average_metrics_ar = []
    for model in models:
        # tar_report[model + '_ER'] = (((tar_report['PE'] * tar_report[model]) / tar_report['StockPrice']) - 1)
        tar_report[model + '_ER'] = (tar_report[model] - tar_report['Concensus']) / tar_report['StockPrice']
        # 计算model_ER列最大的top_rate的阈值
        threshold = tar_report[model + '_ER'].nlargest(int(len(tar_report) * top_rate)).min()
        # threshold = tar_report[model + '_ER'].nlargest(int(top_rate)).min()
        # 根据threshold，将tar_report中model_ER大于threshold的行标记为1，否则为0
        tar_report[model + '_Mask'] = (tar_report[model + '_ER'] > threshold).astype(int)
        # 根据tar_report中model_Flag为1的行，获取stock_croi与stock_car中对应的行，
        # 然后计算stock_croi与stock_car的均值（忽略NA）
        top_ids = tar_report[tar_report[model + '_Mask'] == 1].index.tolist()
        # print("Selected top {} stocks".format(len(top_ids)))
        top_stock_croi = stock_croi[top_ids, :]
        top_stock_car = stock_car[top_ids, :]
        top_stock_metrics_roi = stock_metrics_roi.loc[top_ids, :]  # pandas dataframe
        top_stock_metrics_ar = stock_metrics_ar.loc[top_ids, :]  # pandas dataframe
        # 计算top_stock_croi与top_stock_car的均值（忽略NA）
        top_stock_croi_mean = np.nanmean(top_stock_croi, axis=0)  # 一维数组
        top_stock_car_mean = np.nanmean(top_stock_car, axis=0)  # 一维数组
        top_stock_metrics_roi_mean = top_stock_metrics_roi.mean()  # pandas dataframe
        top_stock_metrics_ar_mean = top_stock_metrics_ar.mean()  # pandas dataframe
        average_croi.append(top_stock_croi_mean)
        average_car.append(top_stock_car_mean)
        average_metrics_roi.append(top_stock_metrics_roi_mean.to_frame().T)
        average_metrics_ar.append(top_stock_metrics_ar_mean.to_frame().T)

    average_croi = np.array(average_croi)
    average_car = np.array(average_car)



    index = models
    columns = [f'D-{i}' for i in range(days)]

    average_croi_df = pd.DataFrame(average_croi, index=index, columns=columns)
    average_car_df = pd.DataFrame(average_car, index=index, columns=columns)
    # 合并average_metrics_roi与average_metrics_ar中的pandas dataframe元素，各个元素有相同的列名
    average_metrics_roi = pd.concat(average_metrics_roi, axis=0, ignore_index=True)
    average_metrics_ar = pd.concat(average_metrics_ar, axis=0, ignore_index=True)
    # 设置index
    average_metrics_roi.index = index
    average_metrics_ar.index = index

    return average_croi_df, average_car_df, average_metrics_roi, average_metrics_ar