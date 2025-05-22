import os
import sys
import platform

import time
import optuna
import timeit
from tqdm import tqdm  # 导入 tqdm 用于显示进度条
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import optuna
from pathlib import Path
from captum.attr import IntegratedGradients
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import spawn
import BaseFunction as BSF
import BayesianORNModels as RANKNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"  # 设置可见的GPU设备
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """清理分布式资源"""
    dist.destroy_process_group()


def softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum() + 1e-10)  # Adding a small constant for numerical stability

def save_feature_importance(feature_importance_np, EXP_ID, SelFeas):
    if dist.get_rank() == 0:
        # 创建原始的特征重要性数据框
        feature_importance_df = pd.DataFrame({
            'Feature': SelFeas,
            'Importance': feature_importance_np
        })

        # 计算softmax后的特征重要性
        importance_values = feature_importance_df['Importance'].values
        normalized_importance = softmax(importance_values)

        # 添加归一化后的特征重要性列
        feature_importance_df['Normalized_Importance'] = normalized_importance

        output_dir = Path(f"./output/table/Experiment-{EXP_ID}")
        output_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        feature_importance_csv_path = output_dir / "FeatureImportance.csv"
        feature_importance_df.to_csv(feature_importance_csv_path, index=False)
        print(f"Feature importance table is saved to {feature_importance_csv_path}")


def run_feature_importance(rank, world_size, EXP_ID, SelFeas, test_attn_layers, batch_size=1000):
    """在指定的rank上运行特征重要性计算"""
    setup(rank, world_size)

    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        mydevice = torch.device(f'cuda:{rank}')
    else:
        mydevice = torch.device('cpu')

    print(f"Experiment-{EXP_ID} on GPU {rank} is beginning now!")

    # 加载实验数据
    start_time = timeit.default_timer()
    _, _, _, _, _, _, size_tuple_list = BSF.LoadTrainData(EXP_ID, SelFeas)
    end_time = timeit.default_timer()
    print(f"Time of data loading and preprocessing is : {end_time - start_time} secs.")
    del (start_time, end_time)  


    # 读取CSV文件以获取最优超参数组合
    df = pd.read_csv(f"./output/table/Experiment-{EXP_ID}/Pairwise/BayesianORN/OptunaStudy.csv", index_col=0)
    best_trial_idx = df['value'].idxmin()  # 如果是最小化目标则使用 idxmin()
    best_trial = df.loc[best_trial_idx]
    best_trial_number = best_trial['number']
    print(f"Best trial number for Experiment-{EXP_ID}: {best_trial_number}")

    # 读取CSV文件以获取最优超参数组合
    df = pd.read_csv(f"./output/table/Experiment-{EXP_ID}/Pairwise/BayesianORN/OptunaStudy.csv", index_col=0)
    best_trial_idx = df['value'].idxmin()  # 如果是最小化目标则使用 idxmin()
    best_trial = df.loc[best_trial_idx]
    best_trial_number = best_trial['number']
    print(f"Best trial number for Experiment-{EXP_ID}: {best_trial_number}")

    # 获取最优超参数字典，并移除 'params_' 前缀
    best_params = {k.replace('params_', ''): v for k, v in best_trial.drop(['number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'state']).items()}
    # 确保所有的参数都是 Python 整数或浮点数
    best_params = {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v for k, v in best_params.items()}
    mock_trial = BSF.MockTrial(best_params)

    # 创建模型实例并加载权重
    model = RANKNN.OpinionRankNet(trial=mock_trial, size_tuple_list=size_tuple_list, attn_layers=test_attn_layers)
    model.to(mydevice)
    learned_model_path = Path(f"./output/table/Experiment-{EXP_ID}/Pairwise/BayesianORN/{best_trial_number}_TrainedModel.pth")
    
    if learned_model_path.exists():
        checkpoint = torch.load(learned_model_path, map_location=mydevice)
        model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(model_state_dict)
        print(f"Model loaded successfully for Experiment-{EXP_ID}.")
    else:
        print(f"Model not found in {learned_model_path} for Experiment-{EXP_ID}.")

    model = DDP(model, device_ids=[rank])

        
    # 定义一个辅助函数来进行前向传播
    def forward_func(X1, X2):
        # 确保模型在评估模式下运行
        model.eval()
        output = model(X1, X2)
        # 模型输出已经是概率值，无需额外转换
        return output  # 返回形状为 (batch_size,) 的张量 
    

    ig = IntegratedGradients(forward_func)
    # 初始化特征重要性累加器
    attributions_sum_x1 = None
    attributions_sum_x2 = None
    total_samples = 0

    for Test_ID in range(1, 6):  # 处理5个分块的测试数据
        print(f"Processing Test_ID {Test_ID}...")

        # 加载测试数据
        test_X_1, test_X_2, test_Y = BSF.LoadTestData(EXP_ID, Test_ID, SelFeas)
        test_dataset = BSF.PairwiseDataset(test_X_1, test_X_2, test_Y)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=4)

        for i, batch in enumerate(test_loader):
            x1_batch, x2_batch, y_batch = batch
            x1_batch = x1_batch.to(device=mydevice, dtype=torch.float).requires_grad_(True)
            x2_batch = x2_batch.to(device=mydevice, dtype=torch.float).requires_grad_(True)

            try:
                # 对于二分类问题，target设置为None，因为output已经是一个标量（概率）
                attributions, delta = ig.attribute((x1_batch, x2_batch), target=None, return_convergence_delta=True)
                # print(f"Delta: {delta.mean().item()}")  # 打印 delta 的平均值以检查收敛性
                # print(attributions[0][0].shape)
                # 更新总样本数
                current_batch_size = x1_batch.size(0)
                total_samples += current_batch_size

               # 检查是否需要初始化累加器
                if attributions_sum_x1 is None:
                    attributions_sum_x1 = torch.zeros_like(attributions[0][0], device=mydevice)
                    attributions_sum_x2 = torch.zeros_like(attributions[1][0], device=mydevice)

                # 累加特征重要性
                attributions_sum_x1 += attributions[0].sum(dim=0)
                attributions_sum_x2 += attributions[1].sum(dim=0)

                print(f'GPU {rank} - Experiment-{EXP_ID} - Test_ID {Test_ID} - Data Batch: {i} out of {len(test_loader)}')
            except RuntimeError as e:
                print(f"Error processing batch {i} on GPU {rank} for Experiment-{EXP_ID} Test_ID {Test_ID}: {e}")
                continue

            del x1_batch, x2_batch, y_batch, attributions, delta
            torch.cuda.empty_cache()

    # 收集所有 GPU 的特征重要性
    dist.barrier()
    if attributions_sum_x1 is not None and attributions_sum_x2 is not None:
        dist.all_reduce(attributions_sum_x1, op=dist.ReduceOp.SUM)
        dist.all_reduce(attributions_sum_x2, op=dist.ReduceOp.SUM)

    if rank == 0 and attributions_sum_x1 is not None and attributions_sum_x2 is not None:
        feature_importance_x1 = attributions_sum_x1 / total_samples
        feature_importance_x2 = attributions_sum_x2 / total_samples

        feature_importance_np_x1 = feature_importance_x1.cpu().detach().numpy()
        feature_importance_np_x2 = feature_importance_x2.cpu().detach().numpy()
        feature_importance_np = (feature_importance_np_x1 + feature_importance_np_x2) / 2

        print("Final feature importance before saving:")
        print(feature_importance_np)

        save_feature_importance(feature_importance_np, EXP_ID, SelFeas)  # 使用已构建好的函数保存特征重要性

    cleanup()




if __name__ == "__main__":
    EXP_IDs = [1, 2, 3]
    test_attn_layers = 3
    world_size = 2  # 使用两块GPU
    batch_size = 500
    SelFeas = [
        "AnaNum","Gender","Degree","StarTimes","AnaReportNum","AnaIndustryNum","AnaStockNum",
        "Per_ME_Err","Per_SD_Err","Per_ME_APE","Per_SD_APE","BrokerName","ActiveAnaNum",
        "BroReportNum","BroIndustryNum","BroStockNum","BroP_ME_APE","BroP_SD_APE",
        "IndustryID","ListYear","Udwnm","PE","PB","PS","Turnover","Liquidility","CircuMarketValue",
        "Volatility","Beta","Corr","NonSysRisk","StockPrice","ChangeRatio","CAR5","CAR20","CAR60",
        "StdRank","RankChan","FEPS","RevHorizon","ForHorizon","PrevEPS","AnaStockReportNum","AnaStockIndustryNum",
        "StockPer_ME_Err","StockPer_SD_Err","StockPer_ME_APE","StockPer_SD_APE","PreForNum","Boldness"]

    start_time = time.time()  # 记录开始时间
    print("Starting experiments...")

    # 使用 tqdm 包装实验 ID 列表以显示进度条
    for EXP_ID in tqdm(EXP_IDs, desc="Experiment Progress", unit="exp"):
        spawn(
            run_feature_importance,
            args=(world_size, EXP_ID, SelFeas, test_attn_layers, 500),
            nprocs=world_size,
            join=True
        )

        # 打印每个实验的耗时
        elapsed_time = time.time() - start_time
        print(f"Experiment-{EXP_ID} completed in {elapsed_time:.2f} seconds.")

    total_elapsed_time = time.time() - start_time
    print(f"All experiments completed in {total_elapsed_time:.2f} seconds.")

