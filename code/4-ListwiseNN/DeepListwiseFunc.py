# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:15:30 2024

@author: jiangshuai
"""

import json
import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import rankdata
from collections import defaultdict




def LoadData(EXP_ID, SelFeas, min_size=5):
    file_path = "{}{}{}".format("./data/Experiment-",EXP_ID,"/ReportData.csv")
    data = pd.read_csv(file_path)
    
    # Filter out groups with size less than min_size
    data = data[data['GroupSize'] >= min_size]
    report_id = data['ReportID'].tolist()
    qid = data['GroupID'].tolist()
    Mask = data["Mask"].tolist()
    
    # Convert APE to rank within each group
    data['Rank'] = data.groupby('GroupID')['APE'].transform(lambda x: rankdata(-x, method='ordinal').astype(int))
    Y = np.array(data["Rank"])
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
    return qid, report_id, X, Y, Mask, size_tuple_list



class ListwiseCustomDataset(Dataset):
    def __init__(self, qid, report_id, X, Y, mask, size_tuple_list, mode='Train'):
        """
        初始化自定义的ListNet数据集。
        
        参数:
        - qid: 查询标识符列表。
        - report_id: 样本标识符列表。
        - X: 特征数组 (numpy 数组)。
        - Y: 排名数组。
        - mask: 数据集模式 ('Train', 'Val', 'Test') 列表。
        - size_tuple_list: 记录每个特征包含的类别数的namedtuple。
        - mode: 数据集模式 ('Train', 'Val', 'Test')。
        """
        self.mode = mode
        mask_array = np.array(mask)
        indices = mask_array == mode
        
        self.report_id = np.array(report_id)[indices]
        self.qid = np.array(qid)[indices]
        self.X = X[indices]
        self.Y = Y[indices]
        self.size_tuple_list = size_tuple_list
        
        # 创建qid到索引的映射
        unique_qids, qid_indices = np.unique(self.qid, return_inverse=True)
        self.qid_to_idx = {qid: idx for idx, qid in enumerate(unique_qids)}
        self.grouped_indices = [np.where(qid_indices == i)[0] for i in range(len(unique_qids))]

    def __len__(self):
        return len(self.qid_to_idx)

    def __getitem__(self, idx):
        group_indices = self.grouped_indices[idx]
        features = self.X[group_indices]
        ranks = self.Y[group_indices]
        report_ids = self.report_id[group_indices]
        
        # 转换为torch张量
        features_tensor = torch.tensor(features, dtype=torch.float32)
        ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
        report_ids_tensor = torch.tensor(report_ids, dtype=torch.int32)
        
        return {'features': features_tensor, 'ranks': ranks_tensor, 'report_ids': report_ids_tensor}



def collate_fn(batch):
    """
    自定义的collate函数用于处理不等长的输入。
    
    参数:
    - batch: 包含字典列表的批量数据，每个字典有'features'，'ranks'和'report_ids'键。
    
    返回:
    - padded_features: [batch_size, max_num_docs, num_features] 填充后的特征张量。
    - padded_ranks: [batch_size, max_num_docs] 填充后的排名张量。
    - doc_counts: [batch_size] 每个查询的有效文档数量。
    - report_ids: [batch_size, max_num_docs] 每个查询对应的报告ID。
    """
    features_list = [item['features'] for item in batch]  # batch_size*group_size*features_num
    ranks_list = [item['ranks'] for item in batch]  # batch_size * group_size
    report_ids_list = [item['report_ids'] for item in batch]  #batch_size * group_size

    # 找出最大文档数量
    max_num_docs = max([f.shape[0] for f in features_list])
    # num_features = features_list[0].shape[1]

    # 填充特征和排名
    padded_features = torch.stack([F.pad(f, (0, 0, 0, max_num_docs - f.shape[0]), value=0) for f in features_list])
    padded_ranks = torch.stack([F.pad(r, (0, max_num_docs - r.shape[0]), value=0) for r in ranks_list])

    # 记录每个查询的有效文档数量
    doc_counts = torch.tensor([f.shape[0] for f in features_list], dtype=torch.long)

    # 处理报告ID，用填充符号表示无效位置
    padded_report_ids = []
    for report_ids in report_ids_list:
        padded_report_ids.extend(list(report_ids) + ['PAD'] * (max_num_docs - len(report_ids)))

    return {
        'features': padded_features, 
        'ranks': padded_ranks, 
        'doc_counts': doc_counts, 
        'report_ids': padded_report_ids
    }





class ListNetLoss(nn.Module):
    """
    计算ListNet模型的损失。
    
    参数:
    - eps: epsilon 值，用于数值稳定性，默认 1e-10
    """
    def __init__(self, eps=1e-10):
        super(ListNetLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true, doc_counts):
        """
        前向传播计算损失。
        
        参数:
        - y_pred: [batch_size, max_num_docs] 预测的文档得分。
        - y_true: [batch_size, max_num_docs] 真实的文档排名。
        - doc_counts: [batch_size] 每个查询的有效文档数量。
        
        返回:
        - loss: 标量张量，表示平均损失。
        """
        # 复制输入以确保不会修改原始张量
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        batch_losses = []
        
        for i in range(len(doc_counts)):
            count = doc_counts[i]  # 获取当前查询的有效文档数量
            
            # 根据有效文档数量切片，只保留有效部分
            pred_i = y_pred[i, :count]  # 形状 [num_valid_docs]
            true_i = y_true[i, :count]  # 形状 [num_valid_docs]

            # 计算预测和真实得分的softmax分布
            preds_smax = F.softmax(pred_i, dim=0)  # 形状 [num_valid_docs]
            true_smax = F.softmax(true_i, dim=0)   # 形状 [num_valid_docs]

            # 添加一个小的epsilon值到预测的softmax结果中，防止log(0)
            preds_smax = preds_smax + self.eps  # 形状 [num_valid_docs]

            # 计算预测softmax结果的对数
            preds_log = torch.log(preds_smax)  # 形状 [num_valid_docs]

            # 计算交叉熵损失：-sum(true_distribution * log(predicted_distribution))
            loss_per_query = -torch.sum(true_smax * preds_log)  # 标量

            batch_losses.append(loss_per_query)

        # 返回批次中所有查询损失的平均值, 标量
        return torch.mean(torch.stack(batch_losses))





class FeatureEmbedding(nn.Module):
    def __init__(self, size_list, embedding_size):
        super(FeatureEmbedding, self).__init__()
        self.size_list = size_list
        self.embedding_size = embedding_size
        self.embeddings = nn.ModuleList()
        for fc in self.size_list:
            if fc.vocab_size > 1:
                self.embeddings.append(nn.Embedding(fc.vocab_size, self.embedding_size))
            else:
                self.embeddings.append(nn.Linear(1, self.embedding_size, bias=False))

        # xavier_uniform initialization for linear layers and embedding layers                
        '''
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Parameter, nn.Embedding)):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
        '''


    def forward(self, X):
        X_embedding_list = []
        for i in range(X.shape[-1]):
            if self.size_list[i].vocab_size == 1:
                tar = X[:, i].unsqueeze(dim=1)  # (batch_size, 1)
                # print(tar.shape, tar.dtype)
                X_embedding_list.append(self.embeddings[i](tar.float()))  # (batch_size,embedding_size)
            else:
                X_embedding_list.append(self.embeddings[i](X[:, i].long()))  # (batch_size,embedding_size)

        # (batch_size*fea_num*embedding_size)
        X_embedding = torch.stack([X_embedding_list[i] for i in range(len(X_embedding_list))], dim=1)

        return X_embedding



class DocFeatureEmbedding(nn.Module):
    def __init__(self, size_tuple_list, embedding_size):
        super(DocFeatureEmbedding, self).__init__()
        self.size_tuple_list = size_tuple_list
        self.embedding_size = embedding_size
        
        # 初始化FeatureEmbedding层
        self.feature_embedding = FeatureEmbedding(size_tuple_list, embedding_size)

    def forward(self, X):
        """
        输入: X - [batch_size, max_num_docs, num_features]
        输出: X_embedding - [batch_size, max_num_docs, num_features, embedding_size]
        """
        batch_size, max_num_docs, num_features = X.shape
        
        # 将X reshape成 [batch_size * max_num_docs, num_features] 以适应FeatureEmbedding层
        X_reshaped = X.view(-1, num_features)
        
        # 应用FeatureEmbedding层
        X_embedding = self.feature_embedding(X_reshaped)
        
        # 将结果reshape回 [batch_size, max_num_docs, num_features, embedding_size]
        X_embedding = X_embedding.view(batch_size, max_num_docs, num_features, self.embedding_size)
        
        return X_embedding



class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, dropout, num_outputs, use_res=False, use_bn=False):
        super(DNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.use_res = use_res
        self.use_bn = use_bn
        self.num_outputs = num_outputs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num
        self.layer_dims = [input_dim] + [hidden_dim] * hidden_num
        self.linears = nn.ModuleList([
            nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
            for i in range(len(self.layer_dims) - 1)
        ])
        if use_bn:
            self.bn = nn.ModuleList([
                nn.BatchNorm1d(self.layer_dims[i])
                for i in range(len(self.layer_dims))
            ])
        self.activation_layer = nn.ModuleList(
            [nn.Tanh() for _ in range(len(self.layer_dims) - 1)])
        
        self.output = nn.Linear(self.hidden_dim, self.num_outputs, bias=False)
        '''
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Parameter, nn.Embedding, nn.BatchNorm1d)):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
        '''


    def forward(self, inputs):
        # (bs, dense_dim + embed_size)
        deep_input = inputs
        #this_input = torch.clone(deep_input)
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            
            fc = self.activation_layer[i](fc)
            if i == 0:
                    this_input = torch.clone(fc)

            if self.use_res & (i > 0):
                deep_input = fc + this_input     # residual connection
            else: 
                deep_input = fc
            
            deep_input = self.dropout(deep_input)
            deep_out = self.output(deep_input)
        return deep_out





class ListNet(nn.Module):
    """
    A ListNet network with MSE loss for analyst opinion quality prediction problem.
    There are four parts in the architecture of this network:
    Domain representation, Second order domain interaction, Third order domain interaction and MLP prediction.
    In this network, we use dropout technology for all hidden layers,
    and "AdamW" method for optimization.
    """

    def __init__(self, trial, size_tuple_list):
        """
        Initialize a new network
        Inputs:
        - size_tuple_list: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interaction.
        """
        super(ListNet, self).__init__()
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = 1
        self.dtype = torch.long
        self.int_act = nn.Tanh()
                                  

        hidden_dims = trial.suggest_int("hidden_dims", 256, 1024, step=256)
        hidden_nums = trial.suggest_int("hidden_nums", 2, 6, step=1)
        dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7, step=0.05)
        embedding_size = trial.suggest_int("embedding_size", 16, 128, step=16)
        self.hidden_dims = hidden_dims
        self.hidden_nums = hidden_nums
        self.dropout = dropout_rate
        self.embedding_size = embedding_size
        self.deep_input_dim = len(self.size_tuple_list) * self.embedding_size   # [num_features*embedding_size]  

        """
            check if use cuda
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        """
            init DNN part
        """
        self.doc_feature_embedding = DocFeatureEmbedding(self.size_tuple_list, self.embedding_size)
        self.dnn = DNN(self.deep_input_dim, self.hidden_dims, self.hidden_nums, self.dropout, self.num_outputs) # [batch_szie*max_num_docs, num_features*embedding_size]  --> [batch_szie*max_num_docs, 1]

        '''
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Parameter, nn.Embedding)):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        '''

    def forward(self, X):
        """
        输入: X - [batch_size, max_num_docs, num_features]
        输出: scores - [batch_size, max_num_docs]
        """
        batch_size, max_num_docs, _ = X.size()
        # 嵌入层
        X_embedding = self.doc_feature_embedding(X)  # [batch_size, max_num_docs, num_features, embedding_size]
        # Reshape using view()  # [batch_size*max_num_docs, num_features*embedding_size]  
        X_flattened = X_embedding.view(X_embedding.size(0) * X_embedding.size(1), X_embedding.size(2) * X_embedding.size(3))
        output_flattened = self.dnn(X_flattened)         # (batch_size*max_num_docs, 1)
        output_flattened = output_flattened.squeeze(-1)  # (batch_size*max_num_docs,)
        """
        Final Output
        """
        final_output = output_flattened.view(batch_size, max_num_docs)         # (batch_size, max_num_docs)

        return final_output





class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        # Linear transformation to map input features to output features
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism parameters
        self.a_src = nn.Parameter(torch.empty(size=(out_features, 1)))
        self.a_dst = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # Initialize adjacency matrix here if it's always a full connection matrix
        self.register_buffer('adj_matrix', None)

    def forward(self, h):
        batch_size, max_num_docs, _ = h.shape
        
        # Create or reuse adjacency matrix on the correct device
        if self.adj_matrix is None or self.adj_matrix.shape[0] != max_num_docs:
            adj_matrix = torch.ones((max_num_docs, max_num_docs), device=h.device)
            # adj_matrix.fill_diagonal_(0)  # No self-loops
            self.adj_matrix = adj_matrix  # Store in buffer for reuse
            
        Wh = torch.matmul(h, self.W)  # [batch_size, max_num_docs, out_features]

        # Compute attention scores using separate parameters for source and destination nodes
        e_src = (Wh @ self.a_src).expand(-1, -1, max_num_docs)  # [batch_size, max_num_docs, max_num_docs]
        e_dst = (Wh @ self.a_dst).transpose(1, 2).expand(-1, max_num_docs, -1)  # [batch_size, max_num_docs, max_num_docs]
        
        e = self.leakyrelu(e_src + e_dst)  # [batch_size, max_num_docs, max_num_docs]

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(self.adj_matrix > 0, e, zero_vec)  # Apply adjacency mask
        
        attention = F.softmax(attention, dim=-1)  # Normalize attention scores
        attention = F.dropout(attention, self.dropout, training=self.training)  # Apply dropout

        h_prime = torch.matmul(attention, Wh)  # Aggregate neighbor features weighted by attention scores
        
        return h_prime





class ListGATNet(nn.Module):
    """
    A ListNet network with MSE loss for analyst opinion quality prediction problem.
    There are four parts in the architecture of this network:
    Domain representation, Second order domain interaction, Third order domain interaction and MLP prediction.
    In this network, we use dropout technology for all hidden layers,
    and "AdamW" method for optimization.
    """

    def __init__(self, trial, size_tuple_list):
        """
        Initialize a new network
        Inputs:
        - size_tuple_list: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interaction.
        """
        super(ListGATNet, self).__init__()
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = 1
        self.dtype = torch.long
        self.int_act = nn.Tanh()
                                  

        hidden_dims = trial.suggest_int("hidden_dims", 256, 1024, step=256)
        hidden_nums = trial.suggest_int("hidden_nums", 2, 6, step=1)
        dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7, step=0.05)
        embedding_size = trial.suggest_int("embedding_size", 16, 128, step=16)
        self.hidden_dims = hidden_dims
        self.hidden_nums = hidden_nums
        self.dropout = dropout_rate
        self.embedding_size = embedding_size
        self.deep_input_dim = len(self.size_tuple_list) * self.embedding_size   # [num_features*embedding_size]  

        """
            check if use cuda
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        """
            init DNN part
        """
        self.doc_feature_embedding = DocFeatureEmbedding(self.size_tuple_list, self.embedding_size)
        self.gnn = GATLayer(self.deep_input_dim, self.deep_input_dim, self.dropout)
        self.dnn = DNN(self.deep_input_dim, self.hidden_dims, self.hidden_nums, self.dropout, self.num_outputs) # [batch_szie,num_features*embedding_size]  -->[batch_szie,1]

        '''
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Parameter, nn.Embedding)):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        '''

    def forward(self, X):
        """
        输入: X - [batch_size, max_num_docs, num_features]
        输出: scores - [batch_size, max_num_docs]
        """
        batch_size, max_num_docs, _ = X.size()
        # 嵌入层
        X_embedding = self.doc_feature_embedding(X)  # [batch_size, max_num_docs, num_features, embedding_size]    
        # Reshape using view()  # [batch_size, max_num_docs, num_features*embedding_size]
        X_flattened_1 = X_embedding.view(X_embedding.size(0), X_embedding.size(1), -1)
        X_gnn = self.gnn(X_flattened_1)  # [batch_size, max_num_docs, num_features*embedding_size]
        del(X_flattened_1)
        # Reshape using view()  # [batch_size*max_num_docs, num_features*embedding_size]
        X_flattened_2 = X_gnn.view(X_gnn.size(0) * X_gnn.size(1), -1)
        del(X_gnn)
        output_flattened = self.dnn(X_flattened_2)         # (batch_size, max_num_docs, 1)
        output_flattened = output_flattened.squeeze(-1)  # (batch_size*max_num_docs,)
        """
        Final Output
        """
        final_output = output_flattened.view(batch_size, max_num_docs)         # (batch_size, max_num_docs)

        return final_output



def ListwisePreds(model, test_loader, mydevice):
    model.eval()
    preds = []
    # all_test_Y = []
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_X = test_data['features']
            test_Y = test_data['ranks']
            test_doc_counts = test_data['doc_counts']
            test_report_ids = test_data['report_ids']  # a list of lenth: batch_size*max_doc_nums
            test_X = test_X.to(device=mydevice, dtype=torch.float)
            test_Y = test_Y.to(device=mydevice, dtype=torch.float)
            test_doc_counts = test_doc_counts.to(device=mydevice, dtype=torch.int32)
            # test_report_ids = test_report_ids.to(device=mydevice, dtype=torch.int32)
            # test_Y = test_Y.to(device=mydevice, dtype=torch.float)
            batch_preds = model(test_X).view(-1)  # [batch_size, max_doc_nums]
            indices = [i for i, x in enumerate(test_report_ids) if x != 'PAD']  
            batch_preds = batch_preds[indices]  
            #Now selected_elements contains the elements from 'a' where b is not 'PAD'  
            test_report_ids = [x for x in test_report_ids if x != 'PAD'] #remove PAD elements from test_report_ids
            test_report_ids = torch.tensor(test_report_ids)
            test_report_ids = test_report_ids.to(device=mydevice, dtype=torch.int32)
            batch_preds = torch.stack((test_report_ids, batch_preds), dim=1)
            preds.append(batch_preds) #
            print(f'Test Batch: {i} out of {len(test_loader)}')
        
 
    preds = torch.cat(preds, dim=0)
    preds = preds.to(device=torch.device('cpu')).numpy()
    # 将 NumPy 数组转换为 Pandas DataFrame，并指定列名
    preds = pd.DataFrame(preds, columns=['ReportID', 'prediction'])
    
    # 如果 ReportID 应该是整数类型，可以进行类型转换
    preds['ReportID'] = preds['ReportID'].astype(int)
    
    return preds