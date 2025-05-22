# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:39:45 2024

@author: jiangshuai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



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
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Parameter, nn.Embedding)):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))


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
    
    

    
class MyAttention(nn.Module):
    def __init__(self, embedding_size):
        super(MyAttention, self).__init__()
        self.embedding_size = embedding_size
        self.query = nn.Linear(self.embedding_size, 1)
        self.act = nn.Tanh()

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Parameter, nn.Embedding)):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))

    def forward(self, X):
        # X.shape = (batch_size, fea_num, embedding_size)
        att_value = self.act(self.query(X))     # (batch_size)
        att_weight = F.softmax(att_value, dim=1)  # (batch_size, fn, 1)
        X_att = torch.squeeze(torch.bmm(X.transpose(1, 2), att_weight))  # (batch_size, embedding_size)
        return X_att, att_weight.squeeze()



class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, dropout, use_res=False, use_bn=False):
        super(DNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.use_res = use_res
        self.use_bn = use_bn
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
        
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Parameter, nn.Embedding, nn.BatchNorm1d)):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))


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
        return deep_input

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       



class MultiheadAttention(nn.Module):
    def __init__(self, emb_dim, head_num, scaling=True, use_residual=True):
        super(MultiheadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.head_num = head_num
        self.scaling = scaling
        self.use_residual = use_residual
        self.att_emb_size = emb_dim // head_num
        assert emb_dim % head_num == 0, "emb_dim must be divisible head_num"

        self.W_Q = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.W_K = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.W_V = nn.Parameter(torch.randn(emb_dim, emb_dim))

        if self.use_residual:
            self.W_R = nn.Parameter(torch.randn(emb_dim, emb_dim))

        # Initialize weights to avoid NaN values in computation
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Parameter, nn.Embedding)):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))

    def forward(self, inputs):
        # inputs (bs, fea_num, emb_size)
        # output (bs, fea_num, emb_size)
        '''1. 线性变换生成Q、K、V'''
        # dim: [batch_size, fields, emb_size]
        querys = torch.tensordot(inputs, self.W_Q, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_K, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_V, dims=([-1], [0]))
        # # 等价于 matmul
        # querys = torch.matmul(inputs, self.W_Q)
        # keys = torch.matmul(inputs, self.W_K)
        # values = torch.matmul(inputs, self.W_V)

        '''2. 分头'''
        # dim: [head_num, batch_size, fields, emb_size // head_num]
        querys = torch.stack(torch.split(querys, self.att_emb_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_emb_size, dim=2))
        values = torch.stack(torch.split(values, self.att_emb_size, dim=2))

        '''3. 缩放点积注意力'''
        # dim: [head_num, batch_size, fields, emb_size // head_num]
        inner_product = torch.matmul(querys, keys.transpose(-2, -1))
        # # 等价于
        # inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)
        if self.scaling:
            inner_product /= self.att_emb_size ** 0.5
        # Softmax归一化权重
        attn_w = F.softmax(inner_product, dim=-1)
        # 加权求和, attention结果与V相乘，得到多头注意力结果
        results = torch.matmul(attn_w, values)

        '''4. 拼接多头空间'''
        # dim: [batch_size, fields, emb_size]
        results = torch.cat(torch.split(results, 1, ), dim=-1)
        results = torch.squeeze(results, dim=0)

        # 跳跃连接
        if self.use_residual:
            results = results + torch.tensordot(inputs, self.W_R, dims=([-1], [0]))

        # results = F.relu(results)
        results = F.tanh(results)

        return results


    


class SymmetricNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, dropout, use_res=False):
        super(SymmetricNet, self).__init__()
        # 定义神经网络的层结构，这里只是示例，可以根据需要自定义层数和节点数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num
        self.dropout = dropout
        self.use_res = use_res                 
        self.fc = DNN(self.input_dim, self.hidden_dim, self.hidden_num, self.dropout, self.use_res) 
        self.output = nn.Linear(self.hidden_dim, 1) # 输出一个数，以便后续应用sigmoid

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # 正向传播输入x
        x_pos = self.fc(x)
        x_pos = self.output(x_pos)
        #x_pos = torch.sigmoid(x_pos)  # 确保输出在[0, 1]范围内

        # 正向传播输入-x
        x_neg = self.fc(-x)
        x_neg = self.output(x_neg)
        #x_neg = torch.sigmoid(x_neg)  # 确保输出在[0, 1]范围内
        # 满足条件（1）f(x) = 1 - f(-x)
        output = torch.sigmoid(x_pos - x_neg)
        return output
    
    

class OpinionRankNet(nn.Module):
    def __init__(self, trial, size_tuple_list, 
                 head_num=4, attn_layers=3, scaling=True, use_res=True, use_bn=True, mode="train"):
        super(OpinionRankNet, self).__init__()
        
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = 1
        self.dtype = torch.long
        self.int_act = nn.Tanh()
        self.head_num = head_num
        self.attn_layers = attn_layers
        self.scaling = scaling
        self.use_bn = use_bn
        self.use_res = use_res
        
        if mode != "train":
            hidden_dims = 256
            hidden_nums = 2
            dropout_rate = 0.5
            embedding_size = 32       
        else:       
            hidden_dims = trial.suggest_int("hidden_dims", 256, 1024, step=256)
            hidden_nums = trial.suggest_int("hidden_nums", 2, 6, step=1)
            dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7, step=0.05)
            embedding_size = trial.suggest_int("embedding_size", 32, 256, step=32)

        self.hidden_dims = hidden_dims
        self.hidden_nums = hidden_nums
        self.dropout = dropout_rate
        self.embedding_size = embedding_size
        
        
        self.dnn_input_dim = len(self.size_tuple_list) * self.embedding_size
        
        
        """
            check if use cuda
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        
        # Embedding layer
        self.fea_embeddings = FeatureEmbedding(self.size_tuple_list, self.embedding_size)
    
        # Interaction Layer
        # self.att_output_dim = len(self.size_tuple_list) * self.embedding_size
        multi_attn_layers = []
        for i in range(self.attn_layers):
            multi_attn_layers.append(MultiheadAttention(emb_dim=self.embedding_size, head_num=self.head_num, scaling=scaling, use_residual=self.use_res))
        self.multi_attn = nn.Sequential(*multi_attn_layers)
        # self.attn_fc = nn.Linear(self.att_output_dim, 1)
        
        # DNN layer
        self.dnn = SymmetricNet(self.dnn_input_dim, self.hidden_dims, self.hidden_nums, self.dropout, False)
        

    def forward(self, X_1, X_2):
        # feature embedding for opinion 1
        fea_emb_1 = self.fea_embeddings(X_1) #  (batch_size, fea_num, embedding_size)
        # feature interaction for opinion 1       
        attn_out_1 = self.multi_attn(fea_emb_1)   #  (batch_size, fea_num, embedding_size)
        attn_out_1 = torch.flatten(attn_out_1, start_dim=1) # (batch_size, fea_num*embedding_size)
        
        # feature embedding for opinion 2
        fea_emb_2 = self.fea_embeddings(X_2) #  (batch_size, fea_num, embedding_size)
        # feature interaction for opinion 2       
        attn_out_2 = self.multi_attn(fea_emb_2)   #  (batch_size, fea_num, embedding_size)
        attn_out_2 = torch.flatten(attn_out_2, start_dim=1) # (batch_size, fea_num*embedding_size)

        # dnn feed forword
        outs = self.dnn(attn_out_1 - attn_out_2).squeeze(1)   # (batch_size,)

        return outs