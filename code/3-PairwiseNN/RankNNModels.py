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

        # 初始化, 避免计算得到nan
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




    
class EWDNN(nn.Module):
    """
    A EWDNN network with MSE loss for analyst opinion quality prediction problem.
    There are four parts in the architecture of this network:
    Domain representation, Second order domain interaction, Third order domain interaction and MLP prediction.
    In this network, we use dropout technology for all hidden layers,
    and "AdamW" method for optimization.
    """

    def __init__(self, size_tuple_list, hidden_nums, hidden_dims,dropout_rate, num_outputs=1):
        """
        Initialize a new network
        Inputs:
        - size_list: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interaction.
        """
        super(EWDNN, self).__init__()
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = num_outputs
        self.dtype = torch.long
        self.int_act = nn.Tanh()
        self.deep_input_dim = sum([fc.vocab_size for fc in self.size_tuple_list])
                                  

        self.hidden_dims = hidden_dims
        self.hidden_nums = hidden_nums
        self.dropout = dropout_rate

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

        self.dnn_0 = nn.Linear(self.deep_input_dim, self.hidden_dims)
        for i in range(1, self.hidden_nums):
            setattr(self, 'dnn_' + str(i),
                    nn.Linear(self.hidden_dims, self.hidden_dims))
            setattr(self, 'dnn_act_' + str(i),
                    self.int_act)
            setattr(self, 'dnn_dropout_' + str(i),
                    nn.Dropout(self.dropout))

        self.final_out = nn.Linear(self.hidden_dims, self.num_outputs, bias=False)

        #self.act_final = nn.Hardtanh(min_val=0, max_val=3)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Parameter, nn.Embedding)):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))

    def forward(self, X):
        """
        Forward process of network.
        Inputs:
        - X: A tensor of input's value, shape of (batch_size, feture_dim)
        """
        
        """
            Feature Embedding Part
        """
        ##  get input by encoding the catgorical features
        X_cat = []
        X_num = []
        for i in range(X.shape[-1]):
            if self.size_tuple_list[i].vocab_size == 1:
                X_num.append(X[:, i].unsqueeze(dim=1))  # (batch_size,1)
            else:
                X_cat.append(F.one_hot(X[:, i].long(), num_classes=self.size_tuple_list[i].vocab_size))

        X_cat = torch.cat(X_cat, dim=1).float()
        X_num = torch.cat(X_num, dim=1).float()
        X = torch.cat([X_num, X_cat], dim=1)
        del(X_cat, X_num)


        """
            MLP Prediction
        """
        dnn_input = self.int_act(self.dnn_0(X))

        for i in range(1, self.hidden_nums):
            dnn_input = getattr(self, 'dnn_' + str(i))(dnn_input)
            dnn_input = getattr(self, 'dnn_act_' + str(i))(dnn_input)
            dnn_input = getattr(self, 'dnn_dropout_' + str(i))(dnn_input)
        # (batch_size, hidden_dims)

        """
        Final Output
        """
        final_output = self.final_out(dnn_input).squeeze()
            # (batch_size)
        return final_output




class ResNet(nn.Module):

    def __init__(self, size_tuple_list, hidden_nums, hidden_dims,dropout_rate, num_outputs=1):

        super(ResNet, self).__init__()
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = num_outputs
        self.dtype = torch.long
        self.int_act = nn.Tanh()
        self.deep_input_dim = sum([fc.vocab_size for fc in self.size_tuple_list])
         
        
        self.hidden_dims = hidden_dims
        self.hidden_nums = hidden_nums
        self.dropout = dropout_rate


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

        self.dnn_0 = nn.Linear(self.deep_input_dim, self.hidden_dims)
        for i in range(1, self.hidden_nums):
            setattr(self, 'dnn_' + str(i),
                    nn.Linear(self.hidden_dims, self.hidden_dims))
            setattr(self, 'dnn_act_' + str(i),
                    self.int_act)
            setattr(self, 'dnn_dropout_' + str(i),
                    nn.Dropout(self.dropout))

        self.final_out = nn.Linear(self.hidden_dims, self.num_outputs, bias=False)

        # self.act_final = nn.Hardtanh(min_val=0, max_val=3)
        
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
 
    def forward(self, X):
        """
        Forward process of network.
        """
        X_cat = []
        X_num = []
        for i in range(X.shape[-1]):
            if self.size_tuple_list[i].vocab_size == 1:
                X_num.append(X[:, i].unsqueeze(dim=1))  # (batch_size,1)
            else:
                X_cat.append(F.one_hot(X[:, i].long(), num_classes=self.size_tuple_list[i].vocab_size))

        X_cat = torch.cat(X_cat, dim=1).float()
        X_num = torch.cat(X_num, dim=1).float()
        X = torch.cat([X_num, X_cat], dim=1)
        del(X_cat, X_num)
        
        
        """
            MLP Prediction
        """
        dnn_input = self.int_act(self.dnn_0(X))
        this_input = torch.clone(dnn_input)
        for i in range(1, self.hidden_nums):
            dnn_input = getattr(self, 'dnn_' + str(i))(dnn_input)
            dnn_input = getattr(self, 'dnn_act_' + str(i))(dnn_input)
            dnn_input = getattr(self, 'dnn_dropout_' + str(i))(dnn_input)
            dnn_input = dnn_input + this_input
            # (batch_size, hidden_dims)

        """
        Final Output
        """
        final_output = self.final_out(dnn_input).squeeze()
            # (batch_size)
        return final_output



class FM(nn.Module):
    """
    A FM network with RMSE loss for rates prediction problem.
    There are two parts in the architecture of this network: fm part for low
    order interactions of features and deep part for higher order. In this
    network, we use bachnorm and dropout technology for all hidden layers,
    and "Adam" method for optimazation.
    You may find more details in this paper:
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
    """

    def __init__(self, size_tuple_list, embedding_size, num_outputs=1):
        """
        Initialize a new network
        Inputs:
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super(FM, self).__init__()
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = num_outputs
        self.dtype = torch.long
        self.int_act = nn.Tanh()
    
        self.embedding_size = embedding_size
        """
            check if use cuda
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        """
            init Linear part
        """
        self.linear_embeddings = FeatureEmbedding(self.size_tuple_list, 1)
        """
            FM part
        """
        self.fm_embeddings = FeatureEmbedding(self.size_tuple_list, self.embedding_size)
        
        #self.act_final = nn.Hardtanh(min_val=0, max_val=3)


    def forward(self, X):
        """
        Forward process of network.
        """
        
        """
            Linear Part
        """
        linear_emb = self.linear_embeddings(X)  # (batch_size*fea_num*1)
        linear_out = linear_emb.squeeze()  # (batch_size, fea_num)
        """
            FM Part
        """
        fm_emb = self.fm_embeddings(X) #  (batch_size,fea_num,embedding_size)
        fm_sum_second_order_emb = torch.sum(fm_emb, dim=1)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb.pow(2)  # (x+y)^2, (batch_size,embedding_size)
        fm_second_order_emb_square = fm_emb * fm_emb  # (batch_size,fea_num,embedding_size)
        fm_second_order_emb_square_sum = torch.sum(fm_second_order_emb_square, dim=1)  # x^2+y^2, (batch_size,embedding_size)
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
        """
            sum
        """
        total_sum = torch.sum(linear_out, 1) + torch.sum(fm_second_order, 1)
        return total_sum



class WideAndDeep(nn.Module):

    def __init__(self, size_tuple_list, hidden_nums, hidden_dims,dropout_rate, num_outputs=1):
        """
        Initialize a new network
        Inputs:
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super(WideAndDeep, self).__init__()
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = num_outputs
        self.dtype = torch.long
        self.int_act = nn.Tanh()
        self.deep_input_dim = sum([fc.vocab_size for fc in self.size_tuple_list])
        
        self.hidden_dims = hidden_dims
        self.hidden_nums = hidden_nums
        self.dropout = dropout_rate
        """
            check if use cuda
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        """
            init Linear part
        """
        self.linear_embeddings = FeatureEmbedding(self.size_tuple_list, 1)
        """
            init DNN part
        """
        self.dnn_0 = nn.Linear(self.deep_input_dim, self.hidden_dims)
        for i in range(1, self.hidden_nums):
            setattr(self, 'dnn_' + str(i),
                    nn.Linear(self.hidden_dims, self.hidden_dims))
            setattr(self, 'dnn_act_' + str(i),
                    self.int_act)
            setattr(self, 'dnn_dropout_' + str(i),
                    nn.Dropout(self.dropout))

        self.final_out = nn.Linear(self.hidden_dims, self.num_outputs, bias=False)

        #self.act_final = nn.Hardtanh(min_val=0, max_val=3)
                
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))


    def forward(self, X):
        """
        Forward process of network.
        """

        """
            Linear Part
        """
        linear_emb = self.linear_embeddings(X)  # (batch_size*fea_num*1)
        linear_out = linear_emb.squeeze()  # (batch_size, fea_num)
        """
            DNN Part
        """
        ##  get input by encoding the catgorical features
        X_cat = []
        X_num = []
        for i in range(X.shape[-1]):
            if self.size_tuple_list[i].vocab_size == 1:
                X_num.append(X[:, i].unsqueeze(dim=1))  # (batch_size,1)
            else:
                X_cat.append(F.one_hot(X[:, i].long(), num_classes=self.size_tuple_list[i].vocab_size))

        X_cat = torch.cat(X_cat, dim=1).float()
        X_num = torch.cat(X_num, dim=1).float()
        X = torch.cat([X_num, X_cat], dim=1)
        del(X_cat, X_num)


        """
            MLP Prediction
        """
        dnn_input = self.int_act(self.dnn_0(X))

        for i in range(1, self.hidden_nums):
            dnn_input = getattr(self, 'dnn_' + str(i))(dnn_input)
            dnn_input = getattr(self, 'dnn_act_' + str(i))(dnn_input)
            dnn_input = getattr(self, 'dnn_dropout_' + str(i))(dnn_input)
        # (batch_size, hidden_dims)

        """
        Final Output
        """
        dnn_out = self.final_out(dnn_input)
        """
            sum
        """
        total_sum = torch.sum(linear_out, 1) + torch.sum(dnn_out, 1)
        return total_sum


class DeepFM(nn.Module):
    """
    A FM network with RMSE loss for rates prediction problem.
    There are two parts in the architecture of this network: fm part for low
    order interactions of features and deep part for higher order. In this
    network, we use bachnorm and dropout technology for all hidden layers,
    and "Adam" method for optimazation.
    You may find more details in this paper:
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
    """

    def __init__(self, size_tuple_list, hidden_nums, hidden_dims,dropout_rate, embedding_size, num_outputs=1):
        """
        Initialize a new network
        Inputs:
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super(DeepFM, self).__init__()
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = num_outputs
        self.dtype = torch.long
        self.int_act = nn.Tanh()
        self.deep_input_dim = sum([fc.vocab_size for fc in self.size_tuple_list])
        
        self.hidden_dims = hidden_dims
        self.hidden_nums = hidden_nums
        self.dropout = dropout_rate
        self.embedding_size = embedding_size
        """
            check if use cuda
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        """
            init Linear part
        """
        self.linear_embeddings = FeatureEmbedding(self.size_tuple_list, 1)
        """
            FM part
        """
        self.fm_embeddings = FeatureEmbedding(self.size_tuple_list, self.embedding_size)
        
        self.act_final = nn.Hardtanh(min_val=0, max_val=3)
        
        """
            init DNN part
        """
        self.dnn_0 = nn.Linear(self.deep_input_dim, self.hidden_dims)
        for i in range(1, self.hidden_nums):
            setattr(self, 'dnn_' + str(i),
                    nn.Linear(self.hidden_dims, self.hidden_dims))
            setattr(self, 'dnn_act_' + str(i),
                    self.int_act)
            setattr(self, 'dnn_dropout_' + str(i),
                    nn.Dropout(self.dropout))

        self.final_out = nn.Linear(self.hidden_dims, self.num_outputs, bias=False)
        #self.act_final = nn.Hardtanh(min_val=0, max_val=3)
        
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))


    def forward(self, X):
        """
        Forward process of network.
        """
        
        """
            Linear Part
        """
        linear_emb = self.linear_embeddings(X)  # (batch_size*fea_num*1)
        linear_out = linear_emb.squeeze()  # (batch_size, fea_num)
        """
            FM Part
        """
        fm_emb = self.fm_embeddings(X) #  (batch_size,fea_num,embedding_size)
        fm_sum_second_order_emb = torch.sum(fm_emb, dim=1)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb.pow(2)  # (x+y)^2, (batch_size,embedding_size)
        fm_second_order_emb_square = fm_emb * fm_emb  # (batch_size,fea_num,embedding_size)
        fm_second_order_emb_square_sum = torch.sum(fm_second_order_emb_square, dim=1)  # x^2+y^2, (batch_size,embedding_size)
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
        """
            DNN Part
        """
        ##  get input by encoding the catgorical features
        X_cat = []
        X_num = []
        for i in range(X.shape[-1]):
            if self.size_tuple_list[i].vocab_size == 1:
                X_num.append(X[:, i].unsqueeze(dim=1))  # (batch_size,1)
            else:
                X_cat.append(F.one_hot(X[:, i].long(), num_classes=self.size_tuple_list[i].vocab_size))

        X_cat = torch.cat(X_cat, dim=1).float()
        X_num = torch.cat(X_num, dim=1).float()
        X = torch.cat([X_num, X_cat], dim=1)
        del(X_cat, X_num)
        """
            DNN Prediction
        """
        dnn_input = self.int_act(self.dnn_0(X))

        for i in range(1, self.hidden_nums):
            dnn_input = getattr(self, 'dnn_' + str(i))(dnn_input)
            dnn_input = getattr(self, 'dnn_act_' + str(i))(dnn_input)
            dnn_input = getattr(self, 'dnn_dropout_' + str(i))(dnn_input)
        # (batch_size, hidden_dims)

        """
            DNN Output
        """
        dnn_out = self.final_out(dnn_input)
        
        """
            sum
        """
        total_sum = torch.sum(linear_out, 1) + torch.sum(fm_second_order, 1) + torch.sum(dnn_out, 1)
        return total_sum



class AutoIntNet(nn.Module):
    def __init__(self, size_tuple_list, hidden_dims, hidden_nums, dropout_rate, embedding_size, num_outputs=1, 
                 head_num=4, attn_layers=3, scaling=True, use_residual=True, use_bn=True):
        super(AutoIntNet, self).__init__()
        
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = num_outputs
        self.dtype = torch.long
        self.int_act = nn.Tanh()
        self.head_num = head_num
        self.attn_layers = attn_layers
        
        
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
            multi_attn_layers.append(MultiheadAttention(emb_dim=self.embedding_size, head_num=self.head_num, scaling=scaling, use_residual=use_residual))
        self.multi_attn = nn.Sequential(*multi_attn_layers)
        # self.attn_fc = nn.Linear(self.att_output_dim, 1)
        
        # DNN layer
        self.dnn = DNN(self.dnn_input_dim, self.hidden_dims, self.hidden_nums, self.dropout)
        
        # Output layer
        self.final = nn.Linear(self.hidden_dims, self.num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, X):
        # feature embedding
        fea_emb = self.fea_embeddings(X) #  (batch_size, fea_num, embedding_size)
        # feature interaction        
        attn_out = self.multi_attn(fea_emb)   #  (batch_size, fea_num, embedding_size)
        attn_out = torch.flatten(attn_out, start_dim=1) # (batch_size, fea_num*embedding_size)
        # dnn feed forword
        dnn_out = self.dnn(attn_out)   # (batch_size, hidden_dims)
        # final prediction
        outs = self.final(dnn_out).squeeze(1) # (batch_size)

        return outs



class RankAutoIntNet(nn.Module):

    def __init__(self, trial, size_tuple_list,
                 head_num=4, attn_layers=3, scaling=True, use_residual=True, use_bn=True, mode="train"):
        super(RankAutoIntNet, self).__init__()
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = 1
        self.dtype = torch.long
        self.int_act = nn.Tanh()
        self.head_num = head_num
        self.attn_layers = attn_layers
        
        if mode != "train":
            hidden_dims = 128
            hidden_nums = 2
            dropout_rate = 0.5
            embedding_size = 16
        else:   
            hidden_dims = trial.suggest_int("hidden_dims", 128, 1024, step=128)
            hidden_nums = trial.suggest_int("hidden_nums", 2, 6, step=1)
            dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7, step=0.05)
            embedding_size = trial.suggest_int("embedding_size", 16, 160, step=16)         
       
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
        
        self.opinion_net = AutoIntNet(self.size_tuple_list, self.hidden_dims, self.hidden_nums, self.dropout, self.embedding_size, self.num_outputs, 
                     head_num=2, attn_layers=2, scaling=scaling, use_residual=use_residual, use_bn=use_bn)
        self.final_act = nn.Sigmoid()
        
        
    def forward(self, X_1, X_2):
        """
        Forward process of network.
        """
        
        """
            Forward Part
        """
        opinion_1 = self.opinion_net(X_1)
        opinion_2 = self.opinion_net(X_2)
        # (batch_size)
        """
            Output
        """
        rank_out = self.final_act(opinion_1 - opinion_2)
        # (batch_size)
        return rank_out


class CrossNet(nn.Module):
    def __init__(self, in_features, layer_num=3, parameterization='vector'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            self.kernels = nn.Parameter(torch.randn(self.layer_num, in_features, 1))
        elif self.parameterization == 'matrix':
            self.kernels = nn.Parameter(torch.randn(self.layer_num, in_features, in_features))
        self.bias = nn.Parameter(torch.randn(self.layer_num, in_features, 1))

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Parameter)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_1 = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                x1_w = torch.tensordot(x_1, self.kernels[i], dims=([1], [0]))
                #print("kernels shape:", self.kernels[i].shape, "x_1 shape:", x_1.shape, "x1_w shape:", x1_w.shape)
                #print("kernels", torch.isnan(self.kernels[i]).any(), torch.isnan(self.kernels[i]).all())
                #print("x1_w", torch.isnan(x1_w).any(), torch.isnan(x1_w).all())
                dot_ = torch.matmul(x_0, x1_w)
                x_1 = dot_ + self.bias[i] + x_1
            else:
                x1_w = torch.tensordot(self.kernels[i], x_1)
                dot_ = x1_w + self.bias[i]
                x_1 = x_0 * dot_ + x_1
        x_1 = torch.squeeze(x_1, dim=2)
        return x_1


class DCN(nn.Module):
    def __init__(self, size_tuple_list, hidden_dims, hidden_nums, dropout_rate, embedding_size,
                cross_num=3, cross_param='vector', init_std=0.0001, l2_reg=0.00001):
        super(DCN, self).__init__()
        
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.dtype = torch.long
        self.int_act = nn.Tanh()
        self.cross_num = cross_num
        self.cross_param = cross_param
        
        self.hidden_dims = hidden_dims
        self.hidden_nums = hidden_nums
        self.dropout = dropout_rate
        self.embedding_size = embedding_size
                
        self.dnn_input_dim = len(self.size_tuple_list) * self.embedding_size
        self.l2_reg = l2_reg
        
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
        self.crossnet = CrossNet(self.dnn_input_dim, layer_num=self.cross_num, parameterization=self.cross_param)
        
        # DNN layer
        self.dnn = DNN(self.dnn_input_dim, self.hidden_dims, self.hidden_nums, self.dropout)
        
        # Output layer
        self.final = nn.Linear(self.dnn_input_dim + self.hidden_dims, 1, bias=False)


    def forward(self, X):
        # feature embedding
        fea_emb = self.fea_embeddings(X) #  (batch_size, fea_num, embedding_size)
        fea_emb = torch.flatten(fea_emb, start_dim=1)  #  (batch_size, fea_num*embedding_size)
        # feature interaction        
        cross_out = self.crossnet(fea_emb)   #  (batch_size, fea_num*embedding_size)
        # dnn feed forword
        dnn_out = self.dnn(fea_emb)   # (batch_size, hidden_dims)
        # final prediction
        stack_out = torch.cat((cross_out, dnn_out), dim=-1)
        outs = self.final(stack_out).squeeze(1) # (batch_size)
        return outs



class RankDCN(nn.Module):

    def __init__(self, trial, size_tuple_list, 
                 cross_num=3, cross_param='vector', init_std=0.0001, l2_reg=0.00001, mode="train"):
        super(RankDCN, self).__init__()
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = 1
        self.dtype = torch.long
        self.int_act = nn.Tanh()
        self.cross_num = cross_num
        self.cross_param = cross_param
        
        if mode != "train":
            hidden_dims = 128
            hidden_nums = 2
            dropout_rate = 0.5
            embedding_size = 16
        else:   
            hidden_dims = trial.suggest_int("hidden_dims", 128, 1024, step=128)
            hidden_nums = trial.suggest_int("hidden_nums", 2, 6, step=1)
            dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7, step=0.05)
            embedding_size = trial.suggest_int("embedding_size", 16, 160, step=16) 

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
        
        self.opinion_net = DCN(self.size_tuple_list, self.hidden_dims, self.hidden_nums, self.dropout, self.embedding_size, self.cross_num, self.cross_param)
        self.final_act = nn.Sigmoid()
        
        
    def forward(self, X_1, X_2):
        """
        Forward process of network.
        """
        
        """
            Forward Part
        """
        opinion_1 = self.opinion_net(X_1)
        opinion_2 = self.opinion_net(X_2)
        # (batch_size)
        """
            Output
        """
        rank_out = self.final_act(opinion_1 - opinion_2)
        # (batch_size)
        return rank_out

    
class RankEWDNN(nn.Module):

    def __init__(self, trial, size_tuple_list, mode="train"):
        """
        Initialize a new network
        Inputs:
        - size_list: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interaction.
        """
        super(RankEWDNN, self).__init__()
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = 1
        self.dtype = torch.long
        self.int_act = nn.Tanh()
        self.deep_input_dim = sum([fc.vocab_size for fc in self.size_tuple_list])

        if mode != "train":
            hidden_dims = 256
            hidden_nums = 2
            dropout_rate = 0.5
        else:   
            hidden_dims = trial.suggest_int("hidden_dims", 256, 1024, step=256)
            hidden_nums = trial.suggest_int("hidden_nums", 2, 6, step=1)
            dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7, step=0.05)      
            
        self.hidden_dims = hidden_dims
        self.hidden_nums = hidden_nums
        self.dropout = dropout_rate
        """
            check if use cuda
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.opinion_net = EWDNN(self.size_tuple_list, self.hidden_nums, self.hidden_dims, self.dropout, self.num_outputs)
        self.final_act = nn.Sigmoid()
        
        
    def forward(self, X_1, X_2):
        """
        Forward process of network.
        """
        
        """
            Forward Part
        """
        opinion_1 = self.opinion_net(X_1)
        opinion_2 = self.opinion_net(X_2)
        # (batch_size)
        """
            Output
        """
        rank_out = self.final_act(opinion_1 - opinion_2)
        # (batch_size)
        return rank_out
    


class RankResNet(nn.Module):

    def __init__(self, trial, size_tuple_list, mode="train"):
        """
        Initialize a new network
        Inputs:
        - size_list: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interaction.
        """
        super(RankResNet, self).__init__()
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = 1
        self.dtype = torch.long
        self.int_act = nn.Tanh()
        self.deep_input_dim = sum([fc.vocab_size for fc in self.size_tuple_list])

        if mode != "train":
            hidden_dims = 256
            hidden_nums = 2
            dropout_rate = 0.5
        else:   
            hidden_dims = trial.suggest_int("hidden_dims", 256, 1024, step=256)
            hidden_nums = trial.suggest_int("hidden_nums", 2, 6, step=1)
            dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7, step=0.05)      
        
        self.hidden_dims = hidden_dims
        self.hidden_nums = hidden_nums
        self.dropout = dropout_rate
        """
            check if use cuda
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.opinion_net = ResNet(self.size_tuple_list, self.hidden_nums, self.hidden_dims, self.dropout, self.num_outputs)
        self.final_act = nn.Sigmoid()
        
        
    def forward(self, X_1, X_2):
        """
        Forward process of network.
        """
        
        """
            Forward Part
        """
        opinion_1 = self.opinion_net(X_1)
        opinion_2 = self.opinion_net(X_2)
        # (batch_size)
        """
            Output
        """
        rank_out = self.final_act(opinion_1 - opinion_2)
        # (batch_size)
        return rank_out


class RankFM(nn.Module):

    def __init__(self, trial, size_tuple_list, mode="train"):
        """
        Initialize a new network
        Inputs:
        - size_list: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interaction.
        """
        super(RankFM, self).__init__()
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = 1
        self.dtype = torch.long
        self.int_act = nn.Tanh()
        self.deep_input_dim = sum([fc.vocab_size for fc in self.size_tuple_list])
        if mode != "train":
            embedding_size = 32
        else:       
            embedding_size = trial.suggest_int("embedding_size", 32, 256, step=32)
        
        self.embedding_size = embedding_size
        """
            check if use cuda
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.opinion_net = FM(self.size_tuple_list, self.embedding_size, self.num_outputs)
        self.final_act = nn.Sigmoid()
        
        
    def forward(self, X_1, X_2):
        """
        Forward process of network.
        """
        
        """
            Forward Part
        """
        opinion_1 = self.opinion_net(X_1)
        opinion_2 = self.opinion_net(X_2)
        # (batch_size)
        """
            Output
        """
        rank_out = self.final_act(opinion_1 - opinion_2)
        # (batch_size)
        return rank_out


class RankWAD(nn.Module):

    def __init__(self, trial, size_tuple_list, mode="train"):
        """
        Initialize a new network
        Inputs:
        - size_list: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interaction.
        """
        super(RankWAD, self).__init__()
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = 1
        self.dtype = torch.long
        self.int_act = nn.Tanh()
        self.deep_input_dim = sum([fc.vocab_size for fc in self.size_tuple_list])
        
        if mode != "train":
            hidden_dims = 256
            hidden_nums = 2
            dropout_rate = 0.5
        else:   
            hidden_dims = trial.suggest_int("hidden_dims", 256, 1024, step=256)
            hidden_nums = trial.suggest_int("hidden_nums", 2, 6, step=1)
            dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7, step=0.05)

        self.hidden_dims = hidden_dims
        self.hidden_nums = hidden_nums
        self.dropout = dropout_rate
        """
            check if use cuda
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.opinion_net = WideAndDeep(self.size_tuple_list, self.hidden_nums, self.hidden_dims, self.dropout, self.num_outputs)
        self.final_act = nn.Sigmoid()
        
        
    def forward(self, X_1, X_2):
        """
        Forward process of network.
        """
        
        """
            Forward Part
        """
        opinion_1 = self.opinion_net(X_1)
        opinion_2 = self.opinion_net(X_2)
        # (batch_size)
        """
            Output
        """
        rank_out = self.final_act(opinion_1 - opinion_2)
        # (batch_size)
        return rank_out
    
    
    
class RankDeepFM(nn.Module):

    def __init__(self, trial, size_tuple_list, mode="train"):
        """
        Initialize a new network
        Inputs:
        - size_list: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interaction.
        """
        super(RankDeepFM, self).__init__()
        self.size_tuple_list = size_tuple_list  # [(name="",vocab_size=10),...]
        self.num_outputs = 1
        self.dtype = torch.long
        self.int_act = nn.Tanh()
        self.deep_input_dim = sum([fc.vocab_size for fc in self.size_tuple_list])

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
        
        """
            check if use cuda
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.opinion_net = DeepFM(self.size_tuple_list, self.hidden_nums, self.hidden_dims, self.dropout, self.embedding_size, self.num_outputs)
        self.final_act = nn.Sigmoid()
        
        
    def forward(self, X_1, X_2):
        """
        Forward process of network.
        """
        
        """
            Forward Part
        """
        opinion_1 = self.opinion_net(X_1)
        opinion_2 = self.opinion_net(X_2)
        # (batch_size)
        """
            Output
        """
        rank_out = self.final_act(opinion_1 - opinion_2)
        # (batch_size)
        return rank_out
    


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




class OpinionRankResNet(nn.Module):
    def __init__(self, trial, size_tuple_list, 
                 head_num=4, attn_layers=3, scaling=True, use_res=True, use_bn=True, mode="train"):
        super(OpinionRankResNet, self).__init__()
        
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
        self.dnn = SymmetricNet(self.dnn_input_dim, self.hidden_dims, self.hidden_nums, self.dropout, True)
        

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
        outs = self.dnn(attn_out_1 - attn_out_2).squeeze(1)   # (batch_size, 1)

        return outs

