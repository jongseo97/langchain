# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:57:29 2024

@author: jspark
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool


######## GCN model ########

class GCN(torch.nn.Module):
    def __init__(self, in_channels, conv_channels, hidden_channels, n_conv, n_MLP, readout, dropout, drop_rate, use_mf, n_mf, use_bn, parallel):
        super().__init__()
        
        self.dropout = dropout
        self.drop_rate = drop_rate
        self.use_mf = use_mf
        self.use_bn = use_bn
        
        # GCN setting
        GCN_list = []
        for i in range(n_conv):
            if i==0:
                GCN_list.append(GCNConv(in_channels, conv_channels))
            else:
                GCN_list.append(GCNConv(conv_channels, conv_channels))
        self.GCN_list = nn.ModuleList(GCN_list)

        # pooling method setting
        pooling = {'max' : global_max_pool, 'mean' : global_mean_pool, 'add' : global_add_pool}
        self.pooling = pooling[readout]
        
        if self.use_mf:
            conv_channels += n_mf
        
        # MLP setting
        MLP_list = []
        bn_list = []
        for i in range(n_MLP):
            if i==0:
                MLP_list.append(nn.Linear(conv_channels, hidden_channels))
            else:
                if parallel:
                    hidden_out_channels = hidden_channels
                else:
                    hidden_out_channels = int(hidden_channels/2)
                MLP_list.append(nn.Linear(hidden_channels, hidden_out_channels))
                hidden_channels = hidden_out_channels
            bn_list.append(nn.BatchNorm1d(hidden_channels))
            
        self.MLP_list = nn.ModuleList(MLP_list)
        self.bn_list = nn.ModuleList(bn_list)
        
        if n_MLP ==0:
            hidden_channels = conv_channels
        self.fc = nn.Linear(hidden_channels, 1)  
    
    def forward(self, data):
        x, edge_index, mf, batch = data.x, data.edge_index, data.mf, data.batch

        for layer in self.GCN_list:
            x = layer(x, edge_index).relu()
            if self.dropout:
                x = F.dropout(x, p=self.drop_rate, training = self.training)      
        
        x = self.pooling(x, batch)
                
        if self.use_mf:
            mf = mf.view([x.shape[0], -1])
            x = torch.cat([x, mf], dim=1)
            retval_1 = x
        
        for bn, layer in zip(self.bn_list, self.MLP_list):
            x = layer(x)
            if self.use_bn:
                x = bn(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=self.drop_rate, training = self.training) 
        retval_2 = x      
        x = self.fc(x)
        
        return x, retval_1, retval_2