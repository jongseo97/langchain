# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:00:46 2024

@author: jspark
"""

import json

import torch
from .skin_model import GCN

def load_config(model_path, KE):
    with open(f'{model_path}/KE{KE}_model.config', 'r') as f:
        config = json.load(f)
    return config

def load_model(model_path, config, KE):
    config = config['model_config']
    
    #in_channels, conv_channels, hidden_channels, n_conv, n_MLP, readout, dropout, drop_rate, use_mf, n_mf, use_bn, parallel    
    model = GCN(**config)
    model.load_state_dict(torch.load(f'{model_path}/KE{KE}.pth', weights_only = True))
    
    return model
