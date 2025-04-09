# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:30:01 2024

@author: jspark
"""

import numpy as np
import torch

def predict(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        probs = []
        labels = []                
        test_vectors_mf = []
        test_vectors_lf = []
        for batch in dataloader:
            batch = batch.to(device)
            logit, test_vector_1, test_vector_2 = model(batch)
            test_vectors_mf.append(test_vector_1.cpu())
            test_vectors_lf.append(test_vector_2.cpu())
            probs.append(torch.sigmoid(logit.cpu()))
            labels.append(torch.sigmoid(logit.cpu())>=0.5)
    probs = np.concatenate(probs).squeeze()
    labels = np.concatenate(labels, dtype = int).squeeze()
    test_vectors_mf = np.concatenate(test_vectors_mf)
    test_vectors_lf = np.concatenate(test_vectors_lf)
    
    return probs, labels, test_vectors_mf, test_vectors_lf
    