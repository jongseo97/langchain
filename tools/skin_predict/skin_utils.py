# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:15:17 2024

@author: jspark
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import euclidean_distances


def merge(output_path, df_list):
    if len(df_list) == 1:
        return df_list[0]
    else:
        merged_df = df_list[0]
        for i in range(1, len(df_list)):
            merged_df = pd.merge(merged_df, df_list[i], on = 'SMILES')
        merged_df.to_excel(output_path, index=False)
        return merged_df
    
def applicability_domain_test(KE, script_path, test_vectors_mf, test_vectors_lf):   
    train_mf = np.array(pd.read_excel(f'{script_path}/domain/KE{KE}_train_mf.xlsx'))
    train_mf_dist_matrix = euclidean_distances(train_mf)
    train_mf_mean_dist = np.mean(train_mf_dist_matrix[train_mf_dist_matrix != 0])
    ad_mf = []
    for test_vector in test_vectors_mf:
        test_train_distance = euclidean_distances(test_vector.reshape(1, -1), train_mf)
        check_ad = (test_train_distance < train_mf_mean_dist).sum()
        if check_ad == 0:
            ad_mf.append('out')
        else:
            ad_mf.append('in')
    
    train_lf = np.array(pd.read_excel(f'{script_path}/domain/KE{KE}_train_lf.xlsx'))
    train_lf_dist_matrix = euclidean_distances(train_lf)
    train_lf_mean_dist = np.mean(train_lf_dist_matrix[train_lf_dist_matrix != 0])
    ad_lf = []
    for test_vector in test_vectors_lf:
        test_train_distance = euclidean_distances(test_vector.reshape(1, -1), train_lf)
        check_ad = (test_train_distance < train_lf_mean_dist).sum()
        if check_ad == 0:
            ad_lf.append('out')
        else:
            ad_lf.append('in')
            
    return ad_mf, ad_lf
    
def applicability_domain_average(KE, script_path, test_vectors_mf, test_vectors_lf):   
    train_mf = np.array(pd.read_excel(f'{script_path}/domain/KE{KE}_train_mf.xlsx'))
    train_mf_dist_matrix = euclidean_distances(train_mf)
    train_mf_mean_dist = np.mean(train_mf_dist_matrix[train_mf_dist_matrix != 0])
    ad_mf = []
    for test_vector in test_vectors_mf:
        test_train_distance = euclidean_distances(test_vector.reshape(1, -1), train_mf)
        test_mean_dist = np.mean(test_train_distance[test_train_distance != 0])
        if test_mean_dist > train_mf_mean_dist:
            ad_mf.append('out')
        else:
            ad_mf.append('in')
    
    train_lf = np.array(pd.read_excel(f'{script_path}/domain/KE{KE}_train_lf.xlsx'))
    train_lf_dist_matrix = euclidean_distances(train_lf)
    train_lf_mean_dist = np.mean(train_lf_dist_matrix[train_lf_dist_matrix != 0])
    ad_lf = []
    for test_vector in test_vectors_lf:
        test_train_distance = euclidean_distances(test_vector.reshape(1, -1), train_lf)
        test_mean_dist = np.mean(test_train_distance[test_train_distance != 0])
        if test_mean_dist > train_lf_mean_dist:
            ad_lf.append('out')
        else:
            ad_lf.append('in')
            
    return ad_mf, ad_lf
    
    
    
    
    
    
    
        
        