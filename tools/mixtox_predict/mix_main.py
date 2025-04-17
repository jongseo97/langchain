# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:20:36 2023

@author: user
"""

import pandas as pd
import numpy as np

from .mix_preprocess import make_binary_mixtures
from .mix_model import load_classification_model, load_regression_model, load_MDR_model, make_prediction, predict_module


encoding = 'utf-8-sig'


def mix_main(smiles_list, ratio_list):
    script_path = f'./tools/mixtox_predict'
    model_path = f'{script_path}/saved_models'

    # make_binary_mixtures로 2종조합 생성 후 아래 경로에 저장
    # 상대적 비율과 절대적 비율 두 버전 저장
    binary_mixtures = make_binary_mixtures(smiles_list, ratio_list, 'rel')

    endpoints = ['ER', 'AR', 'THR', 'NPC', 'EB', 'SG', 'GnRH']
    for endpoint in endpoints:
        print(endpoint)
        binary_mixtures = predict_module(binary_mixtures, model_path, endpoint)
    
    return binary_mixtures
    """
    return_dict = {'summary': "'Predicted' toxicity for binary combinations on mixture.",
                   'endpoint_description':{
                       "ER":"Estrogen Receptor agonist",
                       "AR":"Androgen Receptor antagonist",
                       "THR":"Thyroid Hormone Receptor agonist",
                       "NPC":"Neural Progenitor Cell antagnoist",
                       "EB":"Embryoic Body antagnoist",
                       "SG":"Steroidogenesis agonist",
                       "GnRH":"Gonadotropin Releasing Hormone antagonist"
                   },
                   }
    len_col = len(binary_mixtures.columns)
    mixture_profiles = []
    for i, row in binary_mixtures.iterrows():
        i_profile = {"smiles_A" : row.iloc[1], "smiles_B" : row.iloc[2], "ratio_A" : row.iloc[3], "ratio_B" : row.iloc[4]}
        for row_i in range(5, len_col):
            i_profile[binary_mixtures.columns[row_i]] = row.iloc[row_i]
        mixture_profiles.append(i_profile)
    
    return_dict['mixture_profile'] = mixture_profiles

    return return_dict    
    """
    
    
if __name__ == 'main':
    smiles_list = ['CN(C)C(=S)SSC(=S)N(C)C', 'C1=CC=C(C=C1)C2=CC(=O)C3=CC=CC=C3O2', 'C1C(CC2=CC=CC=C2C1C3=C(C4=CC=CC=C4OC3=O)O)C5=CC=C(C=C5)C6=CC=CC=C6', 'CC1=CC2=C(C(=C(C(=C2C(C)C)O)O)C=O)C(=C1C3=C(C4=C(C=C3C)C(=C(C(=C4C=O)O)O)C(C)C)O)O']
    ratio_list = [0.3,0.1,0.15,0.45]
    mix_main(smiles_list, ratio_list)