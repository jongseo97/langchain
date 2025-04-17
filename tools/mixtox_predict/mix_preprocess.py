# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:45:21 2024

@author: user
"""

import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from torch.utils.data import DataLoader

from .mix_GCN_models import GCNDataset

# .mol file path to SMILES
def MolFile_to_SMILES(file_path):
    mol = Chem.MolFromMolFile(file_path)
    smi = Chem.MolToSmiles(mol)
    return smi


# mixture dataframe to binary mixture dataframes (absolute fraction, relative fraction)
# frac_type = 'abs' or 'rel'
def make_binary_mixtures(smiles_list, ratio_list, frac_type):
    
    n_cpds = len(smiles_list)
    idx = 0
    binary_mixtures = pd.DataFrame()    

    for i in range(n_cpds):
        # i : compound A
        for j in range(i+1, n_cpds):
            # j : compound B 
            
            # SMILES of A, B
            i_smi = smiles_list[i]
            j_smi = smiles_list[j]
            
            # fraction of A, B
            i_frac = ratio_list[i]
            j_frac = ratio_list[j]
            
            # make a row
            if frac_type == 'rel':
                i_frac, j_frac = i_frac/(i_frac+j_frac), j_frac/(i_frac+j_frac)
                
            row = pd.DataFrame([(idx, i_smi, j_smi, i_frac, j_frac)], columns=['no.', 'SMILES_A', 'SMILES_B', 'frac_A', 'frac_B'])
            
            # concat
            binary_mixtures = pd.concat([binary_mixtures, row])
            
            idx += 1
    
    return binary_mixtures




def preprocessing(dt, model_path, endpoint, predict_type):

    # SMILES to rdkit molecular feature
    def smiles_to_mf(smi_list):
        # rdkit mol to rdkit molecular feature
        def get_mol_features(mol):
            bugs = ['MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge','MinAbsPartialCharge','BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW' 
                    ,'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'Ipc','SPS','AvgIpc','NumAmideBonds', 'NumAtomStereoCenters', 'NumBridgeheadAtoms', 'NumHeterocycles','NumSpiroAtoms', 'NumUnspecifiedAtomStereoCenters', 'Phi'] # returns nan
            mf = []
            for nm, fn in Descriptors._descList:
                if nm in bugs:
                    continue
                mf.append(fn(mol))
            return mf

        mf_list = []
        for smi in smi_list:
            mol = Chem.MolFromSmiles(smi)
            mol_feature = get_mol_features(mol)
            mf_list.append(mol_feature)
        return mf_list    

    # smiles 바탕으로 molecular feature 호출
    smi_A = list(dt['SMILES_A'])
    smi_B = list(dt['SMILES_B'])
    ratio_A = list(dt['frac_A'])
    ratio_B = list(dt['frac_B'])
    
    #smiles로 molecular feature 불러옴
    mf_A = smiles_to_mf(smi_A)
    mf_B = smiles_to_mf(smi_B)


    # scaler 호출
    #model_path = '20240124_models/Classification'
    #model_path = '20240124_models/MDR'
    path_A = f'{model_path}/{endpoint}/{predict_type}/scaler_A.sav'
    path_B = f'{model_path}/{endpoint}/{predict_type}/scaler_B.sav'
    
    scaler_A = pickle.load(open(path_A, 'rb'))
    scaler_B = pickle.load(open(path_B, 'rb'))
    
    #molecular feature에 scaling 적용
    mf_A = scaler_A.transform(mf_A)
    mf_B = scaler_B.transform(mf_B)
    
    #데이터셋화 (그래프 input 생성-adj matrix랑 feature matrix)
    my_dataset = GCNDataset(256,smi_A,smi_B,ratio_A,ratio_B,mf_A,mf_B)
    my_dataloader = DataLoader(my_dataset, batch_size = dt.shape[0])
    
    return my_dataloader




