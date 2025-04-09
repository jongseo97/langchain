# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:01:13 2024

@author: jspark
"""

import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import AllChem
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class AtomFeaturizer():
    def __init__(self, allowable_sets):
        self.allowable_sets = allowable_sets
        self.properties = list(allowable_sets.keys())
        
    def encode(self, atom):
        atom_features = []
        for property in self.properties:
            key = getattr(self, property)(atom)
            atom_feature = self.one_hot_encoding(key, self.allowable_sets[property])
            atom_features += atom_feature
        return atom_features
    
    def one_hot_encoding(self, x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return [int(v) for v in list(map(lambda s: x==s, allowable_set))]
    
    def symbol(self, atom):
        return atom.GetSymbol()
    
    def degree(self, atom):
        return atom.GetDegree()
    
    def charge(self, atom):
        return atom.GetFormalCharge()
    
    def chirality(self, atom):
        return atom.GetChiralTag().name.lower()
    
    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()
    
    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()
    
    def ring(self, atom):
        return int(atom.IsInRing())
    
    def aromatic(self, atom):
        return int(atom.GetIsAromatic())
    
    def E_acceptor(self, atom):
        return int(float(atom.GetProp('_GasteigerCharge'))>0)
    

def smiles_to_graph(smiles, atom_featurizer, bond_featurizer = None):
    # Initialize graph
    atom_features = []
    bond_features = []
    edge_index = []
    mf = []

    # Smiles to Molecule
    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    AllChem.ComputeGasteigerCharges(mol)
    
    # Molecule to Graph
    # unique_bonds = []
    for atom in mol.GetAtoms():
        atom_feature = atom_featurizer.encode(atom)
        atom_features.append(atom_feature)
        
        for neighbor in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            # if bond_name in unique_bonds:
            #     continue
            edge_index.append([atom.GetIdx(), neighbor.GetIdx()])
            if bond_featurizer is not None:
                bond_feature = bond_featurizer.encode(bond)
                bond_features.append(bond_feature)
            # unique_bonds.append(bond_name)
            
    atom_features = torch.Tensor(atom_features)
    bond_features = torch.Tensor(bond_features)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    generator = MakeGenerator(('rdkit2dnormalized',))
    mf = generator.process(smiles)
    mf = torch.Tensor(mf)
    # return atom_features, bond_features, edge_index
    data = Data(x=atom_features, edge_index=edge_index.t().contiguous(), mf = mf)
    
    return data



