# DDQN UTILS

import torch
import numpy as np
import os
import sys
import random
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures,Descriptors
from  src.DDQN.config import TYPE_ATOMS, HYBRIDIZATION_TYPE, NUMBER_HYDROGENS, ATOM_DEGREE, PERMITTED_BONDS, STEREO_LIST, FORMAL_CHARGES, MEAN_MW, STD_MW, MEAN_LOGP, STD_LOGP

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

#current_dir = os.path.dirname(os.path.abspath(__file__))
#project_root = os.path.dirname(current_dir)
#sys.path.append(project_root)


fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

#Function to compute MW_LogP
def compute_MW_LogP(mol):
    Mw= Descriptors.MolWt(mol) if mol else None

    LogP=Descriptors.MolLogP(mol) if mol else None
    Mw_norm=(Mw-MEAN_MW)/STD_MW
    LogP_norm=(LogP-MEAN_LOGP)/STD_LOGP
    return Mw_norm,LogP_norm


def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(x == item) for item in permitted_list]
    return binary_encoding

#Function to compute 
    
def mol_to_graph_data(mol, device='cpu'):
    if mol is None: return None, None, None
    
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.AssignStereochemistry(mol)
        
        # --- LOGICA H-BOND ---
        feats = factory.GetFeaturesForMol(mol)
        donor_indices = set()
        acceptor_indices = set()
        for feat in feats:
            if feat.GetFamily() == 'Donor':
                donor_indices.update(feat.GetAtomIds())
            elif feat.GetFamily() == 'Acceptor':
                acceptor_indices.update(feat.GetAtomIds())
        
        # --- ATOM FEATURES (88 DIM) ---
        atom_features_list = []
        for atom in mol.GetAtoms():
            feature = []
            # Atom Type
            feature += one_hot_encoding(atom.GetSymbol(), TYPE_ATOMS)
            # Atomic Number
            feature += [atom.GetAtomicNum() * 0.01]
            # Degree
            feature += one_hot_encoding(atom.GetDegree(), ATOM_DEGREE)
            # Hydrogens
            feature += one_hot_encoding(atom.GetTotalNumHs(), NUMBER_HYDROGENS)
            # Hybridization
            feature += one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPE)
            # Aromatic
            feature += [1 if atom.GetIsAromatic() else 0]
            # Donor/Acceptor
            idx = atom.GetIdx()
            feature += [1 if idx in donor_indices else 0]
            feature += [1 if idx in acceptor_indices else 0]
            # Chirality
            try:
                chirality = atom.GetProp('_CIPCode')
                if chirality == 'R': feature += [1, 0, 0]
                elif chirality == 'S': feature += [0, 1, 0]
                else: feature += [0, 0, 1]
            except: feature += [0, 0, 1]
            # Formal Charge
            feature += one_hot_encoding(atom.GetFormalCharge(), FORMAL_CHARGES)
            
            atom_features_list.append(torch.tensor(feature, dtype=torch.float))
        
        if not atom_features_list:
            return None, None, None
        x = torch.stack(atom_features_list).to(device)
        
        # --- EDGE FEATURES ---
        rows, cols, edge_feats = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows.extend([start, end])
            cols.extend([end, start])
            
            b_feat = [] 
            b_feat += one_hot_encoding(bond.GetBondType(),PERMITTED_BONDS)
            b_feat += [1 if bond.GetIsConjugated() else 0]
            b_feat += [1 if bond.IsInRing() else 0]
            b_feat += one_hot_encoding(bond.GetStereo(), STEREO_LIST)
            
            edge_feats.extend([torch.tensor(b_feat, dtype=torch.float)] * 2)
            
        if len(rows) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, 11), dtype=torch.float, device=device)
        else:
            edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
            edge_attr = torch.stack(edge_feats).to(device)
        
        return x, edge_index, edge_attr

    except Exception as e:
        print(f"Error in conversion: {e}")
        return None, None, None
    

