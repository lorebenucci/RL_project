import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm



def one_hot_encoding(x,list_properties):
    if x not in list_properties:
        x = list_properties[-1]
    return [1 if s == x else 0 for s in list_properties]
    
class Dataset_tox21(Dataset):
    
    def __init__(self,smiles,labels,global_parameters,permitted_atoms,hybrization_type,degree_atoms,number_hydrogens,permitted_bonds,stereo_list):
        
        self.smiles=smiles
        self.labels=labels
        self.permitted_atoms=permitted_atoms
        self.hybrization_type=hybrization_type
        self.degree_atoms=degree_atoms
        self.number_hydrogens=number_hydrogens
        self.permitted_bonds=permitted_bonds
        self.stereo_list=stereo_list
        
        # Chemical rules for acceptor and donators
        self.fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        self.factory = ChemicalFeatures.BuildFeatureFactory(self.fdefName)
        self.data_list = []
        scaler = StandardScaler()
        norm_global_feats = scaler.fit_transform(np.array(global_parameters))
        
        # converting all molecules
        for i in tqdm(range(len(smiles))):
            data = self.conversion_to_graph(smiles[i], labels[i], torch.tensor([norm_global_feats[i]], dtype=torch.float))
            if data is not None:
                self.data_list.append(data)
        
    
    # extract atom (node) features
    def get_atom_features(self,atom,donor_indices,acceptor_indices):
        
        atom_feature=[]
        atom_feature += one_hot_encoding(atom.GetSymbol(),self.permitted_atoms) # atomic symbol
        atom_feature += [atom.GetAtomicNum() * 0.01] # scaled atomic number
        atom_feature += one_hot_encoding(atom.GetDegree(),self.degree_atoms) # atom degree
        atom_feature += one_hot_encoding(atom.GetTotalNumHs(),self.number_hydrogens) # hydrogens
        atom_feature += one_hot_encoding(atom.GetHybridization(), self.hybrization_type)# hybridization
        atom_feature += [1 if atom.GetIsAromatic() else 0] # aromaticity
        
        atom_idx = atom.GetIdx()
        atom_feature += [1 if atom_idx in donor_indices else 0] # h-bond donator
        atom_feature += [1 if atom_idx in acceptor_indices else 0] # h-bond acceptor
        
        # chirality
        try:
            chirality = atom.GetProp('_CIPCode')
            if chirality == 'R': atom_feature += [1, 0, 0] # R
            elif chirality == 'S': atom_feature += [0, 1, 0] # S
            else: atom_feature += [0, 0, 1] # none
        except:
            atom_feature += [0, 0, 1] # Any defined (none)
        
        # formal charge (-2, -1, 0, +1, +2)   
        formal_charge = atom.GetFormalCharge()
        charges_list = [-2, -1, 0, 1, 2]
        atom_feature += one_hot_encoding(formal_charge, charges_list)
        
        return torch.tensor(atom_feature, dtype=torch.float)
        
    
    # extract bond (edge) features
    def get_bond_features(self,bond):
        
        bond_feature = []
        
        bond_type = bond.GetBondType() #extract type of bond (SINGLE, DOUBLE, TRIPLE, AROMATIC)
        bond_feature += one_hot_encoding(bond_type, self.permitted_bonds)
        bond_feature += [1 if bond.GetIsConjugated() else 0] # is Conjugated? (Boolean)
        bond_feature += [1 if bond.IsInRing() else 0] # is in ring ? (Boolean)
        stereo = bond.GetStereo() # stereochemistry
        bond_feature += one_hot_encoding(stereo, self.stereo_list)

        return torch.tensor(bond_feature, dtype=torch.float)
        
    
    
    # SMILES to graph
    def conversion_to_graph(self,smiles,labels,global_features): 
        

        # ATOMS (nodes)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        feats = self.factory.GetFeaturesForMol(mol)
        donor_indices = []
        acceptor_indices = []
        
        for feat in feats:
            if feat.GetFamily() == 'Donor':
                donor_indices.extend(feat.GetAtomIds())
            elif feat.GetFamily() == 'Acceptor':
                acceptor_indices.extend(feat.GetAtomIds())
            
        donor_indices = set(donor_indices)
        acceptor_indices = set(acceptor_indices)
        atom_features_list = []
    
        Chem.AssignStereochemistry(mol)
        for atom in mol.GetAtoms():
            atom_features_list.append(self.get_atom_features(atom, donor_indices, acceptor_indices))
        
        x = torch.stack(atom_features_list) # atom features vector
        
        # BONDS (edges) 
        edge_indices = []
        edge_features_list=[]
        for bond in mol.GetBonds():
            k = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[k, j], [j, k]]
            bond_feat = self.get_bond_features(bond) 
            edge_features_list += [bond_feat, bond_feat]
    
        if len(edge_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 11), dtype=torch.float) #13 number of bond features
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.stack(edge_features_list)

        y_tensor = None
        if labels is not None:
            y_tensor = torch.tensor(labels, dtype=torch.float).view(1, -1)
        
        
        return Data(x=x, edge_index=edge_index,edge_attr=edge_attr, y=y_tensor,global_feat=global_features)
    
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, i):
        return self.data_list[i]
