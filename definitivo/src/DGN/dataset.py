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
    # Se l'elemento 'x' non è nella lista consentita, usiamo l'ultimo elemento (spesso 'Unknown')
    if x not in list_properties:
        x = list_properties[-1]
    
    # Crea una lista: metti 1 se l'elemento corrisponde a x, altrimenti 0
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
        #definition Chemical rules for acceptor and donators
        self.fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        self.factory = ChemicalFeatures.BuildFeatureFactory(self.fdefName)
        
        
        self.data_list = []
        
        scaler = StandardScaler()
        norm_global_feats = scaler.fit_transform(np.array(global_parameters))
        
        # Process from SMILES->GRAPH
        for i in tqdm(range(len(smiles))):
            data = self.conversion_to_graph(smiles[i], labels[i], torch.tensor([norm_global_feats[i]], dtype=torch.float))
            if data is not None:
                self.data_list.append(data)
        
    
    def get_atom_features(self,atom,donor_indices,acceptor_indices):
        
        atom_feature=[]
        # 1. Atom Type (One-Hot) 
        atom_feature += one_hot_encoding(atom.GetSymbol(),self.permitted_atoms)
        
        # 2 Atomic number(Scalar) ---
        # Lo scaliamo diviso 100 per mantenere i valori piccoli (tra 0 e 1 circa) per la rete neurale
        atom_feature += [atom.GetAtomicNum() * 0.01] 
        
        #
        # 3. atom degree
        atom_feature += one_hot_encoding(atom.GetDegree(),self.degree_atoms)
        
        #
        # 4. Hydrogens
        atom_feature += one_hot_encoding(atom.GetTotalNumHs(),self.number_hydrogens)
        
        # 5. Hybridization
        atom_feature += one_hot_encoding(atom.GetHybridization(), self.hybrization_type)
        
        # 6. Aromatic
        atom_feature += [1 if atom.GetIsAromatic() else 0]
        
        # 7 H-Bond Donor
        atom_idx = atom.GetIdx()
        atom_feature += [1 if atom_idx in donor_indices else 0]

        # 8 H-Bond Acceptor (Booleano) ---
        atom_feature += [1 if atom_idx in acceptor_indices else 0]
        
        # 9. Chirality (Tag R/S/None)
        try:
            chirality = atom.GetProp('_CIPCode')
            if chirality == 'R':
                atom_feature += [1, 0, 0] # R
            elif chirality == 'S':
                atom_feature += [0, 1, 0] # S
            else:
                atom_feature += [0, 0, 1] # None/Other
        except:
            atom_feature += [0, 0, 1] # Nessuna chiralità definita
        
        #formal charge (-2,-1,0,+1,+2)   
        formal_charge = atom.GetFormalCharge()
        charges_list = [-2, -1, 0, 1, 2]
        atom_feature += one_hot_encoding(formal_charge, charges_list)
        
        return torch.tensor(atom_feature, dtype=torch.float)
        
    
    def get_bond_features(self,bond):
        
        bond_feature = []
        
        #extract type of bond
        bond_type = bond.GetBondType()
        bond_feature += one_hot_encoding(bond_type, self.permitted_bonds)
    
        # 2. Is Conjugated (Boolean)
        bond_feature += [1 if bond.GetIsConjugated() else 0]
    
        # 3. Is In Ring (Boolean)
        bond_feature += [1 if bond.IsInRing() else 0]
    
        # 4. Stereochimica (Optional: None, E, Z, etc.)
        stereo = bond.GetStereo()
    
        bond_feature += one_hot_encoding(stereo, self.stereo_list)

        return torch.tensor(bond_feature, dtype=torch.float)
        
    def conversion_to_graph(self,smiles,labels,global_features): 
        
        #transformation from smile to graph representation
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        #Compute Donators and Acceptors
        feats = self.factory.GetFeaturesForMol(mol)
        donor_indices = []
        acceptor_indices = []
        
        for feat in feats:
            #selection of feats which belong to Donor or Acceptor
            if feat.GetFamily() == 'Donor':
                donor_indices.extend(feat.GetAtomIds())
            elif feat.GetFamily() == 'Acceptor':
                acceptor_indices.extend(feat.GetAtomIds())
            
        donor_indices = set(donor_indices)
        acceptor_indices = set(acceptor_indices)
        
        
        atom_features_list = []
        #Assing chirality
        Chem.AssignStereochemistry(mol)
        for atom in mol.GetAtoms():
            #obtain for each atom==node a set of features
            atom_features_list.append(self.get_atom_features(atom, donor_indices, acceptor_indices))
        
        #stack in x all features
        x = torch.stack(atom_features_list)
        
        #Computation of arcs 
        edge_indices = []
        edge_features_list=[]
        for bond in mol.GetBonds():
            k = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[k, j], [j, k]]
            
            #bond features
            bond_feat = self.get_bond_features(bond) 
            edge_features_list += [bond_feat, bond_feat]
    
        if len(edge_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 11), dtype=torch.float) #13 number of bond features
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.stack(edge_features_list)

        #labels 
        y_tensor = None
        if labels is not None:
            y_tensor = torch.tensor(labels, dtype=torch.float).view(1, -1)
        
        
        return Data(x=x, edge_index=edge_index,edge_attr=edge_attr, y=y_tensor,global_feat=global_features)
    
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, i):
        return self.data_list[i]
