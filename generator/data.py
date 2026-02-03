import torch
import numpy as np
import os
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, RWMol, ChemicalFeatures

# --- CONFIGURAZIONE FEATURE ---
TYPE_ATOMS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'K', 'Unknown']
HYBRIDIZATION_TYPE = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, 'Misc']
NUMBER_HYDROGENS = [0, 1, 2, 3, 4]
ATOM_DEGREE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
CHIRALITY_LIST = ['R', 'S', 'None']

def one_hot_encoding(x, list_properties):
    if x not in list_properties: x = list_properties[-1]
    return [1 if s == x else 0 for s in list_properties]

def get_atom_features(atom):
    atom_feature = []
    atom_feature += one_hot_encoding(atom.GetSymbol(), TYPE_ATOMS)
    try: hyb = atom.GetHybridization()
    except: hyb = 'Misc'
    atom_feature += one_hot_encoding(hyb, HYBRIDIZATION_TYPE)
    atom_feature += one_hot_encoding(atom.GetTotalNumHs(), NUMBER_HYDROGENS)
    atom_feature += one_hot_encoding(atom.GetTotalDegree(), ATOM_DEGREE)
    atom_feature += one_hot_encoding(atom.GetFormalCharge(), FORMAL_CHARGES)
    try:
        cip = atom.GetProp('_CIPCode')
        tag = 'R' if cip == 'R' else ('S' if cip == 'S' else 'None')
    except: tag = 'None'
    atom_feature += one_hot_encoding(tag, CHIRALITY_LIST)
    atom_feature += [1 if atom.GetIsAromatic() else 0]
    atom_feature += [1 if atom.IsInRing() else 0]
    atom_feature += [atom.GetMass() * 0.01]
    atom_feature += [atom.GetNumRadicalElectrons()]
    
    features = np.array(atom_feature)
    target_dim = 56
    if len(features) < target_dim:
        features = np.concatenate([features, np.zeros(target_dim - len(features))])
    elif len(features) > target_dim:
        features = features[:target_dim]
    return torch.tensor(features, dtype=torch.float)

def mol_to_graph_data(mol, device='cpu'):
    if mol is None: return None, None
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.AssignStereochemistry(mol)
        atoms = mol.GetAtoms()
        x = [get_atom_features(atom) for atom in atoms]
        if len(x) == 0: return None, None
        x = torch.stack(x).to(device)
        rows, cols = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows.extend([start, end])
            cols.extend([end, start])
        if len(rows) == 0: 
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else: 
            edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
        return x, edge_index
    except: return None, None

class ChemicalActionSpace:
    def __init__(self):
        self.atom_types = ['C', 'O', 'N', 'F', 'Cl'] 
        self.bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]
        self.num_actions = 8 
        self.max_valence = {'C':4, 'N':3, 'O':2, 'F':1, 'Cl':1, 'S':6, 'Br':1, 'I':1}

    def _get_free_valence(self, atom):
        return self.max_valence.get(atom.GetSymbol(), 0) - atom.GetExplicitValence()

    def apply_action(self, mol, action_idx):
        try:
            rw_mol = RWMol(mol)
            rw_mol.UpdatePropertyCache(strict=False)
            atoms = list(rw_mol.GetAtoms())
            available_atoms = [a.GetIdx() for a in atoms if self._get_free_valence(a) > 0]

            if action_idx < 5: # ADD ATOM
                if not available_atoms: return None
                idx = np.random.choice(available_atoms)
                atom_sym = self.atom_types[action_idx]
                new_idx = rw_mol.AddAtom(Chem.Atom(atom_sym))
                rw_mol.AddBond(int(idx), int(new_idx), Chem.BondType.SINGLE)

            elif action_idx in [5, 6]: # ADD BOND
                if len(available_atoms) < 2: return None
                for _ in range(15):
                    i1, i2 = np.random.choice(available_atoms, 2, replace=False)
                    is_double = (action_idx == 6)
                    val_req = 2 if is_double else 1
                    a1, a2 = rw_mol.GetAtomWithIdx(int(i1)), rw_mol.GetAtomWithIdx(int(i2))
                    if self._get_free_valence(a1) >= val_req and \
                       self._get_free_valence(a2) >= val_req and \
                       not rw_mol.GetBondBetweenAtoms(int(i1), int(i2)):
                        bt = Chem.BondType.DOUBLE if is_double else Chem.BondType.SINGLE
                        rw_mol.AddBond(int(i1), int(i2), bt)
                        break
                else: return None

            elif action_idx == 7: # REMOVE BOND
                if rw_mol.GetNumBonds() == 0: return None
                bonds = list(rw_mol.GetBonds())
                valid_bonds = [b for b in bonds if not b.GetIsAromatic()]
                if not valid_bonds: return None
                b = valid_bonds[np.random.randint(0, len(valid_bonds))]
                rw_mol.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())

            if len(Chem.GetMolFrags(rw_mol)) > 1: return None
            Chem.SanitizeMol(rw_mol)
            return rw_mol.GetMol()
        except: return None

class MoleculeEnv:
    def __init__(self, start_smiles, gnn_model, max_steps=5, device='cpu'):
        self.start_smiles = start_smiles
        self.gnn_model = gnn_model
        self.max_steps = max_steps
        self.device = device
        self.action_space = ChemicalActionSpace()
        self.action_space_size = self.action_space.num_actions
        
        self.start_mol = Chem.MolFromSmiles(start_smiles)
        self.current_mol = Chem.MolFromSmiles(start_smiles)
        self.steps = 0
        
   
        Chem.GetSSSR(self.start_mol) 
        self.start_rings = self.start_mol.GetRingInfo().NumRings()
        
        self.visited_states = set()
        
        self.start_probs, self.start_is_toxic = self._get_toxicity(self.start_mol)
        
        status = "TOXIC" if self.start_is_toxic else "SAFE"
        print(f"Env Init: {status} (Active Classes: {sum(1 for p in self.start_probs if p > 0.5)}/12)")

    def _get_toxicity(self, mol):
        if not self.gnn_model or not mol: 
            return [0.0]*12, False
            
        x, edge_index = mol_to_graph_data(mol, self.device)
        if x is None: 
            return [0.0]*12, False
            
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
        self.gnn_model.eval()
        with torch.no_grad():
            dummy = torch.zeros((1, 2), dtype=torch.float, device=self.device)
            logits = self.gnn_model(x, edge_index, batch, global_features=dummy)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            
        is_toxic = np.any(probs > 0.5)
        return probs, is_toxic

    def reset(self, specific_smiles=None):
        s = specific_smiles if specific_smiles else self.start_smiles
        self.current_mol = Chem.MolFromSmiles(s)
        self.start_mol = Chem.MolFromSmiles(s)
        self.steps = 0
        
        self.start_probs, self.start_is_toxic = self._get_toxicity(self.start_mol)
        
        
        Chem.GetSSSR(self.start_mol)
        self.start_rings = self.start_mol.GetRingInfo().NumRings()
        
        self.visited_states = set()
        self.visited_states.add(Chem.MolToSmiles(self.current_mol, isomericSmiles=True))
        
        return self._get_state()

    def _get_state(self):
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(self.current_mol, 2, nBits=2048)
            return np.array(fp)
        except: return np.zeros(2048)

    def step(self, action):
        self.steps += 1
        new_mol = self.action_space.apply_action(self.current_mol, action)
        
        if new_mol is None:
            return self._get_state(), -1.0, self.steps >= self.max_steps, {'valid': False, 'smiles': ""}

        current_smiles = Chem.MolToSmiles(new_mol, isomericSmiles=True)
        if current_smiles in self.visited_states:
            return self._get_state(), -5.0, self.steps >= self.max_steps, {'valid': False, 'smiles': current_smiles}
        
        self.visited_states.add(current_smiles)
        self.current_mol = new_mol
        
       
        Chem.GetSSSR(self.current_mol) 
        current_rings = self.current_mol.GetRingInfo().NumRings() 
        
        if current_rings < self.start_rings:
            return self._get_state(), -10.0, True, {'valid': True, 'smiles': current_smiles}

        current_probs, current_is_toxic = self._get_toxicity(self.current_mol)
        
        try:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(self.start_mol, 2)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(self.current_mol, 2)
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        except: sim = 0.0

        reward = 0.0
        reward += 2.0 * sim 
        
        has_flipped = False
        if self.start_is_toxic:
            has_flipped = (not current_is_toxic)
        else:
            has_flipped = current_is_toxic

        if has_flipped:
            if sim > 0.7:
                reward += 30.0 
                done = True
            elif sim > 0.5:
                reward += 5.0
            else:
                reward -= 5.0 
        
        if sim < 0.4: reward -= 5.0

        done = (self.steps >= self.max_steps) or (has_flipped and sim > 0.7)
        
        info = {
            'valid': True, 
            'smiles': current_smiles, 
            'probs': current_probs, 
            'sim': sim, 
            'flipped': has_flipped
        }
        
        return self._get_state(), reward, done, info