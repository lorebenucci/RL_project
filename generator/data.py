import torch
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, RWMol

# --- 1. CONFIGURAZIONE FEATURE ---
def atom_feature(atom):
    allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 
                     'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 
                     'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 
                     'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 
                     'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    
    if atom.GetSymbol() in allowable_set:
        atom_type_enc = [int(atom.GetSymbol() == s) for s in allowable_set] + [0]
    else:
        atom_type_enc = [0] * len(allowable_set) + [1]
    
    degree_enc = [int(atom.GetTotalDegree() == i) for i in range(11)]
    
    misc_enc = [
        int(atom.GetIsAromatic()),
        int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP),
        int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2),
        int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons()
    ]
    
    features = atom_type_enc + degree_enc + misc_enc
    
    target_dim = 56
    if len(features) < target_dim:
        features += [0] * (target_dim - len(features))
    elif len(features) > target_dim:
        features = features[:target_dim]
        
    return np.array(features)

def mol_to_graph_data(mol, device='cpu'):
    if mol is None: return None, None
    try:
        mol.UpdatePropertyCache(strict=False)
        atoms = mol.GetAtoms()
        x = [atom_feature(atom) for atom in atoms]
        if len(x) == 0: return None, None
        x = torch.tensor(np.array(x), dtype=torch.float, device=device)
        rows, cols = [], []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            rows.extend([start, end])
            cols.extend([end, start])
        if len(rows) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
        return x, edge_index
    except:
        return None, None

# --- 2. ACTION SPACE INTELLIGENTE ---
class ChemicalActionSpace:
    def __init__(self):
        self.atom_types = ['O', 'N', 'F', 'Cl', 'C'] 
        self.bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]
        self.num_actions = 8 
        # Valenze massime standard
        self.max_valence = {'C': 4, 'N': 3, 'O': 2, 'F': 1, 'Cl': 1, 'S': 6}

    def _get_available_atoms(self, mol):
        available = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            max_v = self.max_valence.get(symbol, 4)
            # Controllo Valenza Esplicita (ignora H impliciti)
            current_v = atom.GetExplicitValence()
            if current_v < max_v:
                available.append(atom.GetIdx())
        return available

    def apply_action(self, mol, action_idx):
        try:
            rw_mol = RWMol(mol)
            rw_mol.UpdatePropertyCache(strict=False)
            
            available_indices = self._get_available_atoms(rw_mol)

            # ADD ATOM (0-4)
            if action_idx < 5: 
                if not available_indices: return None
                target_idx = np.random.choice(available_indices)
                new_atom_symbol = self.atom_types[action_idx]
                new_idx = rw_mol.AddAtom(Chem.Atom(new_atom_symbol))
                rw_mol.AddBond(int(target_idx), int(new_idx), Chem.BondType.SINGLE)
                
            # ADD BOND (5-6)
            elif action_idx == 5 or action_idx == 6: 
                if len(available_indices) < 2: return None
                for _ in range(20):
                    idx1, idx2 = np.random.choice(available_indices, 2, replace=False)
                    if not rw_mol.GetBondBetweenAtoms(int(idx1), int(idx2)):
                        # Controllo extra per doppi legami
                        if action_idx == 6: 
                            atom1 = rw_mol.GetAtomWithIdx(int(idx1))
                            atom2 = rw_mol.GetAtomWithIdx(int(idx2))
                            if (atom1.GetExplicitValence() + 1 >= self.max_valence.get(atom1.GetSymbol(), 4)) or \
                               (atom2.GetExplicitValence() + 1 >= self.max_valence.get(atom2.GetSymbol(), 4)):
                                continue
                        b_type = self.bond_types[action_idx - 5]
                        rw_mol.AddBond(int(idx1), int(idx2), b_type)
                        break
                else: return None 
                
            # REMOVE BOND (7)
            elif action_idx == 7: 
                if rw_mol.GetNumBonds() == 0: return None
                bonds = list(rw_mol.GetBonds())
                # Proteggi anelli aromatici
                valid_bonds = [b for b in bonds if not b.GetIsAromatic()]
                if not valid_bonds: return None
                target_bond = valid_bonds[np.random.randint(0, len(valid_bonds))]
                rw_mol.RemoveBond(target_bond.GetBeginAtomIdx(), target_bond.GetEndAtomIdx())

            # Anti-Cheat: No frammenti
            if len(Chem.GetMolFrags(rw_mol)) > 1: return None

            try:
                Chem.SanitizeMol(rw_mol)
                return rw_mol.GetMol()
            except Exception:
                return None 
            
        except Exception:
            return None

# --- 3. ENVIRONMENT RIGOROSO ---
class MoleculeEnv:
    def __init__(self, start_smiles, target_class, gnn_model, max_steps=20, alpha=0.3, device='cpu'):
        self.start_smiles_str = start_smiles
        self.target_class_idx = target_class
        self.gnn_model = gnn_model
        self.max_steps = max_steps
        self.alpha = alpha 
        self.device = device
        
        self.action_handler = ChemicalActionSpace()
        self.action_space_size = self.action_handler.num_actions
        
        self.start_mol = Chem.MolFromSmiles(start_smiles)
        self.current_mol = Chem.MolFromSmiles(start_smiles)
        self.steps = 0
        self.goal_is_toxicity = False
        
        if self.start_mol:
            initial_prob = self._get_gnn_prob(self.start_mol)
            self.goal_is_toxicity = (initial_prob < 0.5)

    def _get_gnn_prob(self, mol):
        if self.gnn_model is None or mol is None: return 0.5
        x, edge_index = mol_to_graph_data(mol, self.device)
        if x is None: return 0.5
        batch_vec = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
        self.gnn_model.eval()
        with torch.no_grad():
            logits = self.gnn_model(x, edge_index, batch_vec)
            probs = torch.sigmoid(logits)
            return probs[0, self.target_class_idx].item()

    def reset(self, specific_smiles=None):
        smiles = specific_smiles if specific_smiles else self.start_smiles_str
        self.start_mol = Chem.MolFromSmiles(smiles)
        self.current_mol = Chem.MolFromSmiles(smiles)
        self.steps = 0
        if self.start_mol:
            initial_prob = self._get_gnn_prob(self.start_mol)
            self.goal_is_toxicity = (initial_prob < 0.5)
        return self._get_state()

    def _get_state(self):
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(self.current_mol, 2, nBits=2048)
            return np.array(fp)
        except:
            return np.zeros(2048)

    def step(self, action_idx):
        self.steps += 1
        
        # 1. APPLICAZIONE DIRETTA (NESSUN SALVAGENTE RANDOM)
        new_mol = self.action_handler.apply_action(self.current_mol, action_idx)
        
        valid = False
        prob_target = 0.5 
        
        if new_mol is not None:
            self.current_mol = new_mol
            valid = True
            prob_target = self._get_gnn_prob(self.current_mol)
        
        similarity = self._calculate_similarity(self.start_mol, self.current_mol)
        
        # 2. REWARD SYSTEM
        if not valid:
            reward = -0.5 # Penalità leggera per incoraggiare riprova
        else:
            if self.goal_is_toxicity:
                reward = (0.4 * prob_target) + (0.6 * similarity)
            else:
                reward = (0.4 * (1.0 - prob_target)) + (0.6 * similarity)

            # BONUS VITTORIA MASSICCIO
            flipped = (self.goal_is_toxicity and prob_target > 0.5) or \
                      (not self.goal_is_toxicity and prob_target < 0.5)
            
            if flipped:
                reward += 20.0 
                if similarity > 0.4: reward += 5.0
                if similarity > 0.6: reward += 5.0

        # Stop se vittoria di qualità
        success = valid and flipped and similarity > 0.25
        done = (self.steps >= self.max_steps) or success
        
        info = {
            'valid': valid,
            'similarity': similarity,
            'prob_target': prob_target,
            'smiles': Chem.MolToSmiles(self.current_mol) if valid else ""
        }
        
        return self._get_state(), reward, done, info

    def _calculate_similarity(self, mol1, mol2):
        try:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 0.0