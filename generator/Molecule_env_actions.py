import torch
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, RWMol
from utils import *

class ChemicalActionSpace:
    def __init__(self):
        # Dominio atomico esteso per tossicologia
        self.atom_types = ['C', 'O', 'N', 'F', 'Cl', 'S', 'P', 'Br', 'I']
        self.num_actions = len(self.atom_types) * 3 + 3
        self.max_valence = {
            'C': 4, 'N': 3, 'O': 2, 'F': 1, 'Cl': 1, 
            'S': 6, 'P': 5, 'Br': 1, 'I': 1
        }

    def _get_free_valence(self, atom):
        try:
            return self.max_valence.get(atom.GetSymbol(), 0) - atom.GetTotalValence()
        except:
            return 0

    def apply_action(self, mol, action_idx):
        if mol is None: return None
        
        try:
            rw_mol = RWMol(mol)
            rw_mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(rw_mol)
            
            atoms = list(rw_mol.GetAtoms())
            num_types = len(self.atom_types)
            
            # AZIONI [0-17]: Aggiunta atomo (Legame Singolo o Doppio)
            if action_idx < num_types * 2: 
                is_double = action_idx >= num_types
                bond_type = Chem.BondType.DOUBLE if is_double else Chem.BondType.SINGLE
                required_valence = 2 if is_double else 1

                available_atoms = [a.GetIdx() for a in atoms if self._get_free_valence(a) >= required_valence]
                if not available_atoms: return None 
                
                idx = int(np.random.choice(available_atoms))
                atom_sym = self.atom_types[action_idx % num_types]
                new_idx = rw_mol.AddAtom(Chem.Atom(atom_sym))
                rw_mol.AddBond(int(idx), int(new_idx), bond_type)

            # AZIONI [18-26]: Sostituzione atomo (Bio-isosteria)
            elif action_idx < num_types * 3:
                new_sym = self.atom_types[action_idx % num_types]
                new_atomic_num = Chem.Atom(new_sym).GetAtomicNum()
                new_max_val = self.max_valence.get(new_sym, 0)
                
                # Filtro candidati per valenza e aromaticità
                target_candidates = [
                    a.GetIdx() for a in atoms 
                    if new_max_val >= a.GetTotalValence() and 
                    not (a.GetIsAromatic() and new_sym in ['F', 'Cl', 'Br', 'I'])
                ]
                
                if not target_candidates: return None
                
                target_idx = int(np.random.choice(target_candidates))
                target_atom = rw_mol.GetAtomWithIdx(target_idx)
                target_atom.SetAtomicNum(new_atomic_num)
                target_atom.SetFormalCharge(0)
                target_atom.SetNoImplicit(False)
            
            # AZIONI [27-28]: Aggiunta legame tra atomi esistenti
            elif action_idx in [27, 28]:
                is_double = (action_idx == 28)
                req_val = 2 if is_double else 1
                bt = Chem.BondType.DOUBLE if is_double else Chem.BondType.SINGLE
                
                potential_atoms = [a.GetIdx() for a in atoms if self._get_free_valence(a) >= req_val]
                if len(potential_atoms) < 2: return None
                
                possible_pairs = []
                for i, idx1 in enumerate(potential_atoms):
                    for idx2 in potential_atoms[i+1:]:
                        if is_double and (rw_mol.GetAtomWithIdx(idx1).GetIsAromatic() or rw_mol.GetAtomWithIdx(idx2).GetIsAromatic()):
                            continue
                        if not rw_mol.GetBondBetweenAtoms(idx1, idx2):
                            possible_pairs.append((idx1, idx2))
                
                if not possible_pairs: return None
                i1, i2 = possible_pairs[np.random.randint(0, len(possible_pairs))]
                rw_mol.AddBond(int(i1), int(i2), bt)
                
            # AZIONE [29]: Rimozione legame (Priorità ai cicli per evitare frammentazione)
            elif action_idx == 29:
                valid_bonds = [b for b in rw_mol.GetBonds() if not b.GetIsAromatic()]
                if not valid_bonds: return None
                
                ring_bonds = [b for b in valid_bonds if b.IsInRing()]
                candidates = ring_bonds if ring_bonds else valid_bonds
                
                b = candidates[np.random.randint(0, len(candidates))]
                rw_mol.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())

            # Validazione finale: controllo frammentazione e sanitizzazione
            if len(Chem.GetMolFrags(rw_mol)) > 1: return None
            
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            Chem.Kekulize(new_mol)
            return new_mol
        except: return None



class MoleculeEnv:
    def __init__(self, gnn_model, threshold=0.4, max_steps=5, device='cpu'):
        self.gnn_model = gnn_model
        self.max_steps = max_steps
        self.device = device
        self.threshold = threshold
        self.action_space = ChemicalActionSpace()
        self.action_space_size = self.action_space.num_actions

    def _get_toxicity(self, mol):
        if not self.gnn_model or not mol: return [0.0]*12, False
        
        x, edge_index, edge_attr = mol_to_graph_data(mol, self.device)
        if x is None: return [0.0]*12, False
        
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
        mw_norm, logp_norm = compute_MW_LogP(mol)
        global_feats = torch.tensor([[mw_norm, logp_norm]], dtype=torch.float, device=self.device)
        
        self.gnn_model.eval()
        with torch.no_grad():
            logits = self.gnn_model(x.to(self.device), edge_index.to(self.device), 
                                    edge_attr.to(self.device), batch, global_features=global_feats)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            
        return probs, np.any(probs > 0.5)

    def reset(self, specific_smiles):
        if specific_smiles is None: raise ValueError("SMILES richiesto!")
        
        self.start_mol = Chem.MolFromSmiles(specific_smiles)
        self.current_mol = Chem.MolFromSmiles(specific_smiles)
        self.visited_states = {Chem.MolToSmiles(self.current_mol, isomericSmiles=True)}
        self.steps = 0
        
        Chem.GetSymmSSSR(self.start_mol) 
        self.start_rings = self.start_mol.GetRingInfo().NumRings()
        self.start_probs, self.start_is_toxic = self._get_toxicity(self.start_mol)
        self.start_max_prob = np.max(self.start_probs)
        
        # Logica bidirezionale: -1 (rendi sicuro), +1 (rendi tossico)
        self.direction = -1.0 if self.start_is_toxic else 1.0
        
        return self._get_state()

    def _get_state(self):
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(self.current_mol, 2, nBits=2048)
            return np.array(fp, dtype=np.float32)
        except: return np.zeros(2048, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = 0
        new_mol = self.action_space.apply_action(self.current_mol, action)
        
        if new_mol is None:
            return self._get_state(), -2.0, self.steps >= self.max_steps, {'valid': False}
        
        try:
            current_smiles = Chem.MolToSmiles(new_mol, isomericSmiles=True)
        except:
            return self._get_state(), -3.0, self.steps >= self.max_steps, {'valid': False}
        
        if current_smiles in self.visited_states:
            return self._get_state(), -3.0, self.steps >= self.max_steps, {'valid': False}
        
        self.visited_states.add(current_smiles)
        self.current_mol = new_mol
        
        # Vincolo: non distruggere anelli esistenti
        Chem.GetSymmSSSR(self.current_mol)
        if self.current_mol.GetRingInfo().NumRings() < self.start_rings:
            return self._get_state(), -3.0, True, {'valid': True, 'flipped': False}

        # Calcolo Similarità Tanimoto
        try:
            fp_start = AllChem.GetMorganFingerprintAsBitVect(self.start_mol, 2)
            fp_curr = AllChem.GetMorganFingerprintAsBitVect(self.current_mol, 2)
            sim = DataStructs.TanimotoSimilarity(fp_start, fp_curr)
        except: sim = 0.0
        
        if sim < self.threshold - 0.2:
            return self._get_state(), -3.0, True, {'valid': True, 'sim': sim}
        
        # Penalità se troppo diverso dalla struttura originale
        if sim < self.threshold: reward -= (self.threshold - sim) * 20.0
        
        # Calcolo Reward basato sulla direzione della tossicità
        current_probs, current_is_toxic = self._get_toxicity(self.current_mol)
        curr_max_prob = np.max(current_probs)
        delta = (curr_max_prob - self.start_max_prob) * self.direction
        
        reward += (delta * 40.0) if delta > 0 else (delta * 5.0)
        
        # Verifica se l'obiettivo (flip tossicità) è raggiunto
        has_flipped = (not current_is_toxic) if self.start_is_toxic else current_is_toxic
        
        if has_flipped:
            reward += 10.0 * sim if sim >= self.threshold else 20.0
            done = True
        else:
            reward -= 0.5
            done = (self.steps >= self.max_steps)
            
        return self._get_state(), reward, done, {
            'valid': True, 'smiles': current_smiles, 'sim': sim, 'flipped': has_flipped
        }