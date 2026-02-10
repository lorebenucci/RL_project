import torch
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, RWMol
from src.DDQN.utils import compute_MW_LogP, mol_to_graph_data
from src.DDQN.config import LATENT_DIM, W_TOX, W_FLIP, W_PEN

class ChemicalActionSpace:
    def __init__(self):
        self.atom_types = ['C', 'O', 'N', 'F', 'Cl', 'S', 'P', 'Br', 'I']
        self.num_actions = len(self.atom_types) * 3 + 4
        self.max_valence = {'C': 4, 'N': 3, 'O': 2, 'F': 1, 'Cl': 1, 'S': 6, 'P': 5, 'Br': 1, 'I': 1}

    def _get_free_valence(self, atom):
        try:
            return self.max_valence.get(atom.GetSymbol(), 0) - atom.GetTotalValence()
        except: return 0

    def apply_action(self, mol, action_idx):
        if mol is None: return None
        try:
            rw_mol = RWMol(mol)
            rw_mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(rw_mol)
            atoms = list(rw_mol.GetAtoms())
            num_types = len(self.atom_types)

            # actions [0-17]: add atom (9 types x 2 valences)
            if action_idx < num_types * 2:
                is_double = action_idx >= num_types
                bt = Chem.BondType.DOUBLE if is_double else Chem.BondType.SINGLE
                req_val = 2 if is_double else 1
                available = [a.GetIdx() for a in atoms if self._get_free_valence(a) >= req_val]
                if not available: return None
                idx = int(np.random.choice(available))
                atom_sym = self.atom_types[action_idx % num_types]
                new_idx = rw_mol.AddAtom(Chem.Atom(atom_sym))
                rw_mol.AddBond(int(idx), int(new_idx), bt)

            # action [18-26]: modify atom type
            elif action_idx < num_types * 3:
                new_sym = self.atom_types[action_idx % num_types]
                new_max_val = self.max_valence.get(new_sym, 0)
                candidates = [a.GetIdx() for a in atoms if new_max_val >= a.GetTotalValence() 
                              and not (a.GetIsAromatic() and new_sym in ['F', 'Cl', 'Br', 'I'])]
                if not candidates: return None
                target_atom = rw_mol.GetAtomWithIdx(int(np.random.choice(candidates)))
                target_atom.SetAtomicNum(Chem.Atom(new_sym).GetAtomicNum())
                target_atom.SetFormalCharge(0)
                target_atom.SetNoImplicit(False)

            # action [27-28]: add bond (single/double)
            elif action_idx in [27, 28]:
                is_double = (action_idx == 28)
                req_val = 2 if is_double else 1
                bt = Chem.BondType.DOUBLE if is_double else Chem.BondType.SINGLE
                potential = [a.GetIdx() for a in atoms if self._get_free_valence(a) >= req_val]
                if len(potential) < 2: return None
                pairs = [(p1, p2) for i, p1 in enumerate(potential) for p2 in potential[i+1:] 
                         if not rw_mol.GetBondBetweenAtoms(p1, p2)]
                if not pairs: return None
                i1, i2 = pairs[np.random.randint(0, len(pairs))]
                rw_mol.AddBond(int(i1), int(i2), bt)

            # action [29]: remove bond
            elif action_idx == 29:
                valid_bonds = [b for b in rw_mol.GetBonds() if not b.GetIsAromatic()]
                if not valid_bonds: return None
                candidates = [b for b in valid_bonds if b.IsInRing()] or valid_bonds
                b = candidates[np.random.randint(0, len(candidates))]
                rw_mol.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())

            if len(Chem.GetMolFrags(rw_mol)) > 1: return None
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            return new_mol
        except: return None

class MoleculeEnv:
    
    def __init__(self, gnn_model, threshold=0.6, max_steps=20, device='cpu',w_tox=W_TOX,w_flip=W_FLIP,w_sim_penalty=W_PEN):
       
        self.gnn_model = gnn_model
        self.max_steps = max_steps
        self.device = device
        self.threshold = threshold
        self.action_space = ChemicalActionSpace()
        self.action_space_size = self.action_space.num_actions
        self.w_tox=w_tox
        self.w_flip=w_flip
        self.w_sim_penalty=w_sim_penalty


    def _get_toxicity(self, mol):

        if not mol: return [0.0]*12, False, None
        
        x, edge_index, edge_attr = mol_to_graph_data(mol, self.device)
        
        if x is None: return [0.0]*12, False, None
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
        mw_n, logp_n = compute_MW_LogP(mol)
        gf = torch.tensor([[mw_n, logp_n]], dtype=torch.float, device=self.device)
        self.gnn_model.eval()
        
        with torch.no_grad():
            logits, emb = self.gnn_model(x, edge_index, edge_attr, batch, global_features=gf)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        return probs, np.any(probs > 0.5), emb.cpu()



    def reset(self, specific_smiles):
        
        self.start_mol = Chem.MolFromSmiles(specific_smiles)
        self.current_mol = Chem.MolFromSmiles(specific_smiles)
        self.visited_states = {Chem.MolToSmiles(self.current_mol, isomericSmiles=True)}
        self.steps, self.previous_sim = 0, 1.0
        
        Chem.GetSymmSSSR(self.start_mol)
        self.start_rings = self.start_mol.GetRingInfo().NumRings()
        self.start_probs, self.start_is_toxic, self.start_embedding = self._get_toxicity(self.start_mol)
        self.start_max_prob = np.max(self.start_probs)
        self.direction = -1.0 if self.start_is_toxic else 1.0
        
        return self._get_state_from_embedding(self.start_embedding)


    def _get_state_from_embedding(self, embedding):
        
        if embedding is None: 
            return np.zeros(LATENT_DIM, dtype=np.float32) 
            
        return embedding.detach().numpy().flatten().astype(np.float32)

    def _get_state(self):
        _, _, emb = self._get_toxicity(self.current_mol)
        return self._get_state_from_embedding(emb)

    def step(self, action):
        self.steps += 1
        if self.steps <= 2:
            reward = -0.5
        elif self.steps <= 5:
            reward = -1.5
        else:
            reward = -3.0
            
        new_mol = self.action_space.apply_action(self.current_mol, action)
        
        if new_mol is None: 
            return self._get_state_from_embedding(None), -2.0, self.steps >= self.max_steps, {'valid': False}
        
        try:
            current_smiles = Chem.MolToSmiles(new_mol, isomericSmiles=True)
            
            # if already visited
            if current_smiles in self.visited_states: 
                return self._get_state_from_embedding(None), -2.0, self.steps >= self.max_steps, {'valid': False}
            
            self.visited_states.add(current_smiles)
            self.current_mol = new_mol

            # similarities
            fp_s = AllChem.GetMorganFingerprintAsBitVect(self.start_mol, 1)
            fp_c = AllChem.GetMorganFingerprintAsBitVect(self.current_mol, 1)
            tanimoto = DataStructs.TanimotoSimilarity(fp_s, fp_c)
            curr_probs, curr_toxic, curr_emb = self._get_toxicity(self.current_mol)
            cosine = torch.nn.functional.cosine_similarity(self.start_embedding, curr_emb).item()
            hybrid_sim = 0.7 * tanimoto + 0.3  * cosine 

            info = {
                'valid': True,
                'smiles': current_smiles,
                'sim': hybrid_sim,
                'tanimoto': tanimoto,
                'flipped': False,
                'direction': "Make SAFE" if self.direction == -1 else "Make TOXIC"
            }

            # broken molecule
            if self.current_mol.GetRingInfo().NumRings() < self.start_rings:
                return self._get_state_from_embedding(None), -5.0, True, info

            # very low similarity
            if hybrid_sim < self.threshold - 0.15: 
                return self._get_state_from_embedding(curr_emb), -5.0, True, info

            # soft penalty
            if hybrid_sim < self.threshold:
                reward -= (self.threshold - hybrid_sim) * self.w_sim_penalty
            
            # right direction
            delta_tox = (np.max(curr_probs) - self.start_max_prob) * self.direction
            reward += (delta_tox * self.w_tox) if delta_tox > 0 else (delta_tox * 0.0)

            # if flipped
            has_flipped = (not curr_toxic) if self.start_is_toxic else curr_toxic
            done = False
            if has_flipped:
                
                if hybrid_sim >= self.threshold:
                    reward += (self.w_flip * (hybrid_sim ** 2)) + 60.0 
                    steps_saved = self.max_steps - self.steps
                    reward += steps_saved * 2.0
                else: 
                    reward +=  5.0
                
                
                info['flipped'] = True
                done = True
            else: 
                done = (self.steps >= self.max_steps)

            return self._get_state_from_embedding(curr_emb), reward, done, info

        except Exception as e:
            return self._get_state_from_embedding(None), -3.0, self.steps >= self.max_steps, {'valid': False}