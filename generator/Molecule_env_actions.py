import torch
import numpy as np
import os
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, RWMol, ChemicalFeatures,Descriptors
from utils import *

class ChemicalActionSpace:
    def __init__(self):
        
        # We add domain of atom types depending the Toxicity level like P,S,BR,I 
        self.atom_types = ['C', 'O', 'N', 'F', 'Cl', 'S', 'P', 'Br', 'I']
        #self.bond_types = PERMITTED_BONDS[:len(PERMITTED_BONDS)-1]
        
        
        self.num_actions = len(self.atom_types) * 3 + 3
        self.max_valence = {
            'C': 4, 'N': 3, 'O': 2, 'F': 1, 'Cl': 1, 
            'S': 6, 'P': 5, 'Br': 1, 'I': 1
        }

    def _get_free_valence(self, atom):
        # we consider totalValence so  we resolve also problem of Implicits Hydrogens 
        try:
            return self.max_valence.get(atom.GetSymbol(), 0) - atom.GetTotalValence()
        except:
            return 0

    def apply_action(self, mol, action_idx):
        
        if mol is None: 
            return None
        
        try:
            #We render write and readable the Mol
            rw_mol = RWMol(mol)
            rw_mol.UpdatePropertyCache(strict=False) #update the valence
            Chem.GetSymmSSSR(rw_mol) #update map of rings
            
            atoms = list(rw_mol.GetAtoms())
            
            available_atoms = [a.GetIdx() for a in atoms if self._get_free_valence(a) > 0]
            
            num_types = len(self.atom_types)
            
            # Action [0,17]:  ADD ATOM with SINGLE BOND or DOUBLE BOND clever strategy
            if action_idx < num_types * 2: 
                
                #extract possible idx 
                is_double = action_idx >= num_types
                
                #choice of a bond + requires valence
                bond_type = Chem.BondType.DOUBLE if is_double else Chem.BondType.SINGLE
                required_valence = 2 if is_double else 1

                # Filter only possible atoms which can link with others
                available_atoms = [
                    a.GetIdx() for a in atoms 
                    if self._get_free_valence(a) >= required_valence
                ]
                
                if not available_atoms: 
                    return None # it is impossible to apply this action 
                
                available_atoms.sort()
                #extract idx randomically atoms
                idx = int(np.random.choice(available_atoms))
               
                #LOGICA CHIMICA: Preferenza Sterica
                # Ordiniamo per: 
                # 1. Grado (Degree): Preferiamo atomi meno ingombrati (es. terminali)
                # 2. Indice: Determinismo puro per parità
                #available_atoms.sort(key=lambda a: (a.GetDegree(), a.GetIdx()))
               # target_atom = available_atoms[0]
                #idx = target_atom.GetIdx()
                
                # PHISICAL ADD of new atom with a link
                atom_sym = self.atom_types[action_idx % num_types]
                new_idx = rw_mol.AddAtom(Chem.Atom(atom_sym))
                rw_mol.AddBond(int(idx), int(new_idx),bond_type)

            # ACTION [18 - 26] SUBSTITUTION ATOMS
            elif action_idx < num_types * 3: # SUBSTITUTION (Bio-isteresis)
                
                #Choice of new atom sym
                new_type_idx = action_idx % num_types
                new_sym = self.atom_types[new_type_idx]
                
                #extract atomic number + valence
                new_atomic_num = Chem.Atom(new_sym).GetAtomicNum()
                new_max_val = self.max_valence.get(new_sym, 0)
                
                # Chosen of a target substitution candidates
                target_candidates = []
                for a in atoms:
                    #Un atomo può essere sostituito solo se la sua valenza massima 
                    # è >= ai legami che ha già (ExplicitValence)
                    if a.GetIsAromatic() and new_sym in ['F', 'Cl', 'Br', 'I']:
                        continue
                    if new_max_val >= a.GetTotalValence():
                        target_candidates.append(a.GetIdx())
                
                if not target_candidates:
                    return None # Nessun atomo può essere sostituito con questo nuovo tipo
                
                #Random selection of atom 
                target_candidates.sort()
                target_idx = int(np.random.choice(target_candidates))
                target_atom = rw_mol.GetAtomWithIdx(target_idx)
                
                # LOGICA CHIMICA:
                # 1. Preferiamo sostituire Eteroatomi esistenti (N, O, S) prima dei Carboni
                #    perché cambia drasticamente la chimica.
                # 2. Poi Sterica (Degree basso)
                
                #target_candidates.sort(key=lambda a: (0 if a.GetSymbol() != 'C' else 1, a.GetDegree(), a.GetIdx()))
                #target_atom = candidates[0]
                
                #Trasformation of identity of atom
                target_atom.SetAtomicNum(new_atomic_num)
                target_atom.SetFormalCharge(0)
                
                #Permette a RDKit di ricalcolare gli H
                target_atom.SetNoImplicit(False)
                
            
            #ACTION [27, 28]: ADD BOND between existing atoms
            elif action_idx in [27,28]:
                # we require if action_idx defines single or double bond
                is_double = (action_idx == 28)
                req_val = 2 if is_double else 1
                bt = Chem.BondType.DOUBLE if is_double else Chem.BondType.SINGLE
                
                
                #find all condidates with sufficient valence request based on bond type 
                potential_atoms = [
                    a.GetIdx() for a in atoms if self._get_free_valence(a) >= req_val
                ]
                
                if len(potential_atoms) < 2: return None
                
                # generate all possible couples which don't have bonds betweeen them.
                possible_pairs_bonds= []
                
                for i, idx1 in enumerate(potential_atoms):
                    for idx2 in potential_atoms[i+1:]:
                        
                        # if we add double bond to aromtic atom->causes problems
                        if is_double:
                            if rw_mol.GetAtomWithIdx(idx1).GetIsAromatic() or rw_mol.GetAtomWithIdx(idx2).GetIsAromatic():
                                continue
                                
                                   
                        if not rw_mol.GetBondBetweenAtoms(idx1, idx2):
                            possible_pairs_bonds.append((idx1, idx2))
                
                if not possible_pairs_bonds: return None
                
                # LOGICA: Determinismo basato sugli indici (non c'è una metrica di distanza 3D qui)
                # Ordina per primo indice, poi secondo
                
                #possible_pairs_bonds.sort(key=lambda x: (x[0], x[1]))
                #i1, i2 = possible_pairs_bonds[0]
                possible_pairs_bonds.sort()
                i1, i2 = possible_pairs_bonds[np.random.randint(0, len(possible_pairs_bonds))]
                rw_mol.AddBond(int(i1), int(i2), bt)
                
            #ACTION [29]: REMOVE BOND
            elif action_idx == 29: # REMOVE BOND
                #verify if is possible to remove bond
                if rw_mol.GetNumBonds() == 0: return None
                
                #we identify possible valids bonds excluding the aromatic bonds for stability
                bonds = list(rw_mol.GetBonds())
                valid_bonds = [b for b in bonds if not b.GetIsAromatic()]
                
                if not valid_bonds: return None
                
                #we identify if it is possible to eliminate a bond but it should belong to the ring (otherwise it brokes the molecule in 2 parts)
                ring_bonds = [b for b in valid_bonds if b.IsInRing()]
                
                
                #CHOICE A CANDIDATE
                #if we have ring bond -> we give major priority to this otherwise we cut a functional group
                candidates= ring_bonds if ring_bonds else valid_bonds
                
                # LOGICA CHIMICA:
                # Preferiamo rompere legami Singoli prima dei Doppi (più facili da rompere)
                # Poi usiamo gli indici per determinismo
                #candidates.sort(key=lambda b: (
                 #   0 if b.GetBondType() == Chem.BondType.SINGLE else 1,
                 #   min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), 
                 #   max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                #))
                
                #target_bond = candidates[0]
                #remotion
                candidates.sort()
                rand_idx = np.random.randint(0, len(candidates))
                b= candidates[rand_idx]
                idx1, idx2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                
                #b = valid_bonds[np.random.randint(0, len(valid_bonds))]
                rw_mol.RemoveBond(idx1,idx2)


            #final control to avoid fragmentation
            if len(Chem.GetMolFrags(rw_mol)) > 1: return None
            
            
            #control valid chemical modifies
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            
            #KEKULIZZAZIONE FORZATA per testare la validità
            Chem.Kekulize(new_mol)
            return new_mol
        
        except: 
            return None

class MoleculeEnv:
    def __init__(self, gnn_model,threshold=0.4, max_steps=5, device='cpu'):
        
        # Initialization molecule
        self.start_smiles = None
        self.current_mol = None
        self.direction = 0
        self.gnn_model = gnn_model
        self.max_steps = max_steps
        self.device = device
        self.threshold=threshold
        
        #Action space
        self.action_space = ChemicalActionSpace() #class of chemical action space 30
        self.action_space_size = self.action_space.num_actions
        
        #take start topological structure 
        #self.start_mol = Chem.MolFromSmiles(start_smiles)
        
        
        #Chem.GetSSSR(self.start_mol) 
        #self.start_rings = self.start_mol.GetRingInfo().NumRings()
        
        #self.start_probs, self.start_is_toxic = self._get_toxicity(self.start_mol)
       # self.start_max_prob = np.max(self.start_probs)
        
        #BIDIRECTIONAL LOGIC
        # if toxic(>0.5) -> direction is down (-1)
        # if non-toxic(<0.5) -> direction is up (+1)
       
        
        self.steps = 0
        self.visited_states = set()
        
    def _get_toxicity(self, mol):
        if not self.gnn_model or not mol: 
            return [0.0]*12, False
        
        #build the graph from mol    
        x, edge_index, edge_attr = mol_to_graph_data(mol, self.device)
        if x is None: 
            return [0.0]*12, False
        
        #put in GPU
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)       
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
        
        #scaler application to normalize MW,LogP
        mw_norm,logp_norm=compute_MW_LogP(mol)
        global_feats = torch.tensor([[mw_norm, logp_norm]], dtype=torch.float, device=self.device)
        
        self.gnn_model.eval()
        with torch.no_grad():
            
            logits = self.gnn_model(x, edge_index,edge_attr,batch, global_features=global_feats)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            
        is_toxic = np.any(probs > 0.5)
        return probs, is_toxic

    def reset(self, specific_smiles):
        
        
        if specific_smiles is None:
            raise ValueError("Devi passare uno SMILES valido a env.reset()!")
        
        #Detect change of molecule
        self.start_smiles = specific_smiles
        self.start_mol = Chem.MolFromSmiles(self.start_smiles)
        self.current_mol = Chem.MolFromSmiles(self.start_smiles)
        
        #Reset contatori + visited states
        self.visited_states = set()
        self.steps = 0
        
        #recompute the rings and the restable molecule
        Chem.GetSymmSSSR(self.start_mol) 
        self.start_rings = self.start_mol.GetRingInfo().NumRings()
        
        #ricompute baseline of toxicity
        self.start_probs, self.start_is_toxic = self._get_toxicity(self.start_mol)
        
        self.start_probs, self.start_is_toxic = self._get_toxicity(self.start_mol)
        
        #COMPUTE MAX_PROB
        self.start_max_prob = np.max(self.start_probs)
        
        #implement logic of direction: we should decide if agent should poison or cure the molecule
        if self.start_is_toxic:
            self.direction=-1.0
        else:
            self.direction=1.0 
         
        #update visited states
        #target = "SAFE" if self.start_is_toxic else "TOXIC"
       # print(f"Reset: {self.start_max_prob:.2f} -> Goal: {target}")
        
        # Aggiungi stato iniziale ai visitati
        self.visited_states.add(Chem.MolToSmiles(self.current_mol, isomericSmiles=True))
        
        return self._get_state()

    def _get_state(self):
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(self.current_mol, 2, nBits=2048)
            return np.array(fp,dtype=np.float32)
        except: return np.zeros(2048,dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward=0
        # Choice action
        new_mol = self.action_space.apply_action(self.current_mol, action)
        
        #check transformation 
        if new_mol is None:
            return self._get_state(), -2.0, self.steps >= self.max_steps, {'valid': False, 'smiles': ""}
        
        #check smiles
        try: 
            Chem.SanitizeMol(new_mol)
            current_smiles = Chem.MolToSmiles(new_mol, isomericSmiles=True)
        except:
            return  self._get_state(), -3.0, self.steps >= self.max_steps, {'valid': False, 'smiles': ""}
        
        #check if it has been already visited
        if current_smiles in self.visited_states:
            return self._get_state(), -3.0, self.steps >= self.max_steps, {'valid': False, 'smiles': current_smiles}
        
        #new molecule find...
        self.visited_states.add(current_smiles)
        self.current_mol = new_mol
        
        #recompute the rings and structure of molecule
        Chem.GetSymmSSSR(self.current_mol) 
        current_rings = self.current_mol.GetRingInfo().NumRings() 
        
        
        if current_rings < self.start_rings:
            return self._get_state(), -3.0, True, {'valid': True, 'smiles': current_smiles,'flipped': False}

        
        #CHECK TANIMOTO SIMILARITY
        try:
            fp_start=AllChem.GetMorganFingerprintAsBitVect(self.start_mol, 2)
            fp_current = AllChem.GetMorganFingerprintAsBitVect(self.current_mol, 2)
            sim = DataStructs.TanimotoSimilarity(fp_start, fp_current)
        except: sim = 0.0
        
        
        if sim < self.threshold-0.2:
            return self._get_state(), -3.0, True, {
            'valid': True, 
            'smiles': current_smiles, # Aggiungi questo!
            'sim': sim, 
            'flipped': False
            }
        
        # Penalità progressiva se si allontana dalla struttura
        if sim < self.threshold:
            reward-=(self.threshold - sim) * 20.0
        #Check GNN + Reward GNN INFERENCE
        current_probs, current_is_toxic = self._get_toxicity(self.current_mol)
        
        # check possible bug of GNN No probabilities
        if current_probs is None: return self._get_state(), -1.0, True, {'valid': False}
        
        curr_max_prob = np.max(current_probs)
        
        #BIDIRECTIONAL LOGIC
        # if current_direction is -1 -> (go down): (start-curr) ->positive if curr is less
        # if current_direction is +1 ->(go up): (curr-start) -> positive if curr is major
        
        delta = (curr_max_prob - self.start_max_prob) * self.direction
        
        #COMPUTATION PROPORTIONAL REWARD based on right direction
       
        
        if delta >0:
            reward += (delta * 40.0) 
        else:
            reward += delta * 5.0 #penality if we go in opposite direction
                
        
        #REWARD is flipped 
        has_flipped = False
        
        if self.start_is_toxic:
            has_flipped = (not current_is_toxic)
        else:
            has_flipped = current_is_toxic
            
        
        if has_flipped:
            if sim >= self.threshold:
                reward += 150.0 * sim 
                done = True
            else:
                reward += 20.0
                done = True # Fermiamo comunque l'episodio
        else:
            reward-=0.5
            done = (self.steps >= self.max_steps) 
            
        #done = (self.steps >= self.max_steps) or (has_flipped and sim >=0.7)
        
        info = {
            'valid': True, 
            'smiles': current_smiles, 
            'probs': current_probs, 
            'sim': sim, 
            'flipped': has_flipped,
            'direction': "Make SAFE" if self.direction == -1 else "Make TOXIC"
        }
        
        return self._get_state(), reward, done, info
