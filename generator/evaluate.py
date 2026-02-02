import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from data import mol_to_graph_data, MoleculeEnv

def evaluate_model(agent, env, test_smiles_list, device='cpu'):
    agent.eval()
    success_count = 0
    valid_count = 0
    similarities = []
    
    print("\n--- Evaluation (Probabilistic Sampling) ---")
    
    for smiles in test_smiles_list:
        state = env.reset(specific_smiles=smiles)
        
        # Dati Iniziali
        initial_prob = env._get_gnn_prob(env.start_mol)
        start_class = 1 if initial_prob > 0.5 else 0
        target_class = 1 - start_class 
        
        done = False
        info = {'valid': False, 'smiles': smiles, 'prob_target': initial_prob, 'similarity': 1.0}
        
        print(f"\nTarget: {smiles} (Class: {start_class} -> {target_class})")
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent(state_tensor)
                # Sampling Probabilistico
                probs = F.softmax(q_values, dim=1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            
            next_state, reward, done, step_info = env.step(action)
            state = next_state
            
            if step_info['valid']:
                info = step_info # Salva ultimo stato valido
                
        # Analisi Finale
        if info['valid'] and info['smiles'] != "":
            valid_count += 1
            final_prob = info['prob_target']
            final_class = 1 if final_prob > 0.5 else 0
            
            # Check Flip
            if final_class == target_class:
                success_count += 1
                similarities.append(info['similarity'])
                print(f"✅ SUCCESS: {info['smiles']}")
                print(f"   Sim: {info['similarity']:.3f} | Prob: {initial_prob:.2f} -> {final_prob:.2f}")
            else:
                print(f"❌ FAILED: {info['smiles']}")
                print(f"   Sim: {info['similarity']:.3f} | Did not flip ({final_prob:.2f})")
        else:
             print(f"⚠️ INVALID: Nessuna molecola valida generata.")

    total = len(test_smiles_list)
    avg_sim = np.mean(similarities) if similarities else 0.0
    success_rate = (success_count / total) * 100
    
    print(f"\n================RESULT================")
    print(f"Validity: {(valid_count/total)*100:.1f}%")
    print(f"True Success Rate: {success_rate:.1f}%")
    print(f"Avg Similarity: {avg_sim:.3f}")
    
    return success_rate