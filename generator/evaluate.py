import torch
import numpy as np
from rdkit import Chem
from data import mol_to_graph_data, MoleculeEnv

def evaluate_model(agent, env, test_smiles_list, device='cpu'):
    agent.eval()
    success_count = 0
    valid_count = 0
    similarities = []
    
    print("\n--- Evaluation (Dynamic Flipping) ---")
    
    for smiles in test_smiles_list:
        state = env.reset(specific_smiles=smiles)
        
        initial_prob = env._get_gnn_prob(env.start_mol)
        start_class = 1 if initial_prob > 0.5 else 0
        target_class = 1 - start_class 
        
        done = False
        steps = 0
        final_mol = env.start_mol
        success = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent(state_tensor).argmax().item()
            
            next_state, reward, done, info = env.step(action)
            state = next_state
            steps += 1
            
            if info['valid']:
                final_mol = env.current_mol
                
        if info['valid']:
            valid_count += 1
            final_prob = info['prob_target']
            final_class = 1 if final_prob > 0.5 else 0
            
            if final_class == target_class:
                success_count += 1
                similarities.append(info['similarity'])
                print(f"Success: {smiles} ({initial_prob:.2f}) -> {info['smiles']} ({final_prob:.2f}) | Sim: {info['similarity']:.2f}")
            else:
                print(f"Failed:  {smiles} ({initial_prob:.2f}) -> {info['smiles']} ({final_prob:.2f}) | Did not flip")
        else:
             print(f"Invalid: {smiles} -> Generazione fallita")

    total = len(test_smiles_list)
    avg_sim = np.mean(similarities) if similarities else 0.0
    success_rate = (success_count / total) * 100
    
    print(f"\nValidity: {(valid_count/total)*100:.1f}%")
    print(f"True Success Rate: {success_rate:.1f}%")
    print(f"Avg Similarity: {avg_sim:.3f}")
    
    return success_rate