import torch
import numpy as np
from rdkit import Chem

def evaluate_model(agent, env, test_smiles_list, device='cpu'):
    print("\n--- Evaluation (Multi-Task Flip) ---")
    
    agent.eval()
    
    for start_smiles in test_smiles_list:
        print(f"\nTarget Molecule: {start_smiles}")
        
        # Reset
        state = env.reset(specific_smiles=start_smiles)
        
        start_probs, start_toxic = env.start_probs, env.start_is_toxic
        active_indices = [i for i, p in enumerate(start_probs) if p > 0.5]
        
        status_str = "TOXIC" if start_toxic else "SAFE"
        print(f"Initial Status: {status_str} (Active Classes: {active_indices})")
        
        done = False
        trajectory = [start_smiles]
        step_count = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, done, info = env.step(action)
            
            if info['valid']:
                trajectory.append(info['smiles'])
            
            state = next_state
            step_count += 1
            if step_count >= 10: break # Limite sicurezza eval
        
        final_smiles = env.current_mol
        final_probs, final_toxic = env._get_toxicity(final_smiles)
        final_active = [i for i, p in enumerate(final_probs) if p > 0.5]
        
        print(f"Final Molecule: {Chem.MolToSmiles(final_smiles)}")
        print(f"Final Status: {'TOXIC' if final_toxic else 'SAFE'} (Active: {final_active})")
        print(f"Trajectory: {' -> '.join(trajectory)}")
        
        # Valutazione Successo
        success = False
        if start_toxic:
            success = (not final_toxic) # Volevamo Safe
        else:
            success = final_toxic # Volevamo Toxic
            
        if success:
            print("RESULT: SUCCESS (Flipped!)")
        else:
            print("RESULT: FAILURE (No Flip)")
        print("-" * 30)