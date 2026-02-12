import torch
import numpy as np
from rdkit import Chem

def evaluate_model(agent, env, test_smiles_list, device='cpu',verbose=True):
    #print("\n--- Evaluation (Multi-Task Flip) ---")
    if verbose: print("\n--- Evaluation (Multi-Task Flip & Similarity) ---")
    
    agent.eval()
    
    #define statistics to evaluate best flip+similarity
    stats = {
        'total': 0,
        'success_strict': 0, # Flip + Similarity > 0.4
        'success_loose': 0,  # Only flip
        'avg_similarity': [],
        'avg_steps': [],
        
        #liste di miglior success flippati con highest Hybrid similarity
        'success_flip_fromtox_to_notox': [],
        'success_flip_from_notox_totox': []
    }
    
 
    
    for start_smiles in test_smiles_list:
        
        #count how many molecule we are analyzing
        stats['total'] += 1
        #print(f"\nTarget Molecule: {start_smiles}")
        
        # Reset Environment
        state = env.reset(specific_smiles=start_smiles)
        
        # Initial date
        start_probs, start_toxic = env.start_probs, env.start_is_toxic
        active_indices = [i for i, p in enumerate(start_probs) if p > 0.5]
        
        status_str = "TOXIC" if start_toxic else "SAFE"
        #print(f"Initial Status: {status_str} (Active Classes: {active_indices})")
        
        done = False
        trajectory = [start_smiles]
        step_count = 0
        final_sim = 1.0 # Default se non fa nulla
            
           
        list_fromtox_notox=[]
        list_fromnotox_tox=[]
        while not done:
            
            #Policy Greedy (Argmax)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent(state_tensor)
                action = q_values.argmax().item()
            
            #step in env 
            next_state, reward, done, info = env.step(action)
            
            if info['valid']:
                trajectory.append(info['smiles'])
                #save similarity
                final_sim = info.get('sim', 0.0)
            
            state = next_state
            step_count += 1
            
            if step_count >= env.max_steps: done=True # Limite sicurezza eval
        
        # FINAL ANALYSIS
        final_smiles = env.current_mol
        
        if final_smiles:
            
            #recompute the final toxicitity
            final_probs, final_toxic = env._get_toxicity(final_smiles)
           # final_active = [i for i, p in enumerate(final_probs) if p > 0.5]

            #primary aim (flip)
            has_flipped = (not final_toxic) if start_toxic else (final_toxic)
            
            #simililarity constraint
            is_similar = final_sim >= 0.5
            
            #Update statistics
            if has_flipped:
                stats['success_loose']+=1
                if is_similar:
                    stats['success_strict'] += 1
                    
                    if start_toxic:
                        list_fromtox_notox.append((start_smiles,Chem.MolToSmiles(final_smiles),final_sim))
                    else:
                        list_fromnotox_tox.append((start_smiles,Chem.MolToSmiles(final_smiles),final_sim))
                 
            stats['avg_similarity'].append(final_sim)
            stats['avg_steps'].append(step_count)
            
            
            #stats success... 
            stats['success_flip_fromtox_to_notox'].extend(list_fromtox_notox)
            stats['success_flip_from_notox_totox'].extend(list_fromnotox_tox)
            
            if verbose:
                
                #output dettagliato per debug
                res_type = "FAILURE"
                if has_flipped and is_similar: res_type = "SUCCESS (Strict)"
                elif has_flipped: res_type = "PARTIAL (Flip but Low Sim)"
                
                #print(f"Result: {res_type}")
                #print(f"Final Sim: {final_sim:.2f} | Steps: {step_count}")
                #print(f"Trajectory: {' -> '.join(trajectory)}")
                    
        else:
            if verbose: print("Result: INVALID MOLECULE")    
        
    # Final report
    accuracy_strict = stats['success_strict'] / stats['total'] if stats['total'] > 0 else 0
    accuracy_loose = stats['success_loose'] / stats['total'] if stats['total'] > 0 else 0
    mean_sim = np.mean(stats['avg_similarity']) if stats['avg_similarity'] else 0
        
        #print(f"Final Molecule: {Chem.MolToSmiles(final_smiles)}")
        #print(f"Final Status: {'TOXIC' if final_toxic else 'SAFE'} (Active: {final_active})")
        #print(f"Trajectory: {' -> '.join(trajectory)}")
        
        # Valutazione Successo
       # success = False
       # if start_toxic:
       #     success = (not final_toxic) # Volevamo Safe
       # else:
       #     success = final_toxic # Volevamo Toxic
            
        #if success:
        #    print("RESULT: SUCCESS (Flipped!)")
        #else:
       #     print("RESULT: FAILURE (No Flip)")
       
    print("\n" + "="*40)
    print(f"SUMMARY EVALUATION ({stats['total']} mols):")
    print(f"Strict Success Rate (Flip + Sim>=0.4): {accuracy_strict:.2%}")
    print(f"Loose Success Rate (Any Flip):         {accuracy_loose:.2%}")
    print(f"Average Similarity:                    {mean_sim:.3f}")
    print("="*40 + "\n")
    
    return stats 
        
    

    