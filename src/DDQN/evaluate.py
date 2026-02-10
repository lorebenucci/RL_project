import torch
import numpy as np
from rdkit import Chem


def evaluate_model(agent, env, test_smiles_list, device='cpu', attempts=10):
    print(f"\n--- Starting Evaluation ---")
    
    agent.eval()
    
    stats = {
        'total': 0,
        'success_strict': 0, # flip + Hybrid Sim >= 0.6
        'success_loose': 0,  # only Flip
        
        'all_similarity': [], 
        'all_tanimoto': [],
        'all_steps': [],

        'success_similarity': [],
        'success_tanimoto': [],
        'success_steps': [],
        
        'success_flip_fromtox_to_notox': [],
        'success_flip_from_notox_totox': []
    }
    for start_smiles in test_smiles_list:
        stats['total'] += 1
        
        best_run_data = None
        best_priority = -1 
        
        for attempt in range(attempts):
            state = env.reset(specific_smiles=start_smiles)
            start_probs, start_toxic = env.start_probs, env.start_is_toxic
            
            done = False
            step_count = 0
            final_sim_hybrid = 0.0 
            final_sim_tanimoto = 0.0
            list_fromtox_notox=[]
            list_fromnotox_tox=[]
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = agent(state_tensor)
                    action = q_values.argmax().item()
                
                next_state, reward, done, info = env.step(action)
                
                if info['valid']:
                    final_sim_hybrid = info.get('sim', 0.0)      
                    final_sim_tanimoto = info.get('tanimoto', 0.0)
                
                state = next_state
                step_count += 1
                if step_count >= env.max_steps: done = True 
            
            
            final_smiles = env.current_mol
            
            if final_smiles:
                final_probs, final_toxic, _ = env._get_toxicity(final_smiles)
                has_flipped = (not final_toxic) if start_toxic else (final_toxic)
                is_similar = final_sim_hybrid >= 0.6 # paper threshold
                
                current_priority = 0
                if has_flipped:
                    current_priority = 1
                    
                    if is_similar:
                        current_priority = 2
                        
                        if start_toxic:
                                 
                           list_fromtox_notox.append((start_smiles,Chem.MolToSmiles(final_smiles),final_sim_hybrid))
                            
                        else:
                            
                           list_fromnotox_tox.append((start_smiles,Chem.MolToSmiles(final_smiles),final_sim_hybrid))
                 
                            
                run_data = {
                    'sim': final_sim_hybrid,
                    'tanimoto': final_sim_tanimoto,
                    'steps': step_count,
                    'flipped': has_flipped,
                    'strict': (has_flipped and is_similar),
                    'priority': current_priority,
                    'list_fromtox_notox':list_fromtox_notox,
                    'list_fromnotox_tox':list_fromnotox_tox
                }
                
                if best_run_data is None or current_priority > best_priority:
                    best_priority = current_priority
                    best_run_data = run_data
                    
                
                if best_priority == 2: break
            else:
                if best_run_data is None:
                    best_run_data = {'sim': 0.0, 'tanimoto': 0.0, 'steps': step_count, 'flipped': False, 'strict': False, 'priority': 0}

        
        if best_run_data:
            # global metrics
            stats['all_similarity'].append(best_run_data['sim'])
            stats['all_tanimoto'].append(best_run_data['tanimoto'])
            stats['all_steps'].append(best_run_data['steps'])

            # only if flipped
            if best_run_data['flipped']:

                stats['success_loose'] += 1
                stats['success_similarity'].append(best_run_data['sim'])
                stats['success_tanimoto'].append(best_run_data['tanimoto'])
                stats['success_steps'].append(best_run_data['steps'])
                stats['success_flip_fromtox_to_notox'].extend(best_run_data['list_fromtox_notox'])
                stats['success_flip_from_notox_totox'].extend(best_run_data['list_fromnotox_tox'])
                
                if best_run_data['strict']:
                    stats['success_strict'] += 1
            
    
    accuracy_strict = stats['success_strict'] / stats['total'] if stats['total'] > 0 else 0
    accuracy_loose = stats['success_loose'] / stats['total'] if stats['total'] > 0 else 0
    
    # global means
    mean_sim_all = np.mean(stats['all_similarity']) if stats['all_similarity'] else 0
    mean_tani_all = np.mean(stats['all_tanimoto']) if stats['all_tanimoto'] else 0
    
    # valid counterfactuals means
    mean_sim_success = np.mean(stats['success_similarity']) if stats['success_similarity'] else 0
    mean_tani_success = np.mean(stats['success_tanimoto']) if stats['success_tanimoto'] else 0
    mean_steps_success = np.mean(stats['success_steps']) if stats['success_steps'] else 0
       
    print("\n" + "="*60)
    print(f"Final Evaluation ({stats['total']} mols, {attempts} attempts/mol):")
    print(f"Strict Success Rate (Flip + Sim>=0.6):  {accuracy_strict:.2%}")
    print(f"Loose Success Rate (Any Flip):          {accuracy_loose:.2%}")
    print("-" * 60)
    print("On Valid counter-factuals:")
    print(f"  - Avg Tanimoto (on Success):          {mean_tani_success:.3f}")
    print(f"  - Avg Hybrid Sim (on Success):        {mean_sim_success:.3f}")
    print(f"  - Avg Steps (on Success):             {mean_steps_success:.1f}")
    print("-" * 60)
    print("Including failures:")
    print(f"  - Avg Tanimoto (All):                 {mean_tani_all:.3f}")
    print(f"  - Avg Hybrid Sim (All):               {mean_sim_all:.3f}")
    print("="*60 + "\n")
    
    return stats