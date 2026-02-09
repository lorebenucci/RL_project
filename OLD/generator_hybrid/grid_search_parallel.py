import itertools
import torch
import pandas as pd
import numpy as np
#import tqdm
import random
from tqdm import tqdm
import os
from main import set_seed

set_seed(42)


from joblib import Parallel, delayed
from rdkit import RDLogger
from Molecule_env_actions import MoleculeEnv
from train import train_agent
from evaluate import evaluate_model
from model import Tox21GNN
from config import *
from sklearn.model_selection import train_test_split

RDLogger.DisableLog('rdApp.*')

def run_single_experiment(params, gpu_id, experiment_id):
    """
    Questa funzione esegue un singolo training + evaluation.
    """
    
   
    
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    from main import set_seed, RANDOM_SEED
    
    set_seed(RANDOM_SEED)
   
    
    # Assegna il device (utile se volessi usare più GPU, ma qui usiamo la stessa)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Ricarica dati e modelli DENTRO la funzione (necessario per multiprocessing)
    gnn_model = Tox21GNN(num_node_features=NODE_FEATURES, hidden_channels=HIDDEN_CHANNELS, num_classes=NUM_CLASSES)
    # Assicurati che il path sia assoluto o corretto rispetto a dove lanci lo script
    checkpoint_path = '../checkpoints/best_tox21_model.pth' 
    gnn_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    gnn_model.to(device).eval()
    
    tox21_df = pd.read_csv("../datasets/tox21_processed_features.csv")
    all_smiles = tox21_df["smiles"].values
    
    #create division train and test split
    train_smiles,temp_smiles = train_test_split(all_smiles, train_size=0.7, test_size=0.3, random_state=RANDOM_SEED)
    valid_smiles, test_smiles = train_test_split(temp_smiles,test_size=0.5, random_state=RANDOM_SEED)
    #test_smiles_subset = test_smiles[:100] # Subset veloce per evaluation
    
    print(f"--> START Exp {experiment_id} | Params: {params}")
    
    # Inizializza Env
    env = MoleculeEnv(gnn_model=gnn_model, threshold=0.6, max_steps=MAX_STEPS, device=device,
        w_tox=params['w_tox'],           # Peso Gradiente
        w_flip=params['w_flip'],         # Peso Successo
        w_sim_penalty=params['w_pen'])    # Peso Penalità Similarità)
    
    # Training
    try:
        agent = train_agent(
            env, 
            train_smiles, 
            episodes=EPOCHS_AGENT, # Numero ridotto per grid search veloce
            batch_size=params["batch_size"],
            lr=params["lr"],
            gamma=params["gamma"],           # Passa gamma
            hidden_channels=params["hidden_dim"], # Passa hidden_dim
            device=device,
            path=f"checkpoint_exp_reward{experiment_id}.pth" # Salva checkpoint unico per processo
        )
        
        # Evaluation
        stats = evaluate_model(agent, env, valid_smiles, device=device, verbose=False)
        
        result = {
            'params': params,
            'strict_success': stats['success_strict'] / stats['total'],
            'loose_success': stats['success_loose'] / stats['total'],
            'avg_sim_success': np.mean(stats['success_similarity']) if stats['success_similarity'] else 0,
            'avg_sim_tanimoto': np.mean(stats['all_tanimoto']) if stats['all_tanimoto'] else 0,
            'avg_steps': np.mean(stats['success_steps']) if stats['success_steps'] else 0
        }
        print(f"<-- END Exp {experiment_id} | Strict: {result['strict_success']:.2%}")
        return result
        
    except Exception as e:
        print(f"ERROR in Exp {experiment_id}: {e}")
        return None

def run_parallel_grid_search():
    param_grid = {

   'lr': [3e-4, 5e-4, 1e-4],

   
   'gamma': [0.5, 0.90, 0.99],

    
    'batch_size': [64, 128],

   
    'hidden_dim': [256, 512, 1024],

    'w_tox': [ 100.0, 150.0, 200.0],

       
    'w_flip': [150.0, 200.0, 300.0],

       
    'w_pen': [25.0, 50.0]
    
    }
    
    #create all possible combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total Experiment Combinations: {len(combinations)}")
    
    jobs = (delayed(run_single_experiment)(params, 0, i) for i, params in enumerate(combinations))
    
    # return_as="generator" permette di avere i risultati man mano che finiscono
    results_generator = Parallel(n_jobs=6, return_as="generator")(jobs)
    
    final_results = []
    
    for res in tqdm(results_generator, total=len(combinations), desc="Grid Search Progress"):
        if res is not None:
            final_results.append(res)
    
    # 3. Analisi e Salvataggio
    df = pd.DataFrame(final_results)
    df.to_csv("reward_grid_search_parallel_results.csv", index=False)
    
    # Stampa il migliore
    best_run = df.loc[df['strict_success'].idxmax()]
    print("\n\n BEST HYPERPARAMETERS FOUND:")
    print(best_run)

if __name__ == "__main__":
    # Importante per multiprocessing su Windows/Linux
    torch.multiprocessing.set_start_method('spawn', force=True)
    run_parallel_grid_search()