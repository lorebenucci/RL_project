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
    # 2. IMPORTA E IMPOSTA IL SEED (Fondamentale per la riproducibilità)
    from main import set_seed, RANDOM_SEED
    # Usiamo lo stesso seed per tutti gli exp per confrontare i parametri alla pari
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
    env = MoleculeEnv(gnn_model=gnn_model, threshold=0.6, max_steps=MAX_STEPS, device=device)
    
    # Training
    # NOTA: Assicurati che train_agent accetti 'gamma', 'hidden_dim', 'lr' come argomenti!
    try:
        agent = train_agent(
            env, 
            train_smiles, 
            episodes=EPOCHS_AGENT, # Numero ridotto per grid search veloce
            batch_size=params['batch_size'],
            lr=params['lr'],
            gamma=params['gamma'],           # Passa gamma
            hidden_channels=params['hidden_dim'], # Passa hidden_dim
            device=device,
            path=f"checkpoint_exp_{experiment_id}.pth" # Salva checkpoint unico per processo
        )
        
        # Evaluation
        stats = evaluate_model(agent, env, valid_smiles, device=device, verbose=False)
        
        result = {
            'params': params,
            'strict_success': stats['success_strict'] / stats['total'],
            'loose_success': stats['success_loose'] / stats['total'],
            'avg_sim_success': np.mean(stats['success_similarity']) if stats['success_similarity'] else 0,
            'avg_steps': np.mean(stats['success_steps']) if stats['success_steps'] else 0
        }
        print(f"<-- END Exp {experiment_id} | Strict: {result['strict_success']:.2%}")
        return result
        
    except Exception as e:
        print(f"ERROR in Exp {experiment_id}: {e}")
        return None

def run_parallel_grid_search():
    # 1. Griglia (Esempio ridotto per testare il parallelismo)
    param_grid = {
    # LEARNING RATE
    # 1e-3: Aggressivo. Utile se l'agente è bloccato in minimi locali.
    # 1e-4: Il tuo standard.
    # 1e-5: Conservativo. Utile se vedi che il training è instabile (reward oscilla troppo).
    'lr': [1e-3,3e-4, 5e-4, 1e-4, 5e-5],

    # GAMMA (Il parametro più critico per te!)
    # 0.0 / 0.1: "Greedy Myopic". L'agente se ne frega del futuro. Ottimo per task a 1 step.
    # 0.5: Una via di mezzo (guarda 2 step avanti).
    # 0.9 / 0.99: Standard RL. Guarda all'infinito. Potrebbe introdurre rumore inutile se max_steps=1.
    'gamma': [0.1, 0.5, 0.90, 0.99],

    # BATCH SIZE
    # 64: Standard.
    # 128: Più stabile.
    # 256: Molto stabile. Utile con il Prioritized Replay per avere gradienti meno rumorosi.
    'batch_size': [64, 128, 256],

    # HIDDEN DIM (Capacità del cervello)
    # 256: Più leggero. Potrebbe generalizzare meglio (meno overfitting su pochi dati).
    # 512: Bilanciato.
    # 1024: Potente, ma rischia di memorizzare il train set invece di capire la chimica.
    'hidden_dim': [256, 512, 1024, 2048]
    }
    
    #create all possible combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total Experiment Combinations: {len(combinations)}")
    
    
    # 2. ESECUZIONE PARALLELA
    # n_jobs=4 o 6 è conservativo. Con la tua CPU puoi provare anche n_jobs=8
    # Non mettere n_jobs=-1 (tutti i core) subito, potresti finire la VRAM se i modelli sono grossi.
    jobs = (delayed(run_single_experiment)(params, 0, i) for i, params in enumerate(combinations))
    
    # return_as="generator" permette di avere i risultati man mano che finiscono
    results_generator = Parallel(n_jobs=6, return_as="generator")(jobs)
    
    final_results = []
    
    for res in tqdm(results_generator, total=len(combinations), desc="Grid Search Progress"):
        if res is not None:
            final_results.append(res)
    
    # 3. Analisi e Salvataggio
    df = pd.DataFrame(final_results)
    df.to_csv("grid_search_parallel_results.csv", index=False)
    
    # Stampa il migliore
    best_run = df.loc[df['strict_success'].idxmax()]
    print("\n\n BEST HYPERPARAMETERS FOUND:")
    print(best_run)

if __name__ == "__main__":
    # Importante per multiprocessing su Windows/Linux
    torch.multiprocessing.set_start_method('spawn', force=True)
    run_parallel_grid_search()