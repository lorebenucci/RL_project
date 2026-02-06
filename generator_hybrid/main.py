import torch
import os
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Numpy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
set_seed(42)

from rdkit import Chem, RDLogger
from Molecule_env_actions import *
from train import train_agent
from evaluate import evaluate_model
from model import Tox21GNN
from  config import *
from train import DuelingDQN
import pandas as pd
from sklearn.model_selection import train_test_split

# Disabilita i log di warning di RDKit
#RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.*')

def main():
   
    GNN_PATH = '../checkpoints/best_tox21_model.pth'
    
    
   # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {DEVICE}")

    # --- INIZIALIZZAZIONE MODELLO ---
    print("Initializing GNN Predictor...")
    gnn_model = Tox21GNN(num_node_features=NODE_FEATURES, 
                         hidden_channels=HIDDEN_CHANNELS, 
                         num_classes=NUM_CLASSES, 
                         dropout=DROPOUT,
                         num_global_features=NUM_GLOBAL_FEATURES)
    gnn_model = gnn_model.to(DEVICE)
    
    if os.path.exists(GNN_PATH):
        try:
            gnn_model.load_state_dict(torch.load(GNN_PATH, map_location=DEVICE))
            print(f"Successfully loaded GNN weights from {GNN_PATH}")
        except Exception as e:
            print(f"Error loading GNN weights: {e}")
            return
    else:
        print(f"GNN weights not found at {GNN_PATH}!")
        return

    gnn_model.eval()

    # ---LOAD CUSTOM DATASET Tox21
    tox21_df=pd.read_csv("../datasets/tox21_processed_features.csv")
    all_smiles=tox21_df["smiles"].values
    print(f"DEBUG: Caricate {len(all_smiles)} righe dal CSV.")
    
    train_smiles,test_smiles=train_test_split(
    all_smiles,
    train_size=0.7,
    test_size=0.3, 
    random_state=RANDOM_SEED
   )
    
    print(f"Totale molecole: {len(all_smiles)}")
    print(f"Train Set (per RL Training): {len(train_smiles)} molecole")
    print(f"Test Set (per Evaluation):   {len(test_smiles)} molecole")
    
    
    #start_smiles = "c1ccccc1" 

    print(f"Initializing Environment for: {train_smiles}")
    env = MoleculeEnv(gnn_model=gnn_model,threshold=0.6,max_steps=MAX_STEPS,device=DEVICE)

    path="best_experimental_agent_checkpoint_tuning_parameters.pth"
    # --- TRAINING LOOP ---
    print(f"Starting Training for {EPOCHS_AGENT} episodes...")
    # In main.py, modifica la sezione di training così:
    print(f"Starting Training for {EPOCHS_AGENT} episodes...")
    trained_agent = train_agent(
        env,
        smiles_list=train_smiles,  # <--- Corretto
        episodes=EPOCHS_AGENT,
        batch_size=BATCH_SIZE_RL_AGENT, 
        lr=LR_GENERATOR, 
        device=DEVICE,
        path=path
    )
    # --- SALVATAGGIO ---
    #save_path = "dueling_dqn_agent_multitask.pth"
    #torch.save(trained_agent.state_dict(), save_path)
   # print(f"Agent saved to {save_path}")
    
    
    if os.path.exists(path):
        print(f"Loading Best Agent from {path} for evaluation...")
        # Qui ha senso creare una nuova istanza perché vuoi caricare i pesi migliori,
        # non usare quelli dell'ultimo step che hai in memoria.
        best_agent = DuelingDQN(input_dim=LATENT_DIM, output_dim=env.action_space_size).to(DEVICE)
        best_agent.load_state_dict(torch.load(path))
        stats = evaluate_model(best_agent, env, test_smiles, device=DEVICE)
    else:
        trained_agent = DuelingDQN(input_dim=LATENT_DIM, output_dim=env.action_space_size).to(DEVICE)
        trained_agent.load_state_dict(torch.load(path))

        stats=evaluate_model(trained_agent, env, test_smiles, device=DEVICE)
    
    

if __name__ == "__main__":
    main()