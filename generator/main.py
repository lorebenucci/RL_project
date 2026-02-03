import torch
import os
import numpy as np
from rdkit import Chem
from data import MoleculeEnv
from train import train_agent
from evaluate import evaluate_model
from model import Tox21GNN

def main():
    # --- CONFIGURAZIONE ---
    EPOCHS = 2000
    LR = 5e-4
    GNN_PATH = '../checkpoints/best_tox21_model.pth'
    
    # Parametri GNN
    NODE_FEATURES = 56
    HIDDEN_CHANNELS = 128
    NUM_CLASSES = 12 # Tox21 ha 12 classi
    DROPOUT = 0.3
    NUM_GLOBAL_FEATURES = 2 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # --- INIZIALIZZAZIONE MODELLO ---
    print("Initializing GNN Predictor...")
    gnn_model = Tox21GNN(num_node_features=NODE_FEATURES, 
                         hidden_channels=HIDDEN_CHANNELS, 
                         num_classes=NUM_CLASSES, 
                         dropout=DROPOUT,
                         num_global_features=NUM_GLOBAL_FEATURES)
    gnn_model = gnn_model.to(device)
    
    if os.path.exists(GNN_PATH):
        try:
            gnn_model.load_state_dict(torch.load(GNN_PATH, map_location=device))
            print(f"Successfully loaded GNN weights from {GNN_PATH}")
        except Exception as e:
            print(f"Error loading GNN weights: {e}")
            return
    else:
        print(f"GNN weights not found at {GNN_PATH}!")
        return

    gnn_model.eval()

    # --- DEFINIZIONE MOLECOLA TARGET ---
    start_smiles = "c1ccccc1" 
    
    print(f"Initializing Environment for: {start_smiles}")
    env = MoleculeEnv(start_smiles, 
                      gnn_model=gnn_model, 
                      max_steps=5, 
                      device=device)

    # --- TRAINING LOOP ---
    print(f"Starting Training for {EPOCHS} episodes...")
    trained_agent = train_agent(env, episodes=EPOCHS, lr=LR, device=device)

    # --- SALVATAGGIO ---
    save_path = "dueling_dqn_agent_multitask.pth"
    torch.save(trained_agent.state_dict(), save_path)
    print(f"Agent saved to {save_path}")
    
    from train import DuelingDQN
    trained_agent = DuelingDQN(input_dim=2048, output_dim=env.action_space_size).to(device)
    trained_agent.load_state_dict(torch.load(save_path))

    # --- VALUTAZIONE ---
    test_molecules = ["c1ccccc1", "CCO", "Clc1ccccc1", "CC1=CC=C(C=C1)O"]
    evaluate_model(trained_agent, env, test_molecules, device=device)

if __name__ == "__main__":
    main()