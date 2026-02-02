import torch
import os
import numpy as np
from data import MoleculeEnv, mol_to_graph_data
from train import train_agent
from evaluate import evaluate_model
from model import Tox21GNN

def main():
    EPOCHS = 2000
    LR = 1e-3
    TARGET_CLASS = 0
    GNN_PATH = '../checkpoints/best_tox21_model.pth'
    
    NODE_FEATURES = 56
    HIDDEN_CHANNELS = 128
    NUM_CLASSES = 12
    DROPOUT = 0.3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    print("Initializing GNN Predictor...")
    gnn_model = Tox21GNN(num_node_features=NODE_FEATURES, 
                         hidden_channels=HIDDEN_CHANNELS, 
                         num_classes=NUM_CLASSES, 
                         dropout=DROPOUT)
    gnn_model = gnn_model.to(device)

    if os.path.exists(GNN_PATH):
        try:
            gnn_model.load_state_dict(torch.load(GNN_PATH, map_location=device))
            print(f"Successfully loaded GNN weights from {GNN_PATH}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("CRITICAL: Training will run with RANDOM weights!")
    else:
        print(f"WARNING: '{GNN_PATH}' not found. Using RANDOM GNN weights.")
    
    gnn_model.eval()

    start_smiles = "CCCC" 
    
    print(f"Initializing Environment with Target Class: {TARGET_CLASS}")
    env = MoleculeEnv(start_smiles, 
                      target_class=TARGET_CLASS, 
                      gnn_model=gnn_model, 
                      max_steps=40, 
                      alpha=0.5, 
                      device=device)

    print(f"\n--- Initial Check for {start_smiles} ---")
    x, edge_index = mol_to_graph_data(env.start_mol, device)
    with torch.no_grad():
        batch_vec = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        logits = gnn_model(x, edge_index, batch_vec)
        initial_prob = torch.sigmoid(logits)[0, TARGET_CLASS].item()
    
    print(f"Initial Toxicity Probability: {initial_prob:.4f}")
    if initial_prob < 0.5:
        print("NOTE: Starting molecule is classified as NON-TOXIC.")
        print("The agent will try to keep it valid and similar.")
    else:
        print("NOTE: Starting molecule is TOXIC. Agent will try to detoxify it.")
    print("------------------------------------------\n")
    
    print(f"Starting Training for {EPOCHS} episodes...")
    trained_agent = train_agent(env, episodes=EPOCHS, lr=LR, device=device)
    
    save_path = "dueling_dqn_agent.pth"
    torch.save(trained_agent.state_dict(), save_path)
    print(f"Agent saved to {save_path}")
    
    print("Starting Evaluation on test molecules...")
    test_molecules = [
        "c1ccccc1",
        "CCO",
        "Clc1ccccc1",
        "CC1=CC=C(C=C1)O"
    ]
    
    evaluate_model(trained_agent, env, test_molecules, device=device)

if __name__ == "__main__":
    main()