import torch
from src.DGN.config import DEVICE
from src.DGN.utils import apply_masks
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score


def run_validation_epoch(model,val_loader,criterion):
    
    model.eval()
    running_loss = 0.0
    total_molecules = 0
    with torch.no_grad():
        for batch in val_loader:
            batch=batch = batch.to(DEVICE)
            
            out=model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.global_feat)
            loss,is_labeled=apply_masks(batch,out,criterion)
            
            loss = loss.sum() / is_labeled.sum()
            running_loss += loss.item() * batch.num_graphs
            total_molecules += batch.num_graphs
            
    average_loss = running_loss / total_molecules
    return average_loss







def compute_val_roc_auc(model,loader,isprocess=False):
    model.eval()
    all_probs = []
    all_targets = []
    
   # print("Generazione previsioni sul Test Set...")
    if isprocess:
        progressBar=tqdm(loader, desc="Testing")
    else:
        progressBar=loader
    with torch.no_grad():
        for batch in progressBar:
            batch = batch.to(DEVICE)
            
            # 1. Forward Pass
            logits = model(batch.x, batch.edge_index,batch.edge_attr, batch.batch, batch.global_feat)
            
            # 2. Applicazione Sigmoide (Logits -> Probabilità 0-1)
            probs = torch.sigmoid(logits)
            
            # 3. Spostiamo su CPU e salviamo
            all_probs.append(probs.cpu())
            all_targets.append(batch.y.cpu())
            
    # Concateniamo tutto in due grandi matrici
    y_probs = torch.cat(all_probs, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()
    
    roc_auc_scores=[]
    for i in range(12):
    # Estraiamo la colonna i-esima
        col_true = y_true[:, i]
        col_pred = y_probs[:, i]
    
    # MASCHERA PER I NaN
    # Prendiamo solo gli indici dove abbiamo una etichetta reale
        mask = ~np.isnan(col_true)
        valid_true = col_true[mask]
        valid_pred = col_pred[mask]
    
    # Controllo di sicurezza: Servono almeno una classe positiva e una negativa
    # altrimenti ROC-AUC crasha
        if len(np.unique(valid_true)) < 2:
            print(f"{name:<20} | {'N/A':<10} | {len(valid_true)} (Solo una classe)")
            continue
        else:
            # Calcolo Score
            score = roc_auc_score(valid_true, valid_pred)
            roc_auc_scores.append(score)
    
    
    return np.mean(roc_auc_scores) if roc_auc_scores else 0.0
    