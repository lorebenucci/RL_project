import torch
from src.DGN.config import DEVICE
from src.DGN.utils import apply_masks
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

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
    
   
    if isprocess:
        progressBar=tqdm(loader, desc="Testing")
    else:
        progressBar=loader
    with torch.no_grad():
        for batch in progressBar:
            batch = batch.to(DEVICE)
            
            # 1. Forward Pass
            logits = model(batch.x, batch.edge_index,batch.edge_attr, batch.batch, batch.global_feat)
            # 2. Application  Sigmoid (Logits -> Probabilities 0-1)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_targets.append(batch.y.cpu())
            
   #concat and save predictions and y_true 
    y_probs = torch.cat(all_probs, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()
    
    roc_auc_scores=[]
    for i in range(12):
    
        col_true = y_true[:, i]
        col_pred = y_probs[:, i]
        # Masks NaN
        mask = ~np.isnan(col_true)
        valid_true = col_true[mask]
        valid_pred = col_pred[mask]
    
        if len(np.unique(valid_true)) < 2:
            continue
        else:
            # Compute ROC_AUC
            score = roc_auc_score(valid_true, valid_pred)
            roc_auc_scores.append(score)
    
    
    return np.mean(roc_auc_scores) if roc_auc_scores else 0.0
    