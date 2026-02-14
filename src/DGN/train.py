# TRAINING

import torch
import numpy as np
from tqdm.auto import tqdm
from src.DGN.utils import apply_masks
from src.DGN.config import DEVICE, EPOCHS, MODEL_PATH
from src.DGN.evaluate import run_validation_epoch, compute_val_roc_auc


def compute_class_weights(loader):

    all_labels = []
    
    for batch in loader:
        all_labels.append(batch.y)
    
    all_labels = torch.cat(all_labels, dim=0).numpy()
    weights = []

    for i in range(12): 
        col = all_labels[:, i]
    
        valid_indices = ~np.isnan(col)
        valid_labels = col[valid_indices]
        
        n_pos = np.sum(valid_labels == 1)
        n_neg = np.sum(valid_labels == 0)
        
        if n_pos > 0:
            weight = n_neg / n_pos
        else:
            weight = 1.0 
            
        weights.append(weight)
        
    return torch.tensor(weights, dtype=torch.float)




def create_lr_scheduler(optimizer, num_train_steps, warmup_steps):

    # warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    
    # cosine decay
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(num_train_steps - warmup_steps)
    )
    
    # chaining together
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps]
    )
    return lr_scheduler


def train_one_epoch(model,train_loader,scheduler,criterion,optimizer,isprogress):
    
    model.train()
    total_loss = 0
    total_molecules = 0
    
    if isprogress:
        progress_bar = tqdm(train_loader, desc="Training")
    else:
        progress_bar=train_loader
        
    for batch in progress_bar:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out,_ = model(batch.x, batch.edge_index, batch.edge_attr,batch.batch, batch.global_feat)
        
        
        
        valid_loss,is_labeled=apply_masks(batch,out,criterion)
        loss = valid_loss.sum() / is_labeled.sum()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * batch.num_graphs 
        total_molecules += batch.num_graphs
        
    return total_loss / total_molecules


def train(model, train_loader, val_loader,scheduler, criterion, optimizer):
    
    best_val_auc = 0.0
    save_path = MODEL_PATH

    for epoch in range(EPOCHS):
        train_loss=train_one_epoch(model,train_loader,scheduler,criterion,optimizer,True)
        val_loss = run_validation_epoch(model,val_loader,criterion)
        
        current_val_auc = compute_val_roc_auc(model, val_loader,True)
        
        print(f"Epoch {epoch + 1}/{EPOCHS} | Training Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")
        
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            torch.save(model.state_dict(), save_path)
            print(f"Best saved at epoch: {epoch+1}")
 