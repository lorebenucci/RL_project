import torch
import numpy as np
from tqdm.auto import tqdm
from src.DGN.utils import apply_masks
from src.DGN.config import DEVICE



def compute_class_weights(loader):
    print("Calcolo dei pesi per il bilanciamento classi (pos_weight)...")
    all_labels = []
    
   
    for batch in loader:
        all_labels.append(batch.y)
    
    
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    weights = []
    for i in range(12): # Per (colonna)
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
    """Creates a learning rate scheduler with a linear warmup and cosine decay."""
    
    # Scheduler for the linear warmup phase
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    
    # Scheduler for the cosine decay phase
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(num_train_steps - warmup_steps)
    )
    
    # Chain them together
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
        out = model(batch.x, batch.edge_index, batch.edge_attr,batch.batch, batch.global_feat)
        
        
        valid_loss,is_labeled=apply_masks(batch,out,criterion)
        loss = valid_loss.sum() / is_labeled.sum()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * batch.num_graphs # Per media pesata corretta
        total_molecules += batch.num_graphs
        
    return total_loss / total_molecules