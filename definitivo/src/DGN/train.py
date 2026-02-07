from tqdm.notebook import tqdm
from src.DGN.config import DEVICE
import torch
from src.DGN.utils import apply_masks




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