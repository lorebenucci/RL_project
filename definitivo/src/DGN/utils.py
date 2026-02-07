import torch

def apply_masks(batch,out,criterion):
    
    y = batch.y
    #Gestione dei NaN (La Maschera)
        
    # Creiamo una maschera booleana: True se è un numero, False se è NaN
    is_labeled = ~torch.isnan(y)
    y = torch.nan_to_num(y, nan=0.0)
        
    loss_matrix = criterion(out, y)
        
    # 4. Applicazione Maschera
    # Azzeriamo la loss dove il dato originale era NaN
    valid_loss = loss_matrix * is_labeled.float()
    return valid_loss,is_labeled