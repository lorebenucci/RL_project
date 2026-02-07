import os
import random
import numpy as np
import torch
from src.DGN.config import RANDOM_SEED


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Numpy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed(RANDOM_SEED)



def apply_masks(batch,out,criterion):
    
    y = batch.y
    #detect NaN (masks)
        
    # Creiamo una maschera booleana: True se è un numero, False se è NaN
    is_labeled = ~torch.isnan(y)
    y = torch.nan_to_num(y, nan=0.0)
        
    loss_matrix = criterion(out, y)
        
    # 4. Application of mask 
    valid_loss = loss_matrix * is_labeled.float()
    return valid_loss,is_labeled