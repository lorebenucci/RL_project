# DGN UTILS
import os
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def apply_masks(batch,out,criterion):
    
    y = batch.y
    is_labeled = ~torch.isnan(y)
    y = torch.nan_to_num(y, nan=0.0)
    loss_matrix = criterion(out, y)
    valid_loss = loss_matrix * is_labeled.float()
    return valid_loss,is_labeled