import torch
from tqdm.auto import tqdm

def get_predictions(model,device, test_loader):
    model.eval()
    all_probs = []
    all_targets = []
    
    print("Generazione previsioni sul Test Set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            
            # 1. Forward Pass
            logits = model(batch.x, batch.edge_index,batch.edge_attr, batch.batch, batch.global_feat)
            
            # 2. Application Sigmoid (Logits -> Probabilità 0-1)
            probs = torch.sigmoid(logits)
            
           
            all_probs.append(probs.cpu())
            all_targets.append(batch.y.cpu())
            
    # Concateniamo tutto in due grandi matrici
    y_probs = torch.cat(all_probs, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()
    
    return y_probs, y_true
