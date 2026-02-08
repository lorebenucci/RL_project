# HYPERPARAMETER TUNING

from src.DGN.config import *
from src.DGN.model import Tox21GNN
from src.DGN.evaluate import compute_val_roc_auc
from torch_geometric.loader import DataLoader
import optuna
from src.DGN.train import train_one_epoch, create_lr_scheduler



def objective(trial, class_weights, train_dataset, val_dataset):
    
    param = {
        'hidden_channels': trial.suggest_categorical('hidden_channels', [32, 64, 128, 256]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'dropout': trial.suggest_float('dropout', 0.2, 0.6),
        'epochs': EPOCHS 
    }

    train_loader_tune = DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True,pin_memory=PIN_MEMORY )
    val_loader_tune = DataLoader(val_dataset, batch_size=param['batch_size'], shuffle=False,pin_memory=PIN_MEMORY)

    model = Tox21GNN(
        num_node_features=NODE_FEATURES, 
        hidden_channels=param['hidden_channels'],
        num_classes=NUM_CLASSES,
        dropout=param['dropout']
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
    
    total_steps = param['epochs'] * len(train_loader_tune)
    scheduler = create_lr_scheduler(optimizer, total_steps, warmup_steps=int(total_steps*0.2))
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction='none')
    val_roc_auc=0
    for epoch in range(param['epochs']):
        train_loss=train_one_epoch(model,train_loader=train_loader_tune,scheduler=scheduler,criterion=criterion,optimizer=optimizer,isprogress=False)
        #val_loss = run_validation_epoch(model,val_loader=val_loader_tune,criterion=criterion)
        val_roc_auc=compute_val_roc_auc(model,val_loader_tune,False)
        trial.report(val_roc_auc, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return val_roc_auc
    
