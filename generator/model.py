import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

class GINMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(GINMLP, self).__init__()
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.lin2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU() 
        self.dropout = nn.Dropout(dropout)
        self.has_residual = (in_channels == out_channels)
        
    def forward(self, x):
        identity = x 
        out = self.lin1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.lin2(out)
        out = self.bn2(out)
        out = self.act(out)
        if self.has_residual:
            return out + identity
        else:
            return out

def gin_mlp(input_dim, output_dim):
    return GINMLP(input_dim, output_dim)

class Tox21GNN(nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_classes=12, dropout=0.3):
        super(Tox21GNN, self).__init__()
        
        self.conv1 = GINConv(gin_mlp(num_node_features, hidden_channels))
        self.conv2 = GINConv(gin_mlp(hidden_channels, hidden_channels))
        self.conv3 = GINConv(gin_mlp(hidden_channels, hidden_channels))
        self.conv4 = GINConv(gin_mlp(hidden_channels, hidden_channels))
        self.conv5 = GINConv(gin_mlp(hidden_channels, hidden_channels))
        
        self.classifier = nn.Sequential( 
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, hidden_channels // 2), 
            nn.BatchNorm1d(hidden_channels // 2),
            nn.SiLU(),
            nn.Dropout(p=dropout-0.1),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        
        # Global Pooling
        x = global_add_pool(x, batch)  
        logits = self.classifier(x)
        return logits