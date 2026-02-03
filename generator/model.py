import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool

class GINMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(GINMLP, self).__init__()
        self.lin1 = nn.Linear(in_channels, out_channels*2)
        self.bn1 = nn.BatchNorm1d(out_channels*2)
        self.lin2 = nn.Linear(out_channels*2, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU() 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = self.lin1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.lin2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.dropout(out)
        return out

class Tox21GNN(nn.Module):
    def __init__(self, num_node_features=56, hidden_channels=128, num_classes=12, dropout=0.3, num_global_features=2):
        super(Tox21GNN, self).__init__()
        
        self.conv1 = GINConv(GINMLP(num_node_features, hidden_channels, dropout))
        self.conv2 = GINConv(GINMLP(hidden_channels, hidden_channels, dropout))
        self.conv3 = GINConv(GINMLP(hidden_channels, hidden_channels, dropout))
        self.conv4 = GINConv(GINMLP(hidden_channels, hidden_channels, dropout))
        self.conv5 = GINConv(GINMLP(hidden_channels, hidden_channels, dropout))
        
        # Classifier
        # Input: Hidden * 2 (Pooling Add+Max) + Global Features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + num_global_features, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(hidden_channels, hidden_channels // 2), 
            nn.BatchNorm1d(hidden_channels // 2),
            nn.SiLU(),
            nn.Dropout(p=dropout-0.1),
            
            # Output layer: 12 logit (uno per ogni classe di tossicità)
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
    def forward(self, x, edge_index, batch, global_features):
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        
        x_combined = x1 + x2 + x3 + x4 + x5
        
        x_add = global_add_pool(x_combined, batch)
        x_max = global_max_pool(x_combined, batch)
        x_pool = torch.cat([x_add, x_max], dim=1)
        
        # Concatena le features globali 
        out = torch.cat([x_pool, global_features], dim=1)
        
        # Logits per le 12 classi
        return self.classifier(out)