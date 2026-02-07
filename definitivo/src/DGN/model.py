import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool



class GINEMLP(nn.Module):
    
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(GINEMLP, self).__init__()
    
        self.lin1 = nn.Linear(in_channels, out_channels*2)
        self.bn1 = nn.BatchNorm1d(out_channels*2)
        
        #
        self.lin2 = nn.Linear(out_channels*2, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
      
        self.act = nn.SiLU() 
        self.dropout = nn.Dropout(dropout)
        
      
        
    def forward(self, x):
        
        
        # Block1
        out = self.lin1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        
        # Block 2
        out = self.lin2(out)
        out = self.bn2(out)
        out = self.act(out)
        
       
        return out 
        
#function which adapts input and output channels        
def gin_mlp(input_dim,output_dim,dropout):
    return GINEMLP(input_dim,output_dim,dropout=dropout)

class Tox21GNN(nn.Module):
    
    def __init__(self,num_node_features,hidden_channels=64,num_classes=12,dropout=0.35,num_global_features=2,num_features_edge=11):
        super(Tox21GNN, self).__init__()
        
        self.node_encoder = nn.Linear(num_node_features, hidden_channels)
        
        self.bond_encoder = nn.Sequential(
        nn.Linear(num_features_edge, hidden_channels),
        nn.BatchNorm1d(hidden_channels),
        nn.SiLU(),
        nn.Linear(hidden_channels, hidden_channels),
        nn.SiLU()
        )
       
        self.conv1=GINEConv(gin_mlp(hidden_channels,hidden_channels,dropout))
        
        self.conv2=GINEConv(gin_mlp(hidden_channels,hidden_channels,dropout))
        self.conv3=GINEConv(gin_mlp(hidden_channels,hidden_channels,dropout))
        self.conv4=GINEConv(gin_mlp(hidden_channels,hidden_channels,dropout))
        self.conv5=GINEConv(gin_mlp(hidden_channels,hidden_channels,dropout))
        
       
        #Classifier head to obtain output of 12 classes
        self.classifier= nn.Sequential( 
            nn.Linear((hidden_channels*5*2)+num_global_features,hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
                                    
            nn.Linear(hidden_channels, hidden_channels // 2), 
            nn.BatchNorm1d(hidden_channels // 2),
            nn.SiLU(),                    # <--- Cambio Chiave
            nn.Dropout(p=dropout-0.1),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
    def forward(self,x,edge_index,edge_attr,batch,global_features):
        
        #ENCODER OVER Features
        x = self.node_encoder(x)
        #ENCODER OVER BOND
        edge_emb=self.bond_encoder(edge_attr)
        
        # Layer 1
        x1= self.conv1(x, edge_index,edge_attr=edge_emb)
        
        # Layer 2
        x2 = self.conv2(x1, edge_index,edge_attr=edge_emb)

        # Layer 3
        x3 = self.conv3(x2, edge_index,edge_attr=edge_emb)
        
        # Layer 4
        x4 = self.conv4(x3, edge_index,edge_attr=edge_emb)
        
        # Layer 5
        x5 = self.conv5(x4, edge_index,edge_attr=edge_emb)
        
        #concat embedding obtained from convolution 
        x_combined = torch.cat([x1, x2, x3, x4, x5], dim=-1)
        
        
        x_add = global_add_pool(x_combined, batch)
        x_max = global_max_pool(x_combined, batch)
        x_pool = torch.cat([x_add, x_max], dim=1)
        
        if global_features is not None:
            
            x_final= torch.cat([x_pool, global_features], dim=1)
            
        else:
            
            x_final=x_pool
        
        #calssifier NN
        logits = self.classifier(x_final)
        
        return logits