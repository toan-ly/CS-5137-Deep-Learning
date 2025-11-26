import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GINConv, SAGEConv, GATConv,
    global_add_pool, 
    global_mean_pool, 
    global_max_pool
)

def pool(x, batch, pool_type):
    if pool_type == 'mean':
        return global_mean_pool(x, batch)
    elif pool_type == 'max':
        return global_max_pool(x, batch)
    elif pool_type == 'add':
        return global_add_pool(x, batch)
    elif pool_type == 'concat':
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        add_pool = global_add_pool(x, batch)
        return torch.cat([mean_pool, max_pool, add_pool], dim=1)
    else:
        raise ValueError(f"Unknown pool type: {pool_type}")

class GCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        pooling: str = 'concat',
    ):
        super().__init__()

        self.dropout = dropout
        self.pooling = pooling

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        if pooling == 'concat':
            mlp_input_dim = hidden_dim * 3
        else:
            mlp_input_dim = hidden_dim

        self.readout = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self.bn = nn.ModuleList()
        for _ in range(num_layers):
            self.bn.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bn):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = pool(x, batch, self.pooling)
        x = self.readout(x)

        return x


class GIN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        )

        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        )

        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        )   

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 3, out_dim),
        )

    def forward(self, x, edge_index, batch):
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        x1 = global_add_pool(x1, batch)
        x2 = global_add_pool(x2, batch)
        x3 = global_add_pool(x3, batch)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.mlp(x)

        return x

class GAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.5,
        pooling: str = 'add',
    ):
        super().__init__()

        assert pooling in ['mean', 'max', 'add']

        self.dropout = dropout
        self.pooling = pooling

        self.gat1 = GATConv(
            in_dim, hidden_dim, 
            heads=heads, 
            concat=False,
            dropout=dropout,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.gat2 = GATConv(
            hidden_dim, hidden_dim, 
            heads=heads, 
            concat=False,
            dropout=dropout,
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.gat2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = pool(x, batch, self.pooling)
        x = self.mlp(x)
        return x
