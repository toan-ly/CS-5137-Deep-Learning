
import os
import random
from typing import List, Tuple, Dict
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess_graph_labels(labels_raw):
    """
    Map raw labels to [0, num_classes) for cross entropy 
    loss compatibility
    """
    return labels_raw - 1

def preprocess_node_features(node_features, node_labels):
    """
    Combine node attributes with onehot encoding of node labels
    to create enhanced node features
    """
    node_labels -= 1 # Map labels to [0, num_labels)
    label_onehot = np.eye(len(np.unique(node_labels)), dtype=np.float32)[node_labels]
    node_features = node_features.astype(np.float32)
    
    # Add onehot node labels as new features
    new_features = np.concatenate([node_features, label_onehot], axis=1)

    # Convert to torch tensor
    new_features = torch.from_numpy(new_features)
    return new_features


def load_data(root, prefix='ENZYMES'):
    """
    Load raw data files
    """
    root = Path(root)

    adj_mat = np.loadtxt(root / f"{prefix}_A.txt", delimiter=",", dtype=int)          
    graph_indicator = np.loadtxt(root / f"{prefix}_graph_indicator.txt", dtype=int) 
    graph_labels_raw = np.loadtxt(root / f"{prefix}_graph_labels.txt", dtype=int)   
    node_attributes = np.loadtxt(root / f"{prefix}_node_attributes.txt", delimiter=",")
    node_labels_raw = np.loadtxt(root / f"{prefix}_node_labels.txt", dtype=int)        

    return adj_mat, graph_indicator, graph_labels_raw, node_attributes, node_labels_raw

def preprocess(root):
    adj_mat, graph_indicator, graph_labels_raw, node_attributes, node_labels_raw = load_data(root)
    num_graphs = int(graph_indicator.max())

    # Map graph labels to [0, num_classes) for cross-entropy loss
    graph_labels = preprocess_graph_labels(graph_labels_raw)

    # Enhance node features by adding label onehot
    node_attributes = preprocess_node_features(node_attributes, node_labels_raw)
    n_features = node_attributes.size(1)

    data_list = [] # store each graph as Data object

    adj_mat -= 1 # map nodes in edge list to 0-based

    for g_id in range(1, num_graphs + 1):
        # Find all nodes belong to graph g_id
        node_idx = np.where(graph_indicator == g_id)[0] 
        x = node_attributes[torch.from_numpy(node_idx)] # shape [num_nodes_g, n_features]

        # map global node idx -> local idx 0..(num_nodes_g-1)
        node_map = {n_id: i for i, n_id in enumerate(node_idx)}

        # Find all edges having source node belongs to graph g_id
        mask = (graph_indicator[adj_mat[:, 0]] == g_id)
        edges_g = adj_mat[mask] 

        if edges_g.shape[0] == 0:
            # no edges in this graph
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            src = [node_map[int(u)] for u in edges_g[:, 0]]
            dst = [node_map[int(v)] for v in edges_g[:, 1]]

            # undirected graph: add (u, v) and (v, u)
            src_all = src + dst
            dst_all = dst + src
            edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)

        # graph label
        y = torch.tensor([graph_labels[g_id - 1]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list, n_features

def stratified_split_indices(
    labels,
    train_size: float = 0.8,
    seed: int = 42,
):
    """
    Stratified split indices into train/val/test sets 
    """
    set_seed(seed)

    indices = np.arange(len(labels))
    labels = np.array(labels)

    # train vs (val+test)
    train_idx, valtest_idx, y_train, y_valtest = train_test_split(
        indices,
        labels,
        train_size=train_size,
        random_state=seed,
        stratify=labels,
    )

    # val vs test
    val_idx, test_idx, _, _ = train_test_split(
        valtest_idx,
        y_valtest,
        test_size=0.5,
        random_state=seed,
        stratify=y_valtest,
    )

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

def get_dataloaders(
    data_list,
    batch_size: int = 32,
    seed: int = 42,
    train_size: float = 0.8,
):
    """
    Create data loaders for train/val/test sets
    using stratified splitting
    """
    labels = [int(data.y.item()) for data in data_list]

    # Split based on graph labels
    train_idx, val_idx, test_idx = stratified_split_indices(
        labels,
        train_size=train_size,
        seed=seed,
    )

    train_dataset = [data_list[i] for i in train_idx]
    val_dataset = [data_list[i] for i in val_idx]
    test_dataset = [data_list[i] for i in test_idx]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader