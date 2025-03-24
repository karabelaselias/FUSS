# In networks/edgeconv_network.py

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing  # Using MessagePassing instead of EdgeConv
from torch_sparse import SparseTensor

from utils.registry import NETWORK_REGISTRY

# Implement a more efficient EdgeConv using sparse operations
class SparseEdgeConv(MessagePassing):
    def __init__(self, nn_module):
        super(SparseEdgeConv, self).__init__(aggr='max')  # Use max aggregation
        self.nn_module = nn_module
        
    def forward(self, x, edge_index):
        # Convert edge_index to SparseTensor for more efficient operations
        adj_t = None
        if isinstance(edge_index, SparseTensor):
            adj_t = edge_index
        else:
            adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                sparse_sizes=(x.size(0), x.size(0)))
        
        # No need to create dense matrices or compute full feature differences
        return self.propagate(adj_t, x=x)
    
    def message(self, x_i, x_j):
        # Process only the edge features without creating a dense matrix
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)
        return self.nn_module(edge_features)

# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

# Replace EdgeConv with SparseEdgeConv in ResnetECPos
@NETWORK_REGISTRY.register()
class SparseResnetECPos(torch.nn.Module):
    def __init__(self, c_dim=128, dim=3, hidden_dim=128, use_mlp=False):
        super(SparseResnetECPos, self).__init__()
        self.c_dim = c_dim

        self.fc_pos = torch.nn.Linear(dim, 2*hidden_dim)
        
        # Use SparseEdgeConv instead of EdgeConv
        self.block_0 = SparseEdgeConv(ResnetBlockFC(4*hidden_dim, hidden_dim))
        self.block_1 = SparseEdgeConv(ResnetBlockFC(4*hidden_dim+2*dim, hidden_dim))
        self.block_2 = SparseEdgeConv(ResnetBlockFC(4*hidden_dim+2*dim, hidden_dim))
        self.block_3 = SparseEdgeConv(ResnetBlockFC(4*hidden_dim+2*dim, hidden_dim))
        self.block_4 = SparseEdgeConv(ResnetBlockFC(4*hidden_dim+2*dim, hidden_dim))
        
        if use_mlp:
            self.fc_c = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, c_dim)
            )
        else:
            self.fc_c = torch.nn.Linear(hidden_dim, c_dim)

        self.actvn = torch.nn.ReLU()
        self.pool = maxpool

    def forward(self, verts, faces, feats=None):
        squeeze = False
        if verts.dim() == 3:
            verts, faces = verts.squeeze(0), faces.squeeze(0)
            if feats is not None:
                feats = feats.squeeze(0)
            squeeze = True
        
        # Convert to SparseTensor once for more efficient operations
        edge_index = get_edge_index(faces)
        sparse_adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                                 sparse_sizes=(verts.shape[0], verts.shape[0]))
        
        p = verts if feats is None else feats
        net = self.fc_pos(p)
        
        # Forward pass with sparse operations
        net = self.block_0(net, sparse_adj)
        
        # Use sparse global pooling to avoid creating dense matrices
        pooled = sparse_global_pool(net, batch_size=verts.shape[0])
        net = torch.cat([net, pooled.expand_as(net), p], dim=1)
        
        net = self.block_1(net, sparse_adj)
        pooled = sparse_global_pool(net, batch_size=verts.shape[0])
        net = torch.cat([net, pooled.expand_as(net), p], dim=1)
        
        net = self.block_2(net, sparse_adj)
        pooled = sparse_global_pool(net, batch_size=verts.shape[0])
        net = torch.cat([net, pooled.expand_as(net), p], dim=1)
        
        net = self.block_3(net, sparse_adj)
        pooled = sparse_global_pool(net, batch_size=verts.shape[0])
        net = torch.cat([net, pooled.expand_as(net), p], dim=1)
        
        net = self.block_4(net, sparse_adj)

        c = self.fc_c(self.actvn(net))

        if squeeze:
            c = c.unsqueeze(0)
        return c

# Sparse global pooling operation
def sparse_global_pool(x, batch_size=None):
    """Efficient global pooling without creating dense matrices"""
    if batch_size is None:
        return x.max(dim=0, keepdim=True)[0]
    else:
        # For batched inputs
        return x.reshape(batch_size, -1).max(dim=1, keepdim=True)[0]

# Optimized get_edge_index using sparse operations
def get_edge_index(face):
    """
    Generate edge indices from faces using sparse operations
    to avoid creating large dense matrices
    """
    # Create sparse incidence matrix: faces <-> edges
    num_faces = face.shape[0]
    num_vertices = face.max().item() + 1
    
    # Create edges from faces with sparse operations
    edges_set = set()
    
    # Process face edges
    for i in range(num_faces):
        v1, v2, v3 = face[i]
        v1, v2, v3 = int(v1), int(v2), int(v3)
        edges_set.add((min(v1, v2), max(v1, v2)))
        edges_set.add((min(v1, v3), max(v1, v3)))
        edges_set.add((min(v2, v3), max(v2, v3)))
    
    # Convert to edge index format (for undirected edges)
    edge_index = torch.zeros((2, 2 * len(edges_set)), dtype=face.dtype, device=face.device)
    for i, (src, dst) in enumerate(edges_set):
        # Each edge appears twice for undirected graph (src->dst and dst->src)
        edge_index[0, i] = src
        edge_index[1, i] = dst
        edge_index[0, i + len(edges_set)] = dst
        edge_index[1, i + len(edges_set)] = src
    
    return edge_index