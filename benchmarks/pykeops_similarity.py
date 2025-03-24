import torch
import torch.nn as nn
import torch.nn.functional as F
from pykeops.torch import LazyTensor
from torch_scatter import scatter_max

class PyKeOpsSimilarity(nn.Module):
    """Memory-efficient similarity using PyKeOps with autograd support"""
    def __init__(self, tau=0.05, k_neighbors=10, chunk_size=None, streams=None, hard=False):
        super().__init__()
        self.tau = tau
        self.k_neighbors = k_neighbors
        self.hard = hard
    
    def forward(self, feat_x, feat_y=None):
        """Compute sparse similarity with PyKeOps optimization"""
        if feat_y is None:
            return self._process_similarity_matrix(feat_x)
        
        # Normalize features
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)
        
        batch_size, n_x, feat_dim = feat_x.shape
        n_y = feat_y.shape[1]
        device = feat_x.device
        
        # Pre-allocate output tensors
        max_elements = batch_size * n_x * self.k_neighbors
        indices = torch.zeros((3, max_elements), dtype=torch.long, device=device)
        values = torch.zeros(max_elements, dtype=torch.float32, device=device)
        ptr = 0
        
        # Process each batch separately
        for b in range(batch_size):
            # Reshape for PyKeOps
            x = feat_x[b].contiguous()  # (n_x, dim)
            y = feat_y[b].contiguous()  # (n_y, dim)
            
            # Create LazyTensors
            x_i = LazyTensor(x[:, None, :])  # (n_x, 1, dim)
            y_j = LazyTensor(y[None, :, :])  # (1, n_y, dim)
            
            # Compute dot product similarity with temperature
            sim_ij = (x_i * y_j).sum(dim=2) / self.tau  # (n_x, n_y)
            
            # Get original PyTorch tensor for processing top-k
            sim_matrix = sim_ij.sum_reduction(dim=1).view(n_x, n_y)
            
            # Find top-k using PyTorch (PyKeOps doesn't have direct top-k yet)
            top_values, top_indices = torch.topk(
                sim_matrix, k=self.k_neighbors, dim=1)
            
            # Apply softmax on the selected values
            softmax_values = F.softmax(top_values, dim=1)
            
            # Fill indices and values
            num_entries = n_x * self.k_neighbors
            
            # Create batch, row and column indices
            batch_indices = torch.full((num_entries,), b, device=device, dtype=torch.long)
            row_indices = torch.arange(n_x, device=device, dtype=torch.long)
            row_indices = row_indices.repeat_interleave(self.k_neighbors)
            col_indices = top_indices.reshape(-1)
            
            # Store in output tensors
            indices[:, ptr:ptr+num_entries] = torch.stack([batch_indices, row_indices, col_indices])
            values[ptr:ptr+num_entries] = softmax_values.reshape(-1)
            ptr += num_entries
        
        # Trim excess pre-allocated space
        indices = indices[:, :ptr]
        values = values[:ptr]
        
        # Create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, (batch_size, n_x, n_y), device=device
        ).coalesce()
        
        # Apply hard assignment if needed
        if self.hard:
            return self._convert_to_hard_assignment(sparse_tensor)
        
        return sparse_tensor
    
    def _process_similarity_matrix(self, similarity):
        """Process pre-computed similarity matrix"""
        batch_size, n_x, n_y = similarity.shape
        device = similarity.device
        
        # Pre-allocate output tensors
        max_elements = batch_size * n_x * self.k_neighbors
        indices = torch.zeros((3, max_elements), dtype=torch.long, device=device)
        values = torch.zeros(max_elements, dtype=torch.float32, device=device)
        ptr = 0
        
        for b in range(batch_size):
            # Get similarity matrix for this batch
            sim_matrix = similarity[b] / self.tau
            
            # Find top k values and indices
            top_values, top_indices = torch.topk(
                sim_matrix, k=self.k_neighbors, dim=1)
            
            # Apply softmax
            softmax_values = F.softmax(top_values, dim=1)
            
            # Fill indices and values
            num_entries = n_x * self.k_neighbors
            
            batch_indices = torch.full((num_entries,), b, device=device, dtype=torch.long)
            row_indices = torch.arange(n_x, device=device, dtype=torch.long)
            row_indices = row_indices.repeat_interleave(self.k_neighbors)
            col_indices = top_indices.reshape(-1)
            
            indices[:, ptr:ptr+num_entries] = torch.stack([batch_indices, row_indices, col_indices])
            values[ptr:ptr+num_entries] = softmax_values.reshape(-1)
            ptr += num_entries
        
        # Trim excess pre-allocated space
        indices = indices[:, :ptr]
        values = values[:ptr]
        
        # Create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, (batch_size, n_x, n_y), device=device
        ).coalesce()
        
        # Apply hard assignment if needed
        if self.hard:
            return self._convert_to_hard_assignment(sparse_tensor)
        
        return sparse_tensor
    
    def _convert_to_hard_assignment(self, sparse_tensor):
        """Convert sparse tensor to hard assignment"""
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        size = sparse_tensor.size()
        device = sparse_tensor.device
        
        # Group by batch and row indices
        batch_indices = indices[0]
        row_indices = indices[1]
        
        # Create unique keys for batch-row pairs
        keys = batch_indices * size[1] + row_indices
        
        # Find max value indices
        max_values, argmax = scatter_max(values, keys)
        
        # Create mask for max entries
        keep_mask = torch.zeros_like(values, dtype=torch.bool)
        valid_mask = argmax != -1
        keep_mask[argmax[valid_mask]] = True
        
        # Apply straight-through estimator
        hard_values = keep_mask.float().detach() - values.detach() + values
        
        return torch.sparse_coo_tensor(
            indices[:, keep_mask],
            hard_values[keep_mask],
            size,
            device=device
        ).coalesce()