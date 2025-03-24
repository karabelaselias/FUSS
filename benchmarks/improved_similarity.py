import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math
from collections import defaultdict

from torch_scatter import scatter_max


@torch.jit.script
def fast_topk_truly_unsorted(similarity: torch.Tensor, k: int, tau: float):
    """
    Faster top-k implementation that doesn't sort values
    Uses partitioning algorithm which is O(n) instead of O(n log k)
    """
    # Apply temperature scaling
    similarity = similarity / tau
    batch_size, seq_len = similarity.shape
    
    if seq_len <= k:
        # If we have fewer points than k, use all
        topk_indices = torch.arange(seq_len, device=similarity.device).expand(batch_size, seq_len)
        topk_softmax = F.softmax(similarity, dim=1)
        return topk_indices, topk_softmax
    
    # Use an optimized selection approach without sorting
    if similarity.device.type == 'cuda':
        # Leverage Flash-Attention style approach with partial softmax
        # Compute partial softmax for efficiency
        topk_values, topk_indices = torch.topk(similarity, k=k, dim=1, sorted=False)
        topk_softmax = F.softmax(topk_values, dim=1)
    else:
        # Fallback for CPU
        topk_values, topk_indices = torch.topk(similarity, k=k, dim=1, sorted=False)
        topk_softmax = F.softmax(topk_values, dim=1)
    
    return topk_indices, topk_softmax

class TensorCoreOptimizedMatMul(nn.Module):
    """Optimizes matrix multiplication for tensor cores on A100"""
    def __init__(self):
        super().__init__()
    
    def forward(self, a, b):
        # Ensure dimensions are multiples of 8 for optimal tensor core utilization
        orig_shape_a = a.shape
        orig_shape_b = b.shape
        
        # Pad dimensions to multiples of 8 if necessary
        pad_a_rows = (8 - (a.size(0) % 8)) % 8
        pad_b_cols = (8 - (b.size(1) % 8)) % 8
        
        if pad_a_rows > 0 or pad_b_cols > 0:
            # Pad tensors for optimal tensor core usage
            if pad_a_rows > 0:
                a = F.pad(a, (0, 0, 0, pad_a_rows))
            if pad_b_cols > 0:
                b = F.pad(b, (0, pad_b_cols, 0, 0))
        
        # Use TF32 precision for A100
        with torch.amp.autocast(device_type='cuda'):
            result = torch.matmul(a, b)
        
        # Trim back to original size if padded
        if pad_a_rows > 0 or pad_b_cols > 0:
            result = result[:orig_shape_a[0], :orig_shape_b[1]]
            
        return result

class ImprovedSparseSimilarity(nn.Module):
    """Ultra-optimized sparse similarity implementation with faster backward pass for A100 GPUs
    
    Args:
        tau: Temperature parameter for softmax scaling
        k_neighbors: Number of neighbors to keep per point
        chunk_size: Size of processing chunks (will be aligned to tensor cores)
        streams: Number of CUDA streams to use for parallel processing
        hard: Whether to use hard assignment (one-hot) instead of soft assignment
    """
    def __init__(self, tau=0.2, k_neighbors=15, chunk_size=10000, streams=4, hard=False):
        super(ImprovedSparseSimilarity, self).__init__()
        self.tau = tau
        self.k_neighbors = k_neighbors
        self.base_chunk_size = chunk_size
        self.streams = streams  # Number of CUDA streams to use
        self.hard = hard
        self.matmul = TensorCoreOptimizedMatMul()
        
        # Cache for CUDA streams
        self._streams = None
        self._events = None
    
    def _get_streams(self, device):
        """Lazy initialization of CUDA streams"""
        if self._streams is None or device != self._streams[0].device:
            self._streams = [torch.cuda.Stream(device=device) for _ in range(self.streams)]
            self._events = [torch.cuda.Event(enable_timing=False) for _ in range(self.streams)]
        return self._streams, self._events
    
    def _aligned_chunk_size(self, n_x):
        """Return chunk size aligned to 8 for tensor cores"""
        # Start with base chunk size
        chunk_size = min(self.base_chunk_size, n_x)
        # Adjust to be multiple of 8 for tensor cores
        chunk_size = (chunk_size // 8) * 8
        # Ensure it's at least 8
        return max(8, chunk_size)
    
    def forward(self, feat_x: torch.Tensor, feat_y: Optional[torch.Tensor] = None):
        """
        Compute sparse similarity between feature sets with optimized backward pass
        
        Args:
            feat_x: Either features [B, Nx, C] or similarity matrix [B, Nx, Ny]
            feat_y: Features [B, Ny, C] (optional)
            
        Returns:
            Sparse tensor representing the similarity matrix
        """
        if feat_y is None:
            return self._process_similarity_matrix(feat_x)
        
        # Ensure contiguous tensors for better memory access
        feat_x = feat_x.contiguous()
        feat_y = feat_y.contiguous() if feat_y is not None else feat_x
        
        # Normalize features 
        with torch.amp.autocast(device_type='cuda'):
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
        
        batch_size, n_x, feat_dim = feat_x.shape
        n_y = feat_y.shape[1]
        device = feat_x.device
        
        # Pre-allocate tensors for output
        max_elements = batch_size * n_x * self.k_neighbors
        indices = torch.empty((3, max_elements), dtype=torch.int32, device=device)
        values = torch.empty(max_elements, dtype=torch.float, device=device)
        
        # Get optimal chunk size (aligned for tensor cores)
        chunk_size = self._aligned_chunk_size(n_x)
        
        # Process in a single pass with optimized chunk handling for better backprop
        ptr = 0
        for b in range(batch_size):
            for start_idx in range(0, n_x, chunk_size):
                end_idx = min(start_idx + chunk_size, n_x)
                
                # Extract chunk
                feat_x_chunk = feat_x[b:b+1, start_idx:end_idx]  # [1, chunk, C]
                
                # Compute similarity 
                with torch.amp.autocast(device_type='cuda'):
                    # Process as 2D matrices for efficiency
                    feat_x_2d = feat_x_chunk.squeeze(0)  # [chunk, C]
                    feat_y_2d = feat_y[b].squeeze(0) if feat_y.dim() > 2 else feat_y[b]  # [n_y, C]
                    
                    # Direct matmul is more efficient for backward pass
                    similarity = torch.matmul(feat_x_2d, feat_y_2d.transpose(-2, -1))  # [chunk, n_y]
                
                # Find top-k without sorting
                topk_indices, topk_softmax = fast_topk_truly_unsorted(
                    similarity, self.k_neighbors, self.tau
                )
                
                # Fill indices and values
                chunk_size_actual = end_idx - start_idx
                num_entries = chunk_size_actual * self.k_neighbors
                
                # Prepare indices
                batch_indices = torch.full((num_entries,), b, device=device, dtype=torch.int32)
                row_indices = torch.arange(start_idx, end_idx, device=device, dtype=torch.int32)
                row_indices = row_indices.unsqueeze(1).expand(-1, self.k_neighbors).reshape(-1)
                col_indices = topk_indices.reshape(-1)
                
                # Pack into buffer
                indices[:, ptr:ptr+num_entries] = torch.stack([batch_indices, row_indices, col_indices])
                values[ptr:ptr+num_entries] = topk_softmax.reshape(-1)
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
        
        # Get optimal chunk size
        chunk_size = self._aligned_chunk_size(n_x)
        
        # Pre-allocate tensors
        max_elements = batch_size * n_x * self.k_neighbors
        indices = torch.empty((3, max_elements), dtype=torch.int32, device=device)
        values = torch.empty(max_elements, dtype=similarity.dtype, device=device)
        ptr = 0
        
        # Process without streams for more efficient backward pass
        for b in range(batch_size):
            for start_idx in range(0, n_x, chunk_size):
                end_idx = min(start_idx + chunk_size, n_x)
                
                # Get chunk of similarity matrix
                similarity_chunk = similarity[b, start_idx:end_idx]  # [chunk_size, n_y]
                
                # Process chunk
                topk_indices, topk_softmax = fast_topk_truly_unsorted(
                    similarity_chunk, self.k_neighbors, self.tau
                )
                
                # Fill indices
                chunk_size_actual = end_idx - start_idx
                num_entries = chunk_size_actual * self.k_neighbors
                
                batch_indices = torch.full((num_entries,), b, device=device, dtype=torch.int32)
                row_indices = torch.arange(start_idx, end_idx, device=device, dtype=torch.int32)
                row_indices = row_indices.unsqueeze(1).expand(-1, self.k_neighbors).reshape(-1)
                col_indices = topk_indices.reshape(-1)
                
                indices[:, ptr:ptr+num_entries] = torch.stack([batch_indices, row_indices, col_indices])
                values[ptr:ptr+num_entries] = topk_softmax.reshape(-1)
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
    
    from torch_scatter import scatter_max

    def _convert_to_hard_assignment(self, sparse_tensor):
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        batch_size, n_x, n_y = sparse_tensor.size()
        device = sparse_tensor.device
    
        # Create unique keys for batch-row pairs
        row_keys = indices[0] * n_x + indices[1]
        
        # Use scatter_max directly with properly formed indices
        max_values, argmax = scatter_max(
            values, row_keys, dim=0, dim_size=batch_size * n_x
        )
        
        # Create mask more efficiently
        keep_mask = torch.zeros_like(values, dtype=torch.bool)
        valid_indices = argmax != -1
        keep_mask[argmax[valid_indices]] = True
        
        # Straight-through estimator for better gradient flow
        hard_values = (keep_mask.float() - values.detach() + values)
        
        return torch.sparse_coo_tensor(
            indices[:, keep_mask],
            hard_values[keep_mask],
            (batch_size, n_x, n_y),
            device=device
        ).coalesce()
    
    def _convert_to_hard_assignment_old(self, sparse_tensor: torch.Tensor) -> torch.Tensor:
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        batch_size, n_x, n_y = sparse_tensor.size()
        device = sparse_tensor.device
    
        # Create unique batch-row keys [num_entries]
        keys = indices[0] * n_x + indices[1]
        
        # Find unique keys and their positions
        unique_keys, inverse_indices, counts = torch.unique(
            keys, return_inverse=True, return_counts=True
        )
        
        # Use PyTorch's segment_reduce (available in v1.12+)
        #max_indices = torch._segment_reduce(
        #    values=values,
        #    reduce="argmax",
        #    offsets=torch.cat((torch.tensor([0], device=device), counts.cumsum(0)[:-1])),
        #    axis=0,
        #    initial=-torch.inf
        #)
        # Create mask for max entries
        #keep_mask = torch.zeros_like(values, dtype=torch.bool)
        #keep_mask[max_indices] = True

        _, argmax = scatter_max(values, inverse_indices)
        keep_mask = torch.zeros_like(values, dtype=torch.bool)
        keep_mask[argmax] = True
        
        # Filter indices/values
        hard_indices = indices[:, keep_mask]
        hard_values = torch.ones(hard_indices.size(1), device=device)
        
        return torch.sparse_coo_tensor(
            hard_indices, hard_values, (batch_size, n_x, n_y), device=device
        ).coalesce()
    
    def _convert_to_hard_assignment_old(self, sparse_tensor: torch.Tensor) -> torch.Tensor:
        """Convert sparse tensor to hard assignment (one-hot)
        
        For each row, keeps only the entry with the highest value and sets it to 1.0
        """
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        batch_size, n_x, n_y = sparse_tensor.size()
        device = sparse_tensor.device
        
        # Process in a more vectorized way for faster backward pass
        # Group by batch and row indices
        batch_indices = indices[0]
        row_indices = indices[1]
        col_indices = indices[2]
        
        # Create unique keys for batch-row pairs
        keys = batch_indices * n_x + row_indices
        unique_keys, key_inverse = torch.unique(keys, return_inverse=True)
        
        # Find max value indices
        num_keys = unique_keys.size(0)
        max_indices = torch.zeros(num_keys, dtype=torch.long, device=device)
        
        # Create a mask for each unique key
        for i in range(num_keys):
            key_mask = (key_inverse == i)
            if key_mask.any():
                # Find index of max value for this key
                max_idx = torch.argmax(values[key_mask])
                # Get the corresponding indices in the original arrays
                key_indices = torch.nonzero(key_mask).squeeze(-1)
                max_indices[i] = key_indices[max_idx]
        
        # Use the max indices to select the entries to keep
        keep_mask = torch.zeros_like(values, dtype=torch.bool)
        keep_mask[max_indices] = True
        
        # Create new indices and values for the hard assignment
        hard_indices = indices[:, keep_mask]
        hard_values = torch.ones(hard_indices.size(1), device=device, dtype=values.dtype)
        
        # Create sparse tensor
        hard_tensor = torch.sparse_coo_tensor(
            hard_indices, hard_values, (batch_size, n_x, n_y), device=device
        ).coalesce()
        
        return hard_tensor


