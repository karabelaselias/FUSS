import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from typing import Optional

@torch.jit.script
def fast_topk_2d(similarity: torch.Tensor, k: int, tau: float):
    """JIT-compiled function for faster top-k selection"""
    # Apply temperature scaling
    similarity = similarity / tau
    
    # Find topk values and indices
    if similarity.size(1) <= k:
        # If we have fewer points than k, use all
        topk_indices = torch.arange(similarity.size(1), device=similarity.device).expand(similarity.size(0), similarity.size(1))
        topk_values = similarity
    else:
        # Use topk without sorting
        topk_values, topk_indices = torch.topk(similarity, k=k, dim=1, sorted=False)
    
    # Apply softmax
    topk_softmax = F.softmax(topk_values, dim=1)
    
    return topk_indices, topk_softmax

@torch.jit.script
def fast_topk_2d_mixed_precision(similarity: torch.Tensor, k: int, tau: float):
    """JIT-compiled function for faster top-k selection with mixed precision"""
    # Apply temperature scaling
    similarity = similarity / tau
    
    # Convert to half precision for faster topk computation
    orig_dtype = similarity.dtype
    
    # Only convert to half if needed and supported by the device
    if orig_dtype != torch.float16 and similarity.device.type == 'cuda':
        similarity_half = similarity.half()
    else:
        similarity_half = similarity
    
    # Find topk values and indices
    if similarity_half.size(1) <= k:
        # If we have fewer points than k, use all
        topk_indices = torch.arange(similarity_half.size(1), device=similarity_half.device).expand(similarity_half.size(0), similarity_half.size(1))
        # Return to original dtype for softmax
        topk_values = similarity
    else:
        # Find top-k without sorting on half precision
        topk_values_half, topk_indices = torch.topk(similarity_half, k=k, dim=1, sorted=False)
        
        # Gather from original precision for accurate softmax
        batch_indices = torch.arange(similarity.size(0), device=similarity.device).view(-1, 1).expand(-1, k)
        topk_values = similarity[batch_indices.reshape(-1), topk_indices.reshape(-1)].reshape(similarity.size(0), k)
    
    # Apply softmax on original precision for better accuracy
    topk_softmax = F.softmax(topk_values, dim=1)
    
    return topk_indices, topk_softmax

class OptimizedSparseSimilarity(nn.Module):
    """Memory-efficient sparse similarity implementation"""
    def __init__(self, tau=0.2, k_neighbors=15, chunk_size=5000, hard=False):
        super(OptimizedSparseSimilarity, self).__init__()
        self.tau = tau
        self.k_neighbors = k_neighbors
        self.chunk_size = chunk_size
        self.hard = hard
    
    def forward(self, feat_x: torch.Tensor, feat_y: Optional[torch.Tensor] = None):
        """
        Compute sparse similarity between feature sets
        
        Args:
            feat_x: Either features [B, Nx, C] or similarity matrix [B, Nx, Ny]
            feat_y: Features [B, Ny, C] (optional)
            
        Returns:
            Sparse tensor representing the similarity matrix
        """
        if feat_y is None:
            return self._process_similarity_matrix(feat_x)
        
        # Normalize features
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y if feat_y is not None else feat_x, dim=-1, p=2)
        
        batch_size, n_x, _ = feat_x.shape
        n_y = feat_y.shape[1]
        device = feat_x.device
        
        # Pre-allocate tensors
        max_elements = batch_size * n_x * self.k_neighbors
        indices = torch.empty((3, max_elements), dtype=torch.long, device=device)
        values = torch.empty(max_elements, dtype=feat_x.dtype, device=device)
        ptr = 0
        
        chk_sz = self._get_chunk_size(n_x, n_y)
        
        for b in range(batch_size):
            for start_idx in range(0, n_x, chk_sz):
                end_idx = min(start_idx + chk_sz, n_x)
                feat_x_chunk = feat_x[b, start_idx:end_idx]  # [chunk, C]
                
                # Mixed precision matmul
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    similarity = torch.matmul(feat_x_chunk, feat_y[b].transpose(-1, -2))
                
                topk_indices, topk_softmax = fast_topk_2d_mixed_precision(
                    similarity, self.k_neighbors, self.tau
                )
                
                chunk_size = end_idx - start_idx
                num_entries = chunk_size * self.k_neighbors
                
                # Fill indices
                batch_indices = torch.full((num_entries,), b, device=device)
                row_indices = torch.arange(start_idx, end_idx, device=device)[:, None].expand(-1, self.k_neighbors).flatten()
                col_indices = topk_indices.flatten()
                
                indices[:, ptr:ptr+num_entries] = torch.stack([batch_indices, row_indices, col_indices])
                values[ptr:ptr+num_entries] = topk_softmax.flatten()
                ptr += num_entries
        
        # Trim excess pre-allocated space
        indices = indices[:, :ptr]
        values = values[:ptr]
        
        sparse_tensor = torch.sparse_coo_tensor(indices, values, (batch_size, n_x, n_y), device=device)
        return sparse_tensor.coalesce() if not self.hard else self._convert_to_hard_assignment(sparse_tensor)

    def _get_chunk_size(self, n_x, n_y):
        """Determine optimal chunk size based on problem dimensions"""
        if n_x * n_y < 10_000_000:  # 10M elements
            base = min(self.chunk_size, 10000)  # Max chunk size
            return (base // 16) * 16
        elif n_x * n_y < 100_000_000:  # 100M elements
            base = min(self.chunk_size, 5000)  # Max chunk size
            return (base // 16) * 16
        else:  # Very large problems
            base = min(self.chunk_size, 2000)  # Max chunk size
            return (base // 16) * 16
    
    def _process_similarity_matrix(self, similarity):
        """Process pre-computed similarity matrix"""
        batch_size, n_x, n_y = similarity.shape
        device = similarity.device
        
        indices_list = []
        values_list = []
        
        # Process each batch
        for b in range(batch_size):
            # Process in chunks
            for start_idx in range(0, n_x, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, n_x)
                
                # Get chunk of similarity matrix
                similarity_chunk = similarity[b, start_idx:end_idx]  # [chunk_size, n_y]
                
                # Use JIT-compiled function for faster topk
                topk_indices, topk_softmax = fast_topk_2d(similarity_chunk, self.k_neighbors, self.tau)
                
                # Create indices for sparse tensor
                chunk_size = end_idx - start_idx
                batch_indices = torch.full((chunk_size * topk_indices.size(1),), b, device=device, dtype=torch.long)
                row_indices = torch.arange(start_idx, end_idx, device=device).view(-1, 1).expand(-1, topk_indices.size(1)).reshape(-1)
                col_indices = topk_indices.reshape(-1)
                
                # Stack indices and append
                curr_indices = torch.stack([batch_indices, row_indices, col_indices], dim=0)
                indices_list.append(curr_indices)
                values_list.append(topk_softmax.reshape(-1))
        
        # Concatenate all indices and values
        indices = torch.cat(indices_list, dim=1)
        values = torch.cat(values_list)
        
        # Create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, (batch_size, n_x, n_y), device=device
        )
        
        # Apply hard assignment if needed
        if self.hard:
            return self._convert_to_hard_assignment(sparse_tensor)
        
        return sparse_tensor.coalesce()
    
    def _convert_to_hard_assignment(self, sparse_tensor):
        """Convert sparse tensor to hard assignment (one-hot)"""
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        batch_size, n_x, n_y = sparse_tensor.size()
        device = sparse_tensor.device
        
        # Group by batch and row
        unique_batches = indices[0].unique()
        hard_indices_list = []
        hard_values_list = []
        
        for b in unique_batches:
            batch_mask = indices[0] == b
            batch_indices = indices[:, batch_mask]
            batch_values = values[batch_mask]
            
            # Get unique rows in this batch
            unique_rows = batch_indices[1].unique()
            
            for r in unique_rows:
                # Find indices for this row
                row_mask = batch_indices[1] == r
                row_cols = batch_indices[2, row_mask]
                row_vals = batch_values[row_mask]
                
                # Find max value index
                max_idx = row_vals.argmax().item()
                max_col = row_cols[max_idx].item()
                
                # Create hard assignment
                hard_indices_list.append(torch.tensor([[b], [r], [max_col]], device=device))
                hard_values_list.append(torch.tensor([1.0], device=device))
        
        if not hard_indices_list:
            return torch.sparse_coo_tensor(
                torch.zeros((3, 0), device=device, dtype=torch.long),
                torch.zeros(0, device=device),
                (batch_size, n_x, n_y), device=device
            )
        
        # Concatenate and create sparse tensor
        hard_indices = torch.cat(hard_indices_list, dim=1)
        hard_values = torch.cat(hard_values_list)
        
        return torch.sparse_coo_tensor(
            hard_indices, hard_values, (batch_size, n_x, n_y), device=device
        ).coalesce()