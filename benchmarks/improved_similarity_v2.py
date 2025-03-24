import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math

@torch.jit.script
def fast_topk_truly_unsorted(similarity: torch.Tensor, k: int, tau: float):
    """Faster top-k implementation that doesn't sort values"""
    # Apply temperature scaling
    similarity = similarity / tau
    batch_size, seq_len = similarity.shape
    
    if seq_len <= k:
        # If we have fewer points than k, use all
        topk_indices = torch.arange(seq_len, device=similarity.device).expand(batch_size, seq_len)
        topk_softmax = F.softmax(similarity, dim=1)
        return topk_indices, topk_softmax
    
    # Use an optimized selection approach without sorting
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
    """Ultra-optimized sparse similarity with efficient hard assignment support"""
    def __init__(self, tau=0.2, k_neighbors=15, chunk_size=10000, streams=1, hard=False):
        super(ImprovedSparseSimilarity, self).__init__()
        self.tau = tau
        self.k_neighbors = k_neighbors
        self.base_chunk_size = chunk_size
        self.hard = hard
        self.matmul = TensorCoreOptimizedMatMul()
    
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
        Compute sparse similarity between feature sets with optimized processing
        
        Args:
            feat_x: Either features [B, Nx, C] or similarity matrix [B, Nx, Ny]
            feat_y: Features [B, Ny, C] (optional)
            
        Returns:
            Sparse tensor representing the similarity matrix
        """
        if feat_y is None:
            # Handle pre-computed similarity matrix
            return self._process_similarity_matrix(feat_x)
        
        # Handle feature inputs with either soft or hard assignment
        if self.hard:
            return self._hard_assignment_from_features(feat_x, feat_y)
        else:
            return self._soft_assignment_from_features(feat_x, feat_y)
    
    def _process_similarity_matrix(self, similarity):
        """Process pre-computed similarity matrix"""
        if self.hard:
            return self._hard_assignment_from_matrix(similarity)
        else:
            return self._soft_assignment_from_matrix(similarity)
    
    def _soft_assignment_from_features(self, feat_x, feat_y):
        """Compute soft sparse similarity from feature inputs"""
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
        
        # Get optimal chunk size (aligned for tensor cores)
        chunk_size = self._aligned_chunk_size(n_x)
        
        # Pre-allocate tensors for output
        max_elements = batch_size * n_x * self.k_neighbors
        indices = torch.empty((3, max_elements), dtype=torch.long, device=device)
        values = torch.empty(max_elements, dtype=torch.float, device=device)
        
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
                    
                    # Use optimized matmul
                    similarity = self.matmul(feat_x_2d, feat_y_2d.transpose(-2, -1))  # [chunk, n_y]
                
                # Find top-k without sorting
                topk_indices, topk_softmax = fast_topk_truly_unsorted(
                    similarity, self.k_neighbors, self.tau
                )
                
                # Fill indices and values
                chunk_size_actual = end_idx - start_idx
                num_entries = chunk_size_actual * self.k_neighbors
                
                # Prepare indices
                batch_indices = torch.full((num_entries,), b, device=device, dtype=torch.long)
                row_indices = torch.arange(start_idx, end_idx, device=device, dtype=torch.long)
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
        
        return sparse_tensor
    
    def _hard_assignment_from_features(self, feat_x, feat_y):
        """Compute hard sparse similarity (one-hot) from feature inputs"""
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
        
        # Get optimal chunk size (aligned for tensor cores)
        chunk_size = self._aligned_chunk_size(n_x)
        
        # Pre-allocate tensors for output
        indices = torch.empty((3, batch_size * n_x), dtype=torch.long, device=device)
        values = torch.ones(batch_size * n_x, dtype=torch.float, device=device)
        
        # Process in chunks for memory efficiency
        ptr = 0
        for b in range(batch_size):
            # Pre-allocate max values and indices for this batch
            max_vals = torch.full((n_x,), float('-inf'), device=device)
            max_idxs = torch.zeros(n_x, dtype=torch.long, device=device)
            
            # Process x-features in chunks
            for start_x in range(0, n_x, chunk_size):
                end_x = min(start_x + chunk_size, n_x)
                chunk_x = feat_x[b, start_x:end_x]  # [chunk_x, C]
                
                # Process all y-features at once if they fit in memory
                if n_y <= self.base_chunk_size:
                    # Direct MatMul for tensor cores
                    with torch.amp.autocast(device_type='cuda'):
                        similarity = self.matmul(chunk_x, feat_y[b].transpose(-2, -1))  # [chunk_x, n_y]
                    
                    # Scale by temperature
                    similarity = similarity / self.tau
                    
                    # Find max values and indices
                    chunk_max_vals, chunk_max_idxs = similarity.max(dim=1)
                    
                    # Update global max if better
                    max_vals[start_x:end_x] = chunk_max_vals
                    max_idxs[start_x:end_x] = chunk_max_idxs
                else:
                    # Process y-features in chunks too
                    for start_y in range(0, n_y, self.base_chunk_size):
                        end_y = min(start_y + self.base_chunk_size, n_y)
                        chunk_y = feat_y[b, start_y:end_y]  # [chunk_y, C]
                        
                        # Compute similarity for this chunk
                        with torch.amp.autocast(device_type='cuda'):
                            similarity = self.matmul(chunk_x, chunk_y.transpose(-2, -1))  # [chunk_x, chunk_y]
                        
                        # Scale by temperature
                        similarity = similarity / self.tau
                        
                        # Find max values and indices for this chunk
                        chunk_max_vals, chunk_max_idxs = similarity.max(dim=1)
                        
                        # Adjust indices to account for chunk offset
                        chunk_max_idxs += start_y
                        
                        # Update global max if better
                        for i in range(end_x - start_x):
                            if chunk_max_vals[i] > max_vals[start_x + i]:
                                max_vals[start_x + i] = chunk_max_vals[i]
                                max_idxs[start_x + i] = chunk_max_idxs[i]
            
            # Build indices for this batch
            num_entries = n_x
            batch_indices = torch.full((num_entries,), b, device=device, dtype=torch.long)
            row_indices = torch.arange(n_x, device=device, dtype=torch.long)
            col_indices = max_idxs
            
            # Pack into buffer
            indices[:, ptr:ptr+num_entries] = torch.stack([batch_indices, row_indices, col_indices])
            ptr += num_entries
        
        # Create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            indices[:, :ptr], values[:ptr], (batch_size, n_x, n_y), device=device
        ).coalesce()
        
        return sparse_tensor
    
    def _soft_assignment_from_matrix(self, similarity):
        """Process pre-computed similarity matrix with soft assignment"""
        batch_size, n_x, n_y = similarity.shape
        device = similarity.device
        
        # Get optimal chunk size
        chunk_size = self._aligned_chunk_size(n_x)
        
        # Pre-allocate tensors
        max_elements = batch_size * n_x * self.k_neighbors
        indices = torch.empty((3, max_elements), dtype=torch.long, device=device)
        values = torch.empty(max_elements, dtype=similarity.dtype, device=device)
        ptr = 0
        
        # Process batches
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
                
                batch_indices = torch.full((num_entries,), b, device=device, dtype=torch.long)
                row_indices = torch.arange(start_idx, end_idx, device=device, dtype=torch.long)
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
        
        return sparse_tensor
    
    def _hard_assignment_from_matrix(self, similarity):
        """Process pre-computed similarity matrix with hard assignment (one-hot)"""
        batch_size, n_x, n_y = similarity.shape
        device = similarity.device
        
        # Pre-allocate tensors
        indices = torch.empty((3, batch_size * n_x), dtype=torch.long, device=device)
        values = torch.ones(batch_size * n_x, dtype=similarity.dtype, device=device)
        
        # Process in chunks for memory efficiency
        ptr = 0
        for b in range(batch_size):
            # Find max indices for each row
            _, max_indices = similarity[b].max(dim=1)
            
            # Build indices
            batch_indices = torch.full((n_x,), b, device=device, dtype=torch.long)
            row_indices = torch.arange(n_x, device=device, dtype=torch.long)
            
            # Pack into buffer
            indices[:, ptr:ptr+n_x] = torch.stack([batch_indices, row_indices, max_indices])
            ptr += n_x
        
        # Create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            indices[:, :ptr], values[:ptr], (batch_size, n_x, n_y), device=device
        ).coalesce()
        
        return sparse_tensor