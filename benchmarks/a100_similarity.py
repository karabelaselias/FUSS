import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math
from torch_scatter import scatter_max

class A100OptimizedSparseSimilarity(nn.Module):
    """Ultra-optimized sparse similarity implementation for A100 GPUs
    
    Args:
        tau: Temperature parameter for softmax scaling
        k_neighbors: Number of neighbors to keep per point
        chunk_size: Size of processing chunks (will be aligned to tensor cores)
        streams: Number of CUDA streams to use for parallel processing
        hard: Whether to use hard assignment (one-hot) instead of soft assignment
        use_half: Whether to use half precision (FP16) for computations
    """
    def __init__(self, tau=0.05, k_neighbors=10, chunk_size=4096, streams=2, hard=False, use_half=True):
        super(A100OptimizedSparseSimilarity, self).__init__()
        self.tau = tau
        self.k_neighbors = k_neighbors
        self.k_backup = k_neighbors
        self.base_chunk_size = chunk_size
        self.streams = streams
        self.hard = hard
        self.use_half = use_half
        self.use_streams = streams > 1 and not self.hard
        
        # Enable TF32 for A100
        self._original_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Cache for CUDA streams
        self._streams = None
        self._events = None
    
    def __del__(self):
        # Restore original TF32 setting
        if hasattr(self, '_original_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = self._original_tf32
    
    def _get_streams(self, device):
        """Lazy initialization of CUDA streams with prioritization"""
        if self._streams is None or device != self._streams[0].device:
            # Create streams with different priorities for better scheduling
            self._streams = []
            priorities = [0, 0]  # Default and high priority
            
            for i in range(self.streams):
                # Alternate between priority levels for better work distribution
                priority = priorities[i % len(priorities)]
                stream = torch.cuda.Stream(device=device, priority=priority)
                self._streams.append(stream)
            
            # Create events for synchronization
            self._events = [torch.cuda.Event(enable_timing=False) for _ in range(self.streams)]
        return self._streams, self._events
    
    def _aligned_chunk_size(self, n_x):
        """Return chunk size aligned to 16 for A100 tensor cores"""
        # Start with base chunk size
        chunk_size = min(self.base_chunk_size, n_x)
        # Adjust to be multiple of 16 for A100 tensor cores
        chunk_size = (chunk_size // 16) * 16
        # Ensure it's at least 16
        return max(16, chunk_size)
    
    def forward(self, feat_x: torch.Tensor, feat_y: Optional[torch.Tensor] = None):
        """
        Compute sparse similarity between feature sets with optimized A100 processing
        
        Args:
            feat_x: Features [1, Nx, C]
            feat_y: Features [1, Ny, C] (optional)
            
        Returns:
            Sparse tensor representing the 2D similarity matrix
        """
        # Ensure input is 3D with batch size 1
        assert feat_x.dim() == 3 and feat_x.size(0) == 1, "Input must be [1, Nx, C]"

        if self.hard == True:
            self.k_neighbors = 1
        
        if feat_y is None:
            return self._process_similarity_matrix(feat_x)
        
        assert feat_y.dim() == 3 and feat_y.size(0) == 1, "Input must be [1, Ny, C]"
        
        # Ensure contiguous tensors
        feat_x = feat_x.contiguous()
        feat_y = feat_y.contiguous()
        
        # Normalize features with mixed precision
        with torch.amp.autocast(device_type='cuda', enabled=self.use_half):
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
        
        n_x = feat_x.shape[1]
        n_y = feat_y.shape[1]
        feat_dim = feat_x.shape[2]
        device = feat_x.device
        
        # Pre-allocate outputs with optimized dtype
        max_elements = n_x * self.k_neighbors
        indices = torch.empty((2, max_elements), dtype=torch.int32, device=device)
        values = torch.empty(max_elements, dtype=torch.float16 if self.use_half else torch.float32, device=device)
        
        # Get optimal chunk size
        chunk_size = self._aligned_chunk_size(n_x)
        
        # Use A100-optimized parallel processing
        if self.use_streams and device.type == 'cuda' and n_x > chunk_size:
            # Process with multiple streams for A100 parallelism
            return self._forward_multi_stream(feat_x, feat_y, indices, values, chunk_size)
        
        # Process in chunks
        ptr = 0
        for start_idx in range(0, n_x, chunk_size):
            end_idx = min(start_idx + chunk_size, n_x)
            chunk_size_actual = end_idx - start_idx
            
            # Extract chunk
            feat_x_chunk = feat_x[0, start_idx:end_idx]  # [chunk, C]
            
            # Compute similarity with tensor core optimization
            with torch.amp.autocast(device_type='cuda', enabled=self.use_half):
                # Process as 2D matrices 
                feat_x_2d = feat_x_chunk  # [chunk, C]
                feat_y_2d = feat_y[0]  # [n_y, C]
                
                # Use optimized matmul for A100 tensor cores
                similarity = torch.matmul(feat_x_2d, feat_y_2d.transpose(-2, -1))  # [chunk, n_y]
            
            # Find top-k with A100 optimization
            top_values, top_indices = self._optimized_topk(similarity)
            
            # Fill indices and values efficiently
            num_entries = chunk_size_actual * self.k_neighbors
            
            # Prepare indices with vectorized operations
            row_indices = torch.arange(start_idx, end_idx, device=device, dtype=torch.int32)
            row_indices = row_indices.unsqueeze(1).expand(-1, self.k_neighbors).reshape(-1)
            col_indices = top_indices.reshape(-1)
            # Add clamping
            col_indices = torch.clamp(col_indices, 0, n_y - 1)
            
            # Pack into buffer with optimized memory access
            indices[:, ptr:ptr+num_entries] = torch.stack([row_indices, col_indices])
            values[ptr:ptr+num_entries] = top_values.reshape(-1)
            ptr += num_entries
        
        # Trim excess pre-allocated space
        indices = indices[:, :ptr]
        values = values[:ptr]
        
        # Create sparse tensor with 2D layout
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, (n_x, n_y), device=device
        ).coalesce()
        
        # Apply hard assignment if needed
        if self.hard:
            self.k_neighbors = self.k_backup
            #return self._convert_to_hard_assignment_sparse(sparse_tensor)
            #return self._convert_to_hard_assignment(sparse_tensor)
        
        return sparse_tensor.to_sparse_csr()
    
    def _forward_multi_stream(self, feat_x, feat_y, indices, values, chunk_size):
        """Process forward pass using multiple CUDA streams for parallelism"""
        n_x, feat_dim = feat_x.shape[1], feat_x.shape[2]
        n_y = feat_y.shape[1]
        device = feat_x.device
        
        # Get streams and events
        streams, events = self._get_streams(device)
        num_streams = len(streams)
        
        # Calculate work per stream
        chunks_per_stream = {}
        ptr_offsets = {}
        total_ptr = 0
        
        # Distribute chunks across streams
        for start_idx in range(0, n_x, chunk_size):
            end_idx = min(start_idx + chunk_size, n_x)
            chunk_size_actual = end_idx - start_idx
            num_entries = chunk_size_actual * self.k_neighbors
            
            # Assign to stream (round-robin)
            stream_idx = (start_idx // chunk_size) % num_streams
            
            if stream_idx not in chunks_per_stream:
                chunks_per_stream[stream_idx] = []
                ptr_offsets[stream_idx] = total_ptr
                
            chunks_per_stream[stream_idx].append((start_idx, end_idx, num_entries))
            total_ptr += num_entries
        
        # Process chunks in parallel
        for stream_idx, chunks in chunks_per_stream.items():
            stream = streams[stream_idx]
            ptr = ptr_offsets[stream_idx]
            
            with torch.cuda.stream(stream):
                for start_idx, end_idx, num_entries in chunks:
                    chunk_size_actual = end_idx - start_idx
                    
                    # Extract chunk
                    feat_x_chunk = feat_x[0, start_idx:end_idx].contiguous()
                    
                    # Compute similarity with tensor core optimization
                    with torch.amp.autocast(device_type='cuda', enabled=self.use_half):
                        # Process as 2D matrices 
                        feat_x_2d = feat_x_chunk  # [chunk, C]
                        feat_y_2d = feat_y[0]  # [n_y, C]
                        
                        # Use optimized matmul for A100 tensor cores
                        similarity = torch.matmul(feat_x_2d, feat_y_2d.transpose(-2, -1))  # [chunk, n_y]
                    
                    # Find top-k with A100 optimization
                    top_values, top_indices = self._optimized_topk(similarity)
                    
                    # Prepare indices with vectorized operations
                    row_indices = torch.arange(start_idx, end_idx, device=device, dtype=torch.int32)
                    row_indices = row_indices.unsqueeze(1).expand(-1, self.k_neighbors).reshape(-1)
                    col_indices = top_indices.reshape(-1)
                    # Add clamping
                    col_indices = torch.clamp(col_indices, 0, n_y - 1)
                    
                    # Pack into buffer with optimized memory access
                    indices[:, ptr:ptr+num_entries] = torch.stack([row_indices, col_indices])
                    values[ptr:ptr+num_entries] = top_values.reshape(-1)
                    ptr += num_entries
            
            # Record event for synchronization
            events[stream_idx].record(stream)
        
        # Wait for all streams to complete
        for event in events:
            event.synchronize()
        
        # Trim to actual size
        indices = indices[:, :total_ptr]
        values = values[:total_ptr]
        
        # Create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, (n_x, n_y), device=device
        ).coalesce()
        
        # Apply hard assignment if needed
        if self.hard:
            self.k_neighbors = self.k_backup
            #return self._convert_to_hard_assignment_sparse(sparse_tensor)
            #return self._convert_to_hard_assignment(sparse_tensor)
        
        return sparse_tensor.to_sparse_csr()
    
    def _optimized_topk(self, similarity):
        """A100-optimized top-k implementation with memory layout optimizations"""
        # Apply temperature scaling
        similarity = similarity / self.tau
        
        # Ensure contiguous memory layout for A100
        if not similarity.is_contiguous():
            similarity = similarity.contiguous()
        
        # Use FP16 for topk computation which is faster on A100
        with torch.amp.autocast(device_type='cuda', enabled=self.use_half):
            # Find top-k values and indices
            topk_values, topk_indices = torch.topk(
                similarity, k=self.k_neighbors, dim=1, sorted=False
            )
            
            # Apply softmax with FP16 precision for better tensor core utilization
            topk_softmax = F.softmax(topk_values, dim=1)
        
        return topk_softmax, topk_indices
    
    def _process_similarity_matrix(self, similarity):
        """Process pre-computed similarity matrix with A100 optimizations"""
        # Ensure input is 3D with batch size 1
        assert similarity.dim() == 3 and similarity.size(0) == 1, "Input must be [1, Nx, Ny]"
        
        n_x, n_y = similarity.shape[1], similarity.shape[2]
        device = similarity.device
        
        # Get optimal chunk size aligned for A100
        chunk_size = self._aligned_chunk_size(n_x)
        
        # Pre-allocate tensors with proper alignment
        max_elements = n_x * self.k_neighbors
        indices = torch.empty((2, max_elements), dtype=torch.int32, device=device)
        values = torch.empty(max_elements, dtype=similarity.dtype, device=device)
        ptr = 0
        
        # Process in chunks for better memory locality
        for start_idx in range(0, n_x, chunk_size):
            end_idx = min(start_idx + chunk_size, n_x)
            
            # Get chunk of similarity matrix
            similarity_chunk = similarity[0, start_idx:end_idx].contiguous()
            
            # Process chunk with A100 optimization
            top_values, top_indices = self._optimized_topk(similarity_chunk)
            
            # Fill indices efficiently
            chunk_size_actual = end_idx - start_idx
            num_entries = chunk_size_actual * self.k_neighbors
            
            row_indices = torch.arange(start_idx, end_idx, device=device, dtype=torch.int32)
            row_indices = row_indices.unsqueeze(1).expand(-1, self.k_neighbors).reshape(-1)
            col_indices = top_indices.reshape(-1)
            # Add clamping
            col_indices = torch.clamp(col_indices, 0, n_y - 1)
            
            indices[:, ptr:ptr+num_entries] = torch.stack([row_indices, col_indices])
            values[ptr:ptr+num_entries] = top_values.reshape(-1)
            ptr += num_entries
        
        # Trim excess pre-allocated space
        indices = indices[:, :ptr]
        values = values[:ptr]
        
        # Create sparse tensor with 2D layout
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, (n_x, n_y), device=device
        ).coalesce()
        
        # Apply hard assignment if needed
        if self.hard:
            self.k_neighbors = self.k_backup
            #return self._convert_to_hard_assignment_sparse(sparse_tensor)
            #return self._convert_to_hard_assignment(sparse_tensor)
        
        return sparse_tensor.to_sparse_csr()
    
    def _convert_to_hard_assignment_sparse(self, sparse_tensor):
        """Fast GPU-based hard assignment that avoids memory explosion
        
        This method:
        1. Uses sorting and grouping for efficient batch processing
        2. Stays entirely on GPU
        3. Minimizes Python loop overhead
        4. Preserves gradient flow
        
        Args:
            sparse_tensor: Input sparse tensor
            
        Returns:
            Sparse tensor with hard assignment (one-hot)
        """
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        n_x, n_y = sparse_tensor.size()
        device = sparse_tensor.device
        
        # Sort by row to group indices
        perm = torch.argsort(indices[0])
        sorted_indices = indices[:, perm]
        sorted_values = values[perm]
        
        # Find row boundaries
        rows = sorted_indices[0]
        row_boundaries = torch.cat([
            torch.tensor([0], device=device),
            torch.nonzero(rows[1:] != rows[:-1]).reshape(-1) + 1,
            torch.tensor([rows.shape[0]], device=device)
        ])
        
        # Pre-allocate result arrays with estimated size
        est_size = min(indices.shape[1] // 10 + 1000, n_x)  # Estimate number of unique rows
        row_indices = torch.zeros(est_size, dtype=torch.int32, device=device)
        col_indices = torch.zeros(est_size, dtype=torch.int32, device=device)
        max_values = torch.zeros(est_size, device=device)
        max_orig_indices = torch.zeros(est_size, dtype=torch.int32, device=device)
        
        # Process row groups
        count = 0
        for i in range(len(row_boundaries) - 1):
            start_idx = row_boundaries[i]
            end_idx = row_boundaries[i+1]
            
            if start_idx == end_idx:
                continue
                
            # Get values for this row
            row_vals = sorted_values[start_idx:end_idx]
            
            # Find max value
            max_idx = row_vals.argmax() + start_idx
            
            # Store result
            if count >= est_size:
                # Resize arrays if needed
                new_size = est_size * 2
                row_indices = torch.cat([row_indices, torch.zeros(est_size, dtype=torch.int32, device=device)])
                col_indices = torch.cat([col_indices, torch.zeros(est_size, dtype=torch.int32, device=device)])
                max_values = torch.cat([max_values, torch.zeros(est_size, device=device)])
                max_orig_indices = torch.cat([max_orig_indices, torch.zeros(est_size, dtype=torch.int32, device=device)])
                est_size = new_size
                
            row_indices[count] = sorted_indices[0, max_idx]
            col_indices[count] = sorted_indices[1, max_idx]
            max_values[count] = sorted_values[max_idx]
            max_orig_indices[count] = perm[max_idx]  # Store original index
            count += 1
        
        # Trim to actual size
        row_indices = row_indices[:count]
        col_indices = col_indices[:count]
        max_values = max_values[:count]
        max_orig_indices = max_orig_indices[:count]
        
        # Create mask for max entries
        keep_mask = torch.zeros_like(values, dtype=torch.bool)
        keep_mask[max_orig_indices] = True
        
        # Create hard assignment with straight-through estimator
        hard_values = torch.zeros_like(values)
        hard_values[keep_mask] = 1.0
        final_values = hard_values - values.detach() + values
        
        # Create result with only maximal values
        new_indices = torch.stack([row_indices, col_indices])
        new_values = torch.ones_like(max_values)  # Use ones for hard assignment
        
        # Add straight-through estimator for gradient flow
        new_values = new_values - max_values.detach() + max_values
        
        result = torch.sparse_coo_tensor(
            new_indices,
            new_values,
            (n_x, n_y),
            device=device
        ).coalesce()
        
        return result.to_sparse_csr()
    
    def _convert_to_hard_assignment(self, sparse_tensor):
        """A100-optimized hard assignment conversion with better gradient flow"""
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        n_x, n_y = sparse_tensor.size()
        device = sparse_tensor.device
        
        # Create unique keys for batch-row pairs with optimized bitshift operations
        # Using bit shifting is faster than multiplication for power-of-2 dimensions
        row_shift = int(torch.ceil(torch.log2(torch.tensor(n_x, dtype=torch.float))))
        row_keys = (indices[0] << row_shift) + indices[1]
        
        # Use torch_scatter which has optimized CUDA kernels
        # This is much faster than manual looping or torch.unique operations
        max_values, argmax = scatter_max(
            values, row_keys, dim=0
        )
        # Create mask for max entries directly with optimized indexing
        keep_mask = torch.zeros_like(values, dtype=torch.bool)
        valid_mask = argmax != -1  # Filter out empty rows
        
        # Gather the argmax indices to use in the mask
        # This provides better gradient flow for backward pass
        if valid_mask.any():
            keep_mask[argmax[valid_mask]] = True
        
        # Straight-through estimator for gradient flow
        # This is critical for training stability and convergence
        # The detach() operation ensures we don't double-count gradients
        hard_values = torch.ones_like(values)
        hard_values = hard_values.masked_fill(~keep_mask, 0)
        
        # Create optimized straight-through gradient estimator
        # This allows gradients to flow through the hard assignment
        final_values = hard_values - values.detach() + values
        
        # Create sparse tensor directly with optimized memory layout
        result = torch.sparse_coo_tensor(
            indices[:, keep_mask],
            final_values[keep_mask],
            (n_x, n_y),
            device=device
        ).coalesce()
        
        return result.to_sparse_csr()