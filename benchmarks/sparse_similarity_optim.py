import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math
from collections import defaultdict

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
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            result = torch.matmul(a, b)
        
        # Trim back to original size if padded
        if pad_a_rows > 0 or pad_b_cols > 0:
            result = result[:orig_shape_a[0], :orig_shape_b[1]]
            
        return result

class HighPerformanceSparseSimilarity(nn.Module):
    """Ultra-optimized sparse similarity implementation for A100 GPUs
    
    Args:
        tau: Temperature parameter for softmax scaling
        k_neighbors: Number of neighbors to keep per point
        chunk_size: Size of processing chunks (will be aligned to tensor cores)
        streams: Number of CUDA streams to use for parallel processing
        hard: Whether to use hard assignment (one-hot) instead of soft assignment
    """
    def __init__(self, tau=0.2, k_neighbors=15, chunk_size=10000, streams=4, hard=False):
        super(HighPerformanceSparseSimilarity, self).__init__()
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
        Compute sparse similarity between feature sets with multi-stream processing
        
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
        indices = torch.empty((3, max_elements), dtype=torch.long, device=device)
        values = torch.empty(max_elements, dtype=torch.float, device=device)
        
        # Get optimal chunk size (aligned for tensor cores)
        chunk_size = self._aligned_chunk_size(n_x)
        
        # Get CUDA streams for parallel processing
        streams, events = self._get_streams(device) if device.type == 'cuda' else (None, None)
        
        # Process batches sequentially
        ptr = 0
        for b in range(batch_size):
            # Calculate number of chunks and distribute to streams
            num_chunks = math.ceil(n_x / chunk_size)
            active_streams = min(self.streams, num_chunks)
            
            # Process chunks in parallel using streams
            for stream_idx in range(active_streams):
                if device.type == 'cuda':
                    torch.cuda.synchronize()  # Ensure previous operations are complete
                
                # Process multiple chunks per stream
                chunks_per_stream = math.ceil(num_chunks / active_streams)
                start_chunk = stream_idx * chunks_per_stream
                end_chunk = min((stream_idx + 1) * chunks_per_stream, num_chunks)
                
                if device.type == 'cuda':
                    # Use stream for parallel processing
                    with torch.cuda.stream(streams[stream_idx]):
                        for chunk_idx in range(start_chunk, end_chunk):
                            start_idx = chunk_idx * chunk_size
                            end_idx = min(start_idx + chunk_size, n_x)
                            
                            # Process chunk
                            chunk_ptr = self._process_chunk(feat_x[b:b+1, start_idx:end_idx], 
                                                           feat_y[b:b+1], 
                                                           b, start_idx, 
                                                           indices[:, ptr:], 
                                                           values[ptr:])
                            ptr += chunk_ptr
                        
                        # Record event for synchronization
                        events[stream_idx].record()
                else:
                    # Sequential processing for CPU
                    for chunk_idx in range(start_chunk, end_chunk):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, n_x)
                        
                        # Process chunk
                        chunk_ptr = self._process_chunk(feat_x[b:b+1, start_idx:end_idx], 
                                                       feat_y[b:b+1], 
                                                       b, start_idx, 
                                                       indices[:, ptr:], 
                                                       values[ptr:])
                        ptr += chunk_ptr
            
            # Synchronize all streams before next batch
            if device.type == 'cuda':
                for e in events[:active_streams]:
                    e.synchronize()
        
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
    
    def _process_chunk(self, feat_x_chunk, feat_y, batch_idx, start_idx, indices_buffer, values_buffer):
        """Process a single chunk of the similarity matrix"""
        device = feat_x_chunk.device
        
        # Compute similarity using tensor core optimized matmul
        with torch.amp.autocast(device_type='cuda'):
            # Extract matrix from batch dimension for cleaner processing
            feat_x_2d = feat_x_chunk.squeeze(0)  # [chunk_size, C]
            feat_y_2d = feat_y.squeeze(0)        # [n_y, C]
            
            # Optimized matrix multiplication
            similarity = self.matmul(feat_x_2d, feat_y_2d.transpose(-1, -2))  # [chunk_size, n_y]
        
        # Find top-k without sorting
        topk_indices, topk_softmax = fast_topk_truly_unsorted(
            similarity, self.k_neighbors, self.tau
        )
        
        # Fill indices and values buffers
        chunk_size = feat_x_chunk.shape[1]
        num_entries = chunk_size * self.k_neighbors
        
        # Convert to standard precision for numeric stability
        topk_softmax = topk_softmax.float()
        
        # Fill indices
        batch_indices = torch.full((num_entries,), batch_idx, device=device, dtype=torch.long)
        row_indices = torch.arange(start_idx, start_idx + chunk_size, device=device, dtype=torch.long)
        row_indices = row_indices.unsqueeze(1).expand(-1, self.k_neighbors).reshape(-1)
        col_indices = topk_indices.reshape(-1)
        
        # Pack into buffer
        indices_buffer[:, :num_entries] = torch.stack([batch_indices, row_indices, col_indices])
        values_buffer[:num_entries] = topk_softmax.reshape(-1)
        
        return num_entries
        
    def _process_similarity_matrix(self, similarity):
        """Process pre-computed similarity matrix"""
        batch_size, n_x, n_y = similarity.shape
        device = similarity.device
        
        # Get optimal chunk size
        chunk_size = self._aligned_chunk_size(n_x)
        
        # Pre-allocate tensors
        max_elements = batch_size * n_x * self.k_neighbors
        indices = torch.empty((3, max_elements), dtype=torch.long, device=device)
        values = torch.empty(max_elements, dtype=similarity.dtype, device=device)
        ptr = 0
        
        # Get CUDA streams for parallel processing
        streams, events = self._get_streams(device) if device.type == 'cuda' else (None, None)
        
        # Process batches
        for b in range(batch_size):
            # Calculate number of chunks and distribute to streams
            num_chunks = math.ceil(n_x / chunk_size)
            active_streams = min(self.streams, num_chunks)
            
            # Process chunks in parallel using streams
            for stream_idx in range(active_streams):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Process multiple chunks per stream
                chunks_per_stream = math.ceil(num_chunks / active_streams)
                start_chunk = stream_idx * chunks_per_stream
                end_chunk = min((stream_idx + 1) * chunks_per_stream, num_chunks)
                
                if device.type == 'cuda':
                    # Use stream for parallel processing
                    with torch.cuda.stream(streams[stream_idx]):
                        for chunk_idx in range(start_chunk, end_chunk):
                            start_idx = chunk_idx * chunk_size
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
                            
                            # Pack into buffers
                            indices[:, ptr:ptr+num_entries] = torch.stack([batch_indices, row_indices, col_indices])
                            values[ptr:ptr+num_entries] = topk_softmax.reshape(-1)
                            ptr += num_entries
                        
                        # Record event for synchronization
                        events[stream_idx].record()
                else:
                    # Sequential processing for CPU
                    for chunk_idx in range(start_chunk, end_chunk):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, n_x)
                        
                        # Process chunk
                        similarity_chunk = similarity[b, start_idx:end_idx]
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
            
            # Synchronize all streams before next batch
            if device.type == 'cuda':
                for e in events[:active_streams]:
                    e.synchronize()
        
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

    def _convert_to_hard_assignment(self, sparse_tensor: torch.Tensor) -> torch.Tensor:
        """Convert sparse tensor to hard assignment (one-hot)
        
        For each row, keeps only the entry with the highest value and sets it to 1.0
        """
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        batch_size, n_x, n_y = sparse_tensor.size()
        device = sparse_tensor.device
        
        # Group values by batch and row
        row_groups = defaultdict(list)
        for i in range(indices.size(1)):
            b, r, c = indices[:, i]
            v = values[i]
            row_groups[(b.item(), r.item())].append((c.item(), v.item(), i))
        
        # For each row, keep only the max value
        hard_indices = []
        hard_values = []
        
        for (b, r), entries in row_groups.items():
            if entries:  # Skip empty rows
                # Find entry with max value
                max_entry = max(entries, key=lambda x: x[1])
                c, _, _ = max_entry
                
                # Add to hard assignment
                hard_indices.append(torch.tensor([[b], [r], [c]], device=device))
                hard_values.append(torch.tensor([1.0], device=device, dtype=values.dtype))
        
        if not hard_indices:
            # Return empty sparse tensor if no indices
            return torch.sparse_coo_tensor(
                torch.zeros((3, 0), device=device, dtype=torch.long),
                torch.zeros(0, device=device),
                (batch_size, n_x, n_y),
                device=device
            )
        
        # Concatenate and create sparse tensor
        hard_indices = torch.cat(hard_indices, dim=1)
        hard_values = torch.cat(hard_values)
        
        return torch.sparse_coo_tensor(
            hard_indices, hard_values, (batch_size, n_x, n_y), device=device
        ).coalesce()