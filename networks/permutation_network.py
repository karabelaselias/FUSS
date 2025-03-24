import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import NETWORK_REGISTRY

@torch.jit.script
def fast_topk_2d_mixed_precision(similarity: torch.Tensor, k: int):
    """JIT-compiled function for faster top-k selection with mixed precision"""
    # Convert to half precision for faster topk computation
    orig_dtype = similarity.dtype
    
    # Only convert to half if needed and supported by the device
    if orig_dtype != torch.float16 and similarity.device.type == 'cuda':
        similarity_half = similarity.half()
    else:
        similarity_half = similarity
    
    if similarity_half.size(1) <= k:
        # If we have fewer points than k, use all
        topk_indices = torch.arange(similarity_half.size(1), device=similarity_half.device).expand(similarity_half.size(0), similarity_half.size(1))
        # Return to original dtype for softmax
        topk_values = similarity
    else:
        # Find top-k without sorting on half precision
        _, topk_indices = torch.topk(similarity_half, k=k, dim=1, sorted=False)
        
        # Gather from original precision for accurate softmax
        batch_indices = torch.arange(similarity.size(0), device=similarity.device).view(-1, 1).expand(-1, k)
        topk_values = similarity[batch_indices.reshape(-1), topk_indices.reshape(-1)].reshape(similarity.size(0), k)
    
    # Apply softmax on original precision for better accuracy
    topk_softmax = F.softmax(topk_values, dim=1)
    
    return topk_indices, topk_softmax


def fast_sparse_bmm(indices, values, size, dense_matrix):
    """
    Efficient implementation of sparse-dense batch matrix multiplication
    """
    # Check if we're dealing with 2D or 3D case
    if len(size) == 2:
        # 2D case: Standard sparse mm
        sparse_matrix = torch.sparse_coo_tensor(indices, values, size)
        return torch.sparse.mm(sparse_matrix, dense_matrix)
    else:
        # 3D case: Batched multiplication
        B = size[0]
        M = size[1]
        N = size[2]
        K = dense_matrix.shape[2]
        
        # Create result tensor
        result = torch.zeros((B, M, K), device=dense_matrix.device, dtype=dense_matrix.dtype)
        
        # Process each batch
        for b in range(B):
            # Extract indices for this batch
            batch_mask = (indices[0] == b)
            rows = indices[1, batch_mask]
            cols = indices[2, batch_mask]
            batch_values = values[batch_mask]
            
            # Create 2D sparse tensor for this batch
            batch_indices = torch.stack([rows, cols])
            batch_sparse = torch.sparse_coo_tensor(
                batch_indices, batch_values, (M, N),
                device=dense_matrix.device, dtype=values.dtype
            ).coalesce()
            
            # Multiply with dense matrix and store result
            result[b] = torch.sparse.mm(batch_sparse, dense_matrix[b])
        
        return result

def apply_sparse_similarity(sparse_matrix, features):
    """
    Apply sparse similarity matrix to feature tensor
    """
    # Handle different dimension cases
    if sparse_matrix.dim() == 2 and features.dim() == 2:
        # 2D case - direct sparse multiply
        return torch.sparse.mm(sparse_matrix, features)
    
    elif sparse_matrix.dim() == 3 and features.dim() == 3:
        # 3D case - use optimized version
        return fast_sparse_bmm(
            sparse_matrix.indices(), 
            sparse_matrix.values(), 
            sparse_matrix.size(),
            features
        )
    
    else:
        raise ValueError(f"Dimension mismatch: sparse_matrix.dim()={sparse_matrix.dim()}, features.dim()={features.dim()}")

def compute_permutation_matrix_sparse(feat_x, feat_y, model):
    """
    Compute sparse permutation matrix between two feature sets
    """
    # Normalize feature vectors
    if feat_x.dim() == 3:
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)
    else:
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)
    
    # Use direct sparse computation
    sparse_matrix = model(feat_x, feat_y)
    
    return sparse_matrix

@NETWORK_REGISTRY.register()
class SparseSimilarityOld(nn.Module):
    """
    Optimized sparse similarity implementation with optional hard assignment
    """
    def __init__(self, tau=0.2, k_neighbors=15, chunk_size=10000, 
                 use_mixed_precision=True, fp16_topk=True, adaptive_chunk=True,
                 hard=False):
        """
        Initialize SparseSimilarity module
        
        Args:
            tau (float): Temperature parameter for softmax
            k_neighbors (int): Number of neighbors to keep for each row
            chunk_size (int): Size of chunks for processing large matrices
            use_mixed_precision (bool): Whether to use mixed precision for computation
            fp16_topk (bool): Whether to use FP16 for topk operation
            adaptive_chunk (bool): Whether to adaptively determine chunk size
            hard (bool): Whether to use hard assignment (one-hot)
        """
        super(SparseSimilarity, self).__init__()
        self.tau = tau
        self.k_neighbors = k_neighbors
        self.base_chunk_size = chunk_size
        self.use_mixed_precision = use_mixed_precision
        self.fp16_topk = fp16_topk
        self.adaptive_chunk = adaptive_chunk
        self.hard = hard
    
    def forward(self, feat_x, feat_y=None):
        """
        Compute sparse similarity between feature sets
        
        Args:
            feat_x: Either 2D [Nx, C] or 3D [B, Nx, C] features, or similarity matrix
            feat_y: Either 2D [Ny, C] or 3D [B, Ny, C] features (optional)
            
        Returns:
            Sparse tensor representing the similarity matrix
        """
        # Check dimensions
        is_2d = feat_x.dim() == 2
        
        # For 2D inputs (batch size 1), use optimized 2D path
        if is_2d:
            return self._process_2d(feat_x, feat_y)
        
        # For 3D with batch size 1, squeeze and use 2D path
        if feat_x.size(0) == 1 and (feat_y is None or feat_y.size(0) == 1):
            if feat_y is None:
                result_2d = self._process_2d(feat_x.squeeze(0), None)
            else:
                result_2d = self._process_2d(feat_x.squeeze(0), feat_y.squeeze(0))
            
            # Convert back to 3D
            if isinstance(result_2d, tuple):
                sparse_2d = result_2d[0]
            else:
                sparse_2d = result_2d
                
            # Add batch dimension back
            indices_2d = sparse_2d.indices()
            values = sparse_2d.values()
            indices_3d = torch.cat([
                torch.zeros(1, indices_2d.size(1), device=indices_2d.device, dtype=indices_2d.dtype),
                indices_2d
            ], dim=0)
            sparse_3d = torch.sparse_coo_tensor(
                indices_3d, values, (1, sparse_2d.size(0), sparse_2d.size(1)), 
                device=sparse_2d.device, dtype=sparse_2d.dtype
            ).coalesce()
            return sparse_3d
        
        # Otherwise use 3D path
        return self._process_3d(feat_x, feat_y)

    def _get_chunk_size(self, Nx, Ny):
        """Determine optimal chunk size based on problem dimensions"""
        if not self.adaptive_chunk:
            return self.base_chunk_size
            
        # Recommended chunk sizes based on benchmark results
        if Nx <= 1000:
            return min(self.base_chunk_size, 5000)
        elif Nx <= 10000:
            return min(self.base_chunk_size, 10000)
        else:
            # For large matrices, use the largest chunk size that fits in memory
            return min(self.base_chunk_size, 10000)
    
    def _apply_hard_assignment_2d(self, indices, values, shape, device, dtype):
        """Apply hard assignment for 2D sparse tensor"""
        if not self.hard:
            return torch.sparse_coo_tensor(indices, values, shape, device=device, dtype=dtype).coalesce()
        
        # For hard assignment, we need to find the max value for each row
        rows = indices[0].unique()
        new_indices = []
        new_values = []
        
        for row in rows:
            # Get all column indices for this row
            row_mask = indices[0] == row
            row_cols = indices[1, row_mask]
            row_vals = values[row_mask]
            
            # Find the max value index
            max_idx = row_vals.argmax()
            max_col = row_cols[max_idx]
            
            # Store only the max value with a weight of 1.0
            new_indices.append(torch.tensor([[row], [max_col]], device=device))
            new_values.append(torch.tensor([1.0], device=device, dtype=dtype))
        
        if new_indices:
            # Concatenate all indices and values
            all_indices = torch.cat(new_indices, dim=1)
            all_values = torch.cat(new_values)
            
            return torch.sparse_coo_tensor(all_indices, all_values, shape, device=device, dtype=dtype).coalesce()
        else:
            # Empty sparse tensor
            empty_indices = torch.zeros((2, 0), device=device, dtype=torch.long)
            empty_values = torch.zeros(0, device=device, dtype=dtype)
            return torch.sparse_coo_tensor(empty_indices, empty_values, shape)
    
    def _apply_hard_assignment_3d(self, indices, values, shape, device, dtype):
        """Apply hard assignment for 3D sparse tensor"""
        if not self.hard:
            return torch.sparse_coo_tensor(indices, values, shape, device=device, dtype=dtype).coalesce()
        
        # For hard assignment, we need to find the max value for each row in each batch
        batches = indices[0].unique()
        new_indices = []
        new_values = []
        
        for batch in batches:
            # Get all indices for this batch
            batch_mask = indices[0] == batch
            batch_rows = indices[1, batch_mask]
            batch_cols = indices[2, batch_mask]
            batch_vals = values[batch_mask]
            
            # Get unique rows for this batch
            rows = batch_rows.unique()
            
            for row in rows:
                # Get all column indices for this row in this batch
                row_mask = batch_rows == row
                row_cols = batch_cols[row_mask]
                row_vals = batch_vals[row_mask]
                
                # Find the max value index
                max_idx = row_vals.argmax()
                max_col = row_cols[max_idx]
                
                # Store only the max value with a weight of 1.0
                new_indices.append(torch.tensor([[batch], [row], [max_col]], device=device))
                new_values.append(torch.tensor([1.0], device=device, dtype=dtype))
        
        if new_indices:
            # Concatenate all indices and values
            all_indices = torch.cat(new_indices, dim=1)
            all_values = torch.cat(new_values)
            
            return torch.sparse_coo_tensor(all_indices, all_values, shape, device=device, dtype=dtype).coalesce()
        else:
            # Empty sparse tensor
            empty_indices = torch.zeros((3, 0), device=device, dtype=torch.long)
            empty_values = torch.zeros(0, device=device, dtype=dtype)
            return torch.sparse_coo_tensor(empty_indices, empty_values, shape)

    def _process_2d(self, feat_x, feat_y=None):
        """
        Process 2D tensors (optimized path for batch size 1)
        """
        # Handle case where feat_x is already a similarity matrix
        if feat_y is None:
            # Assuming feat_x is a similarity matrix [Nx, Ny]
            return self._process_similarity_matrix_2d(feat_x / self.tau)
        
        # Get dimensions
        Nx, C = feat_x.shape
        Ny, _ = feat_y.shape
        device = feat_x.device
        orig_dtype = feat_x.dtype
        
        # Determine optimal chunk size
        chunk_size = self._get_chunk_size(Nx, Ny)
        
        # Use mixed precision for compute-intensive operations
        compute_dtype = torch.float16 if self.use_mixed_precision and device.type == 'cuda' else orig_dtype
        
        # Convert to compute precision if needed
        if compute_dtype != orig_dtype:
            feat_x_compute = feat_x.to(compute_dtype)
            feat_y_compute = feat_y.to(compute_dtype)
        else:
            feat_x_compute = feat_x
            feat_y_compute = feat_y
        
        # Transpose for matrix multiplication
        y_transposed = feat_y_compute.t()
        
        # Initialize containers for sparse tensor creation
        indices_list = []
        values_list = []
        
        # Process in chunks
        for start_idx in range(0, Nx, chunk_size):
            end_idx = min(start_idx + chunk_size, Nx)
            current_chunk_size = end_idx - start_idx
            
            # Get current chunk
            feat_x_chunk = feat_x_compute[start_idx:end_idx]
            
            # Compute similarity with mixed precision
            similarity = torch.matmul(feat_x_chunk, y_transposed) / self.tau
            
            # If using FP16 for topk, we can use the compute-type similarity directly
            # Otherwise, convert back to original precision for accuracy
            if not self.fp16_topk and compute_dtype != orig_dtype:
                similarity_topk = similarity.to(orig_dtype)
            else:
                similarity_topk = similarity
            
            # Fast top-k with mixed precision
            topk_indices, topk_softmax = fast_topk_2d_mixed_precision(similarity_topk, self.k_neighbors)
            
            # Create row indices
            row_indices = torch.arange(start_idx, end_idx, device=device).view(-1, 1).expand(
                current_chunk_size, topk_indices.size(1))
            
            # Reshape for COO format
            rows = row_indices.reshape(-1)
            cols = topk_indices.reshape(-1)
            vals = topk_softmax.reshape(-1)
            
            # Stack indices and store
            curr_indices = torch.stack([rows, cols], dim=0)
            indices_list.append(curr_indices)
            values_list.append(vals)
            
            # Free memory
            del similarity, similarity_topk, topk_indices, topk_softmax, row_indices
            torch.cuda.empty_cache()
        
        # Create sparse tensor
        if indices_list:
            indices = torch.cat(indices_list, dim=1)
            values = torch.cat(values_list)
            
            # Ensure values are in original precision for return
            if values.dtype != orig_dtype:
                values = values.to(orig_dtype)
            
            # Apply hard assignment if needed
            return self._apply_hard_assignment_2d(indices, values, (Nx, Ny), device, orig_dtype)
        else:
            # Empty sparse tensor
            indices = torch.zeros((2, 0), device=device, dtype=torch.long)
            values = torch.zeros(0, device=device, dtype=orig_dtype)
            return torch.sparse_coo_tensor(indices, values, (Nx, Ny))
    
    def _process_similarity_matrix_2d(self, similarity):
        """
        Process 2D similarity matrix
        """
        Nx, Ny = similarity.shape
        device = similarity.device
        orig_dtype = similarity.dtype
        
        # Determine compute precision
        compute_dtype = torch.float16 if self.use_mixed_precision and device.type == 'cuda' else orig_dtype
        chunk_size = self._get_chunk_size(Nx, Ny)
        
        # Convert to compute precision if different
        if compute_dtype != orig_dtype:
            similarity_compute = similarity.to(compute_dtype)
        else:
            similarity_compute = similarity
        
        indices_list = []
        values_list = []
        
        # Process in chunks
        for start_idx in range(0, Nx, chunk_size):
            end_idx = min(start_idx + chunk_size, Nx)
            current_chunk_size = end_idx - start_idx
            
            # Get chunk
            chunk = similarity_compute[start_idx:end_idx]
            
            # For topk, use original precision if needed for accuracy
            if not self.fp16_topk and compute_dtype != orig_dtype:
                chunk_topk = chunk.to(orig_dtype)
            else:
                chunk_topk = chunk
            
            # Fast top-k with mixed precision
            topk_indices, topk_softmax = fast_topk_2d_mixed_precision(chunk_topk, self.k_neighbors)
            
            # Create row indices
            row_indices = torch.arange(start_idx, end_idx, device=device).view(-1, 1).expand(
                current_chunk_size, topk_indices.size(1))
            
            # Reshape for COO format
            rows = row_indices.reshape(-1)
            cols = topk_indices.reshape(-1)
            vals = topk_softmax.reshape(-1)
            
            # Stack indices and store
            curr_indices = torch.stack([rows, cols], dim=0)
            indices_list.append(curr_indices)
            values_list.append(vals)
            
            # Free memory
            del chunk, chunk_topk, topk_indices, topk_softmax, row_indices
            torch.cuda.empty_cache()
        
        # Create sparse tensor
        if indices_list:
            indices = torch.cat(indices_list, dim=1)
            values = torch.cat(values_list)
            
            # Ensure values are in original precision for return
            if values.dtype != orig_dtype:
                values = values.to(orig_dtype)
            
            # Apply hard assignment if needed
            return self._apply_hard_assignment_2d(indices, values, (Nx, Ny), device, orig_dtype)
        else:
            # Empty sparse tensor
            indices = torch.zeros((2, 0), device=device, dtype=torch.long)
            values = torch.zeros(0, device=device, dtype=orig_dtype)
            return torch.sparse_coo_tensor(indices, values, (Nx, Ny))
    
    def _process_3d(self, feat_x, feat_y=None):
        """
        Process 3D tensors (batched processing) with optimized memory usage.
        Always uses sequential processing to minimize peak memory footprint.
        """
        if feat_y is None:
            # Similarity matrix case
            return self._process_similarity_matrix_3d(feat_x / self.tau)
        
        # Get dimensions
        B, Nx, C = feat_x.shape
        _, Ny, _ = feat_y.shape
        device = feat_x.device
        orig_dtype = feat_x.dtype
        
        # Determine compute precision
        compute_dtype = torch.float16 if self.use_mixed_precision and device.type == 'cuda' else orig_dtype
        
        # For large point clouds, use smaller chunks
        if Nx * Ny > 5_000_000:  # Reduced threshold for large point clouds
            chunk_size = min(self.base_chunk_size // 4, 2000)  # Much smaller chunks
        else:
            chunk_size = self._get_chunk_size(Nx, Ny)
        
        # Convert to compute precision if needed
        if compute_dtype != orig_dtype:
            feat_x_compute = feat_x.to(compute_dtype)
            feat_y_compute = feat_y.to(compute_dtype)
        else:
            feat_x_compute = feat_x
            feat_y_compute = feat_y
        
        indices_list = []
        values_list = []
        
        # Always process batch by batch (sequentially)
        for b in range(B):
            feat_x_b = feat_x_compute[b]  # [Nx, C]
            feat_y_b = feat_y_compute[b]  # [Ny, C]
            y_transposed = feat_y_b.t()  # [C, Ny]
            
            # Process in chunks
            for start_idx in range(0, Nx, chunk_size):
                end_idx = min(start_idx + chunk_size, Nx)
                current_chunk_size = end_idx - start_idx
                
                # Get current chunk
                feat_x_chunk = feat_x_b[start_idx:end_idx]  # [chunk_size, C]
                
                # Compute similarity chunk
                similarity = torch.matmul(feat_x_chunk, y_transposed) / self.tau  # [chunk_size, Ny]
                
                # Convert to original precision for topk if needed
                if not self.fp16_topk and compute_dtype != orig_dtype:
                    similarity_topk = similarity.to(orig_dtype)
                else:
                    similarity_topk = similarity
                
                # Fast top-k with mixed precision
                topk_indices, topk_softmax = fast_topk_2d_mixed_precision(similarity_topk, self.k_neighbors)
                
                # Create indices for sparse tensor
                batch_indices = torch.full((current_chunk_size * topk_indices.size(1),), b, 
                                          device=device, dtype=torch.long)
                row_indices = torch.arange(start_idx, end_idx, device=device).view(-1, 1).expand(
                              current_chunk_size, topk_indices.size(1)).reshape(-1)
                col_indices = topk_indices.reshape(-1)
                
                # Stack indices and append
                curr_indices = torch.stack([batch_indices, row_indices, col_indices], dim=0)
                indices_list.append(curr_indices)
                values_list.append(topk_softmax.reshape(-1))
                
                # Free memory immediately
                del similarity, similarity_topk, topk_indices, topk_softmax
                del batch_indices, row_indices, col_indices
                torch.cuda.empty_cache()
        
        # Create sparse tensor
        if indices_list:
            indices = torch.cat(indices_list, dim=1)
            values = torch.cat(values_list)
            
            # Ensure values are in original precision for return
            if values.dtype != orig_dtype:
                values = values.to(orig_dtype)
            
            # Apply hard assignment if needed
            return self._apply_hard_assignment_3d(indices, values, (B, Nx, Ny), device, orig_dtype)
        else:
            # Empty sparse tensor
            indices = torch.zeros((3, 0), device=device, dtype=torch.long)
            values = torch.zeros(0, device=device, dtype=orig_dtype)
            return torch.sparse_coo_tensor(indices, values, (B, Nx, Ny))
            
    def _process_similarity_matrix_3d(self, log_alpha):
        """
        Process 3D similarity matrix with optimized memory usage.
        Always uses sequential processing to minimize peak memory footprint.
        """
        B, Nx, Ny = log_alpha.shape
        device = log_alpha.device
        orig_dtype = log_alpha.dtype
        
        # Determine compute precision
        compute_dtype = torch.float16 if self.use_mixed_precision and device.type == 'cuda' else orig_dtype
        
        # For large matrices, use smaller chunks
        if Nx * Ny > 5_000_000:  # Reduced threshold for large matrices
            chunk_size = min(self.base_chunk_size // 4, 2000)  # Much smaller chunks
        else:
            chunk_size = self._get_chunk_size(Nx, Ny)
        
        # Convert to compute precision if different
        if compute_dtype != orig_dtype:
            log_alpha_compute = log_alpha.to(compute_dtype)
        else:
            log_alpha_compute = log_alpha
        
        indices_list = []
        values_list = []
        
        # Always process batch by batch (sequentially)
        for b in range(B):
            for start_idx in range(0, Nx, chunk_size):
                end_idx = min(start_idx + chunk_size, Nx)
                current_chunk_size = end_idx - start_idx
                
                # Get chunk
                chunk = log_alpha_compute[b, start_idx:end_idx]  # [chunk_size, Ny]
                
                # Convert to original precision for topk if needed
                if not self.fp16_topk and compute_dtype != orig_dtype:
                    chunk_topk = chunk.to(orig_dtype)
                else:
                    chunk_topk = chunk
                
                # Fast top-k with mixed precision
                topk_indices, topk_softmax = fast_topk_2d_mixed_precision(chunk_topk, self.k_neighbors)
                
                # Create indices for sparse tensor
                batch_indices = torch.full((current_chunk_size * topk_indices.size(1),), b, 
                                          device=device, dtype=torch.long)
                row_indices = torch.arange(start_idx, end_idx, device=device).view(-1, 1).expand(
                              current_chunk_size, topk_indices.size(1)).reshape(-1)
                col_indices = topk_indices.reshape(-1)
                
                # Stack indices and append
                curr_indices = torch.stack([batch_indices, row_indices, col_indices], dim=0)
                indices_list.append(curr_indices)
                values_list.append(topk_softmax.reshape(-1))
                
                # Free memory immediately
                del chunk, chunk_topk, topk_indices, topk_softmax
                del batch_indices, row_indices, col_indices
                torch.cuda.empty_cache()
        
        # Create sparse tensor
        if indices_list:
            indices = torch.cat(indices_list, dim=1)
            values = torch.cat(values_list)
            
            # Ensure values are in original precision for return
            if values.dtype != orig_dtype:
                values = values.to(orig_dtype)
            
            # Apply hard assignment if needed
            return self._apply_hard_assignment_3d(indices, values, (B, Nx, Ny), device, orig_dtype)
        else:
            # Empty sparse tensor
            indices = torch.zeros((3, 0), device=device, dtype=torch.long)
            values = torch.zeros(0, device=device, dtype=orig_dtype)
            return torch.sparse_coo_tensor(indices, values, (B, Nx, Ny))