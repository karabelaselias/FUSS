import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from typing import Optional
import math
import time

from torch_sparse_mm import sparse_mm

# Mock implementation for torch_scatter if not available
try:
    from torch_scatter import scatter_max
except ImportError:
    def scatter_max(src, index, dim=0):
        # Simple implementation for testing purposes
        unique_indices = torch.unique(index)
        max_values = torch.zeros(unique_indices.size(0), dtype=src.dtype, device=src.device)
        argmax = torch.zeros(unique_indices.size(0), dtype=torch.long, device=src.device) - 1
        
        for i, idx in enumerate(unique_indices):
            mask = index == idx
            if mask.any():
                values = src[mask]
                max_val, max_idx = values.max(dim=0)
                max_values[i] = max_val
                argmax[i] = mask.nonzero()[max_idx].item()
        
        return max_values, argmax

class A100OptimizedSparseSimilarity(nn.Module):
    """Simplified version of the sparse similarity implementation for testing"""
    def __init__(self, tau=0.05, k_neighbors=10, chunk_size=4096, streams=2, hard=False, use_half=True):
        super(A100OptimizedSparseSimilarity, self).__init__()
        self.tau = tau
        self.k_neighbors = k_neighbors
        self.base_chunk_size = chunk_size
        self.streams = streams
        self.hard = hard
        self.use_half = use_half
        
    def forward(self, feat_x: torch.Tensor, feat_y: Optional[torch.Tensor] = None):
        """
        Compute sparse similarity between feature sets
        
        Args:
            feat_x: Features [1, Nx, C]
            feat_y: Features [1, Ny, C] (optional)
            
        Returns:
            Sparse tensor representing the 2D similarity matrix
        """
        # Ensure input is 3D with batch size 1
        assert feat_x.dim() == 3 and feat_x.size(0) == 1, "Input must be [1, Nx, C]"
        
        if feat_y is None:
            feat_y = feat_x.clone()
        
        # Ensure contiguous tensors
        feat_x = feat_x.contiguous()
        feat_y = feat_y.contiguous()
        
        # Normalize features with mixed precision
        with autocast(device_type='cuda', enabled=self.use_half):
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
            
            # Compute similarity matrix
            similarity = torch.matmul(feat_x[0], feat_y[0].transpose(-2, -1))  # [Nx, Ny]
            
            # Apply temperature scaling
            similarity = similarity / self.tau
            
            # Find top-k values and indices
            topk_values, topk_indices = torch.topk(
                similarity, k=self.k_neighbors, dim=1, sorted=False
            )
            
            # Apply softmax to get attention weights
            topk_softmax = F.softmax(topk_values, dim=1)
        
        # Convert to sparse format
        n_x, n_y = feat_x.shape[1], feat_y.shape[1]
        device = feat_x.device
        
        # Create indices
        row_indices = torch.arange(n_x, device=device).unsqueeze(1).expand(-1, self.k_neighbors).reshape(-1)
        col_indices = topk_indices.reshape(-1)
        indices = torch.stack([row_indices, col_indices])
        
        # Create values
        values = topk_softmax.reshape(-1)
        
        # Create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, (n_x, n_y), device=device
        ).coalesce()
        
        # Apply hard assignment if needed
        if self.hard:
            sparse_tensor = self._convert_to_hard_assignment(sparse_tensor)
        
        return sparse_tensor.to_sparse_csr()
    
    def _convert_to_hard_assignment(self, sparse_tensor):
        """Simplified hard assignment conversion"""
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        n_x, n_y = sparse_tensor.size()
        device = sparse_tensor.device
        
        # Create unique keys for rows
        row_keys = indices[0]
        
        # Get max values and indices
        max_values, argmax = scatter_max(values, row_keys, dim=0)
        
        # Create mask for max entries
        keep_mask = torch.zeros_like(values, dtype=torch.bool)
        valid_mask = argmax != -1
        
        if valid_mask.any():
            keep_mask[argmax[valid_mask]] = True
        
        # Straight-through estimator for gradient flow
        hard_values = torch.ones_like(values)
        hard_values = hard_values.masked_fill(~keep_mask, 0)
        final_values = hard_values - values.detach() + values
        
        # Create sparse tensor
        result = torch.sparse_coo_tensor(
            indices[:, keep_mask],
            final_values[keep_mask],
            (n_x, n_y),
            device=device
        ).coalesce()
        
        return result

class EncoderNetwork(nn.Module):
    """Simple encoder network"""
    def __init__(self, input_dim=256, output_dim=64):
        super(EncoderNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CompleteModel(nn.Module):
    """Model that uses SparseSimiliarity as a component"""
    def __init__(self, input_dim=256, embed_dim=64, tau=0.05, k_neighbors=10, hard=False, use_half=True):
        super(CompleteModel, self).__init__()
        self.encoder = EncoderNetwork(input_dim, embed_dim)
        self.similarity = A100OptimizedSparseSimilarity(
            tau=tau, k_neighbors=k_neighbors, hard=hard, use_half=use_half
        )
        self.decoder = nn.Linear(embed_dim, input_dim)
        
    def forward(self, x):
        # x shape: [B, N, D]
        batch_size, num_points, _ = x.shape
        
        # Encode features
        encoded = []
        for b in range(batch_size):
            # Process one batch at a time to match the similarity's expected input shape
            features = self.encoder(x[b])  # [N, embed_dim]
            features = features.unsqueeze(0)  # [1, N, embed_dim]
            
            # Compute similarity
            sim = self.similarity(features)
            # Apply similarity as attention
            if isinstance(sim, torch.Tensor) and sim.is_sparse:
                # Convert to dense for this example (in practice, you'd use sparse operations)
                attended_features = sparse_mm(sim, features[0])
                #sim_dense = sim.to_dense()
                #attended_features = torch.matmul(sim_dense, features[0])  # [N, embed_dim]
            else:
                attended_features = features[0]  # Fallback if similarity fails
            
            encoded.append(attended_features)
        
        encoded = torch.stack(encoded)  # [B, N, embed_dim]
        
        # Decode
        output = self.decoder(encoded)  # [B, N, D]
        
        return output

def test_with_amp():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create synthetic data
    batch_size = 2
    num_points = 100
    input_dim = 256
    
    # Model parameters
    embed_dim = 64
    k_neighbors = 10
    use_amp = True
    
    # Create model configurations to test
    configs = [
        {"hard": False, "use_half": use_amp},
        {"hard": True, "use_half": use_amp}
    ]
    
    for config in configs:
        print(f"\nTesting configuration: {config}")
        
        # Create model
        model = CompleteModel(
            input_dim=input_dim, 
            embed_dim=embed_dim,
            k_neighbors=k_neighbors,
            hard=config["hard"],
            use_half=config["use_half"]
        )
        model.to(device)
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create grad scaler for AMP
        scaler = GradScaler(enabled=use_amp)
        
        # Run training for a few steps
        for step in range(5):
            start_time = time.time()
            
            # Create random input
            x = torch.randn(batch_size, num_points, input_dim, device=device)
            target = x.clone()  # Autoencoder-like task
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast(device_type='cuda', enabled=use_amp):
                output = model(x)
                loss = F.mse_loss(output, target)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            # Check for inf/nan gradients
            has_inf_or_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"Warning: {name} has inf or nan gradients!")
                        has_inf_or_nan = True
            
            # Unscale for gradient clipping or other operations (optional)
            scaler.unscale_(optimizer)
            
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step with scaler
            scaler.step(optimizer)
            
            # Update scaler
            scaler.update()
            
            elapsed_time = time.time() - start_time
            
            # Print progress
            print(f"Step {step}: Loss: {loss.item():.6f}, Time: {elapsed_time:.4f}s, "
                  f"Gradient issues: {has_inf_or_nan}")
            
        print("Training completed for this configuration")
        
        # Test inference
        with torch.no_grad():
            with autocast(device_type='cuda', enabled=use_amp):
                x_test = torch.randn(1, num_points, input_dim, device=device)
                out_test = model(x_test)
                print(f"Inference output shape: {out_test.shape}")

if __name__ == "__main__":
    test_with_amp()