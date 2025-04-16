import torch
import torch.nn as nn

from utils.registry import NETWORK_REGISTRY
from utils.amp_utils import disable_amp


def get_mask(evals1, evals2, resolvant_gamma):
    """Compute mask for functional map regularization"""
    masks = []
    for bs in range(evals1.shape[0]):
        scaling_factor = max(torch.max(evals1[bs]), torch.max(evals2[bs]))
        evals1_norm, evals2_norm = evals1[bs] / scaling_factor, evals2[bs] / scaling_factor
        evals_gamma1 = (evals1_norm ** resolvant_gamma)[None, :]
        evals_gamma2 = (evals2_norm ** resolvant_gamma)[:, None]

        M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
        M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
        masks.append(M_re.square() + M_im.square())
    return torch.stack(masks, dim=0)

# Optimized implementation
@NETWORK_REGISTRY.register()
class RegularizedFMNetOptimized(torch.nn.Module):
    def __init__(self, lmbda=100, resolvant_gamma=0.5, bidirectional=False):
        super(RegularizedFMNetOptimized, self).__init__()
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma
        self.bidirectional = bidirectional

    def _comp_fmap(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        """Optimized implementation using linear solve and chunking"""
        A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = get_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]

        A_t = A.transpose(1, 2)  # [B, C, K]
        A_A_t = torch.bmm(A, A_t)  # [B, K, K]
        B_A_t = torch.bmm(B, A_t)  # [B, K, K]

        B_size, K, _ = A.shape
        _, num_points_x, _ = feat_x.shape  # Extract number of points
        _, num_points_y, _ = feat_y.shape
        num_points = max(num_points_x, num_points_y)
        
        Cxy = torch.zeros((B_size, K, K), device=feat_x.device, dtype=feat_x.dtype)
        
        # Process in chunks to reduce memory usage
        # Adaptive chunk sizing for optimal performance
        if num_points < 10000:  # Small scale
            chunk_size = min(32, K)
        elif num_points < 100000:  # Medium scale
            chunk_size = min(64, K)
        else:  # Large scale
            chunk_size = K  # No chunking for large scale

        for chunk_start in range(0, K, chunk_size):
            chunk_end = min(chunk_start + chunk_size, K)
            
            # Pre-allocate a batch of diagonal matrices for this chunk
            D_batch = torch.zeros((B_size, chunk_end-chunk_start, K, K), device=A_A_t.device, dtype=A_A_t.dtype)
            
            # Fill diagonal matrices efficiently for all batch items and chunk rows
            for b in range(B_size):
                for i_rel, i_abs in enumerate(range(chunk_start, chunk_end)):
                    D_batch[b, i_rel].diagonal().copy_(D[b, i_abs])
            
            # Stack the system matrices and right-hand sides
            systems = A_A_t.unsqueeze(1) + self.lmbda * D_batch  # [B, chunk_size, K, K]
            rhs = B_A_t[:, chunk_start:chunk_end].transpose(2, 1).unsqueeze(-1)  # [B, K, chunk_size, 1]
            
            # Reshape for batched solve
            systems_flat = systems.reshape(-1, K, K)  # [B*chunk_size, K, K]
            rhs_flat = rhs.permute(0, 2, 1, 3).reshape(-1, K, 1)  # [B*chunk_size, K, 1]

            # CRITICAL: Ensure we're using float32 for solve regardless of AMP
            systems_flat = systems_flat.to(torch.float32)
            rhs_flat = rhs_flat.to(torch.float32)
            
            # Solve all systems in the chunk at once
            with disable_amp():
                C_flat = torch.linalg.solve(systems_flat, rhs_flat)
            
            # Convert back to original dtype if needed
            C_flat = C_flat.to(feat_x.dtype)
            
            # Reshape back and store results
            C_chunk = C_flat.reshape(B_size, chunk_end-chunk_start, K, 1)
            Cxy[:, chunk_start:chunk_end] = C_chunk.squeeze(-1)
            del systems, rhs, systems_flat, rhs_flat, C_flat, C_chunk, D_batch
            
        return Cxy
    
    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        Cxy = self._comp_fmap(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
        Cyx = None
        if self.bidirectional:
            Cyx = self._comp_fmap(feat_y, feat_x, evals_y, evals_x, evecs_trans_y, evecs_trans_x)
        return Cxy, Cyx