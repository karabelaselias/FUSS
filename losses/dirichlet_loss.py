import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import LOSS_REGISTRY
from utils.torch_sparse_mm import sparse_mm


def cdot(X, Y, dim):
    assert X.dim() == Y.dim()
    return torch.sum(torch.mul(X, Y), dim=dim)


@LOSS_REGISTRY.register()
class DirichletLoss(nn.Module):
    def __init__(self, normalize=False, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize

    def forward(self, feats, L):
        assert feats.dim() == 3

        if self.normalize:
            feats = F.normalize(feats, p=2, dim=-1)

        B, _, _ = feats.shape
    
        # Process each batch separately
        products = []
        with torch.no_grad():
            L = L.detach()
        
        for b in range(B):
            # Take one batch of features: [N, K]
            feats_single = feats[b]
            # Multiply: [N, N] @ [N, K] -> [N, K]
            prod_single = sparse_mm(L, feats_single)
            #prod_single = torch.sparse.mm(L, feats_single)
            products.append(prod_single)
    
        # Stack back to [B, N, K]
        prod = torch.stack(products, dim=0)
        de = cdot(feats, prod, dim=1)
        loss = torch.mean(de)
        
        return self.loss_weight * loss
