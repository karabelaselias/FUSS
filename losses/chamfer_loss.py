import torch
import torch.nn as nn

from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance,  mesh_edge_loss
from pykeops.torch import LazyTensor
from utils.registry import LOSS_REGISTRY

def squared_distances(x, y):
    """
     source, target -> squared distances
     (N, D), (M, D) -> (N, M)
    """
    N, D = x.shape # Batch size, number of source points, features
    M, _ = y.shape # Batch size, number of target points, features

    # Encode as symbolic tensors:
    x_i = LazyTensor(x.view(N, 1, D)) # (N, 1, D)
    y_j = LazyTensor(y.view(1, M, D)) # (1, M, D)

    # Symbolic matrix of squared distances:
    D_ij = ((x_i - y_j)**2).sum(-1) # (N, M), squared distances
    return D_ij

def chamfer_loss(x, y):
    """
    source, target -> loss values
    (N, D), (M, D) -> (,)
    """
    D_ij = squared_distances(x, y) # (N, M) symbolic matrix
    s_i = D_ij.argmin(dim=0).view(-1)  # (M,) x -> y
    s_j = D_ij.argmin(dim=1).view(-1)  # (N,) y -> x

    D_xy = ((torch.index_select(x, 0, s_i)-y)**2).sum(-1).sqrt()
    D_yx = ((torch.index_select(y, 0, s_j)-x)**2).sum(-1).sqrt()
    loss = (D_xy.mean(dim=0) + D_yx.mean(dim=0)) / 2
    return loss

#@LOSS_REGISTRY.register()
#class ChamferLoss(nn.Module):
#    def __init__(self, loss_weight=1.0):
#        super().__init__()
#        self.loss_weight = loss_weight
#    def forward(self, x, y):
#        if not x.is_contiguous():
#            x = x.contiguous()
#        if not y.is_contiguous():
#            y = y.contiguous()
#        loss = chamfer_loss(x, y)
#        return self.loss_weight * loss

@LOSS_REGISTRY.register()
class ChamferLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, mesh_x, mesh_y):
        loss, _ = chamfer_distance(mesh_x, mesh_y, norm=1)
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, verts, faces):
        mesh = Meshes(verts=[verts], faces=[faces])
        loss = mesh_edge_loss(mesh)
        return self.loss_weight * loss

