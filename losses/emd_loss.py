import torch
import torch.nn as nn

from utils.registry import LOSS_REGISTRY
from geomloss import SamplesLoss


@LOSS_REGISTRY.register()
class WassersteinLoss(nn.Module):
    def __init__(self, loss_weight=1.0, p=1, blur=0.01, reach=None):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss = SamplesLoss("sinkhorn", p=p, blur=blur, reach=reach)

    def forward(self, x, y):
        x = x.contiguous()
        y = y.contiguous()
        loss = self.loss(x, y)
        return self.loss_weight * loss