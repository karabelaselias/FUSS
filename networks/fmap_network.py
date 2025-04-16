import torch
import torch.nn as nn

from utils.registry import NETWORK_REGISTRY

def get_mask(evals_x, evals_y, resolvant_gamma):
    # Compute mask more efficiently
    scaling_factor = torch.maximum(
        torch.max(evals_x, dim=1)[0],
        torch.max(evals_y, dim=1)[0]
    )[:, None]
    
    evals_x_scaled = evals_x / scaling_factor
    evals_y_scaled = evals_y / scaling_factor
    
    # Compute D more efficiently
    evals_gamma1 = torch.pow(evals_x_scaled, resolvant_gamma).unsqueeze(2)  # [B, K, 1]
    evals_gamma2 = torch.pow(evals_y_scaled, resolvant_gamma).unsqueeze(1)  # [B, 1, K]
    
    # Store squares to avoid recomputation
    evals_gamma1_sq = evals_gamma1.square()
    evals_gamma2_sq = evals_gamma2.square()
    
    # Compute M_re and M_im more efficiently
    denom1 = 1 / (evals_gamma1_sq + 1)
    denom2 = 1 / (evals_gamma2_sq + 1)
    
    M_re = evals_gamma2 * denom2 - evals_gamma1 * denom1
    M_im = denom2 - denom1
    
    return M_re.square() + M_im.square()  # [B, K, K]

@NETWORK_REGISTRY.register()
class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation in DPFM"""
    def __init__(self, lmbda=100, resolvant_gamma=0.5, bidirectional=False):
        super(RegularizedFMNet, self).__init__()
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma
        self.bidirectional = bidirectional

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        """
        Forward pass to compute functional map
        Args:
            feat_x (torch.Tensor): feature vector of shape x. [B, Vx, C].
            feat_y (torch.Tensor): feature vector of shape y. [B, Vy, C].
            evals_x (torch.Tensor): eigenvalues of shape x. [B, K].
            evals_y (torch.Tensor): eigenvalues of shape y. [B, K].
            evecs_trans_x (torch.Tensor): pseudo inverse of eigenvectors of shape x. [B, K, Vx].
            evecs_trans_y (torch.Tensor): pseudo inverse of eigenvectors of shape y. [B, K, Vy].

        Returns:
            C (torch.Tensor): functional map from shape x to shape y. [B, K, K].
        """
        A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = get_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]

        A_A_t = torch.bmm(A, A.mT)  # [B, K, K]
        B_A_t = torch.bmm(B, A.mT)  # [B, K, K]

        C_i = []
        for i in range(evals_x.shape[1]):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.shape[0])], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + self.lmbda * D_i), B_A_t[:, [i], :].transpose(1, 2))
            C_i.append(C.transpose(1, 2))

        Cxy = torch.cat(C_i, dim=1)

        if self.bidirectional:
            D = get_mask(evals_y, evals_x, self.resolvant_gamma)  # [B, K, K]
            B_B_t = torch.bmm(B, B.mT)  # [B, K, K]
            A_B_t = torch.bmm(A, B.mT)  # [B, K, K]

            C_i = []
            for i in range(evals_y.shape[1]):
                D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_y.shape[0])],
                                dim=0)
                C = torch.bmm(torch.inverse(B_B_t + self.lmbda * D_i), A_B_t[:, [i], :].transpose(1, 2))
                C_i.append(C.transpose(1, 2))

            Cyx = torch.cat(C_i, dim=1)
        else:
            Cyx = None

        return Cxy, Cyx
