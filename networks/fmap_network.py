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
    def __init__(self, lmbda=100, resolvant_gamma=0.5, bidirectional=False, lambda_min=10, lambda_max=1000):
        super(RegularizedFMNet, self).__init__()
        
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # For gamma: start with direct value and use softplus + scaling in forward pass
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))  # initialize to produce value close to 0.5

        # For lambda: use a different parameterization
        # We'll use softplus which is more stable for positive values
        lambda_init = torch.log(torch.exp(torch.tensor(float(lmbda))) - 1.0)
        self.lambda_raw = nn.Parameter(lambda_init)

        #self.lmbda = lmbda
        #self.resolvant_gamma = resolvant_gamma
        
        self.bidirectional = bidirectional

    def get_constrained_parameters(self):
        # For gamma - map to (0,1) range
        gamma = torch.sigmoid(self.gamma_raw)
        
        # For lambda - ensure it's positive and in range
        lmbda = self.lambda_min + torch.nn.functional.softplus(self.lambda_raw) * (self.lambda_max - self.lambda_min) / (self.lambda_max)
        lmbda = torch.clamp(lmbda, self.lambda_min, self.lambda_max)
        
        return lmbda, gamma
    
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
        
        # Get constrained parameter values
        lmbda, resolvant_gamma = self.get_constrained_parameters()

        A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = get_mask(evals_x, evals_y, resolvant_gamma)  # [B, K, K]

        A_A_t = torch.bmm(A, A.mT)  # [B, K, K]
        B_A_t = torch.bmm(B, A.mT)  # [B, K, K]

        C_i = []
        for i in range(evals_x.shape[1]):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.shape[0])], dim=0)
            matrix = A_A_t + lmbda * D_i
            rhs = B_A_t[:, [i], :].transpose(1,2)
            if matrix.dtype != torch.float32 or rhs.dtype != torch.float32:
                matrix_f32 = matrix.to(torch.float32)
                rhs_f32 = rhs.to(torch.float32)
                C = torch.linalg.solve(matrix_f32, rhs_f32)
                # convert back
                C = C.to(A.dtype)
            else:
                C = torch.linalg.solve(matrix, rhs)
            
            C_i.append(C.transpose(1, 2))

        Cxy = torch.cat(C_i, dim=1)

        if self.bidirectional:
            D = get_mask(evals_y, evals_x, resolvant_gamma)  # [B, K, K]
            B_B_t = torch.bmm(B, B.mT)  # [B, K, K]
            A_B_t = torch.bmm(A, B.mT)  # [B, K, K]

            C_i = []
            for i in range(evals_y.shape[1]):
                D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_y.shape[0])],
                                dim=0)
                #C = torch.bmm(torch.inverse(B_B_t + self.lmbda * D_i), A_B_t[:, [i], :].transpose(1, 2))
                matrix = B_B_t + lmbda * D_i
                rhs = A_B_t[:, [i], :].transpose(1,2)
                if matrix.dtype != torch.float32 or rhs.dtype != torch.float32:
                    matrix_f32 = matrix.to(torch.float32)
                    rhs_f32 = rhs.to(torch.float32)
                    C = torch.linalg.solve(matrix_f32, rhs_f32)
                    # convert back
                    C = C.to(A.dtype)
                else:
                    C = torch.linalg.solve(matrix, rhs)
                
                C_i.append(C.transpose(1, 2))

            Cyx = torch.cat(C_i, dim=1)
        else:
            Cyx = None

        return Cxy, Cyx
