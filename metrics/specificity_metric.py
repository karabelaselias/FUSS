import os
import numpy as np
import torch
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance

from utils.tensor_util import to_device, to_numpy
from utils.misc import plot_with_std
from utils.registry import METRIC_REGISTRY

def squared_distances(x, y):
    """
    source, target -> squared distances
    (B, N, D), (B, M, D) -> (B, N, M)
    """
    from pykeops.torch import LazyTensor
    B, N, D = x.shape # Batch size, number of source points, features
    _, M, _ = y.shape # Batch size, number of target points, features

    # Encode as symbolic tensors:
    x_i = LazyTensor(x.view(B, N, 1, D)) # (B, N, 1, D)
    y_j = LazyTensor(y.view(B, 1, M, D)) # (B, 1, M, D)

    # Symbolic matrix of squared distances:
    D_ij = 0.5*((x_i - y_j)**2).sum(-1) # (B, N, M), squared distances
    return D_ij

def chamfer_loss(x, y):
    D_ij = squared_distances(x, y) # (B, N, M) symbolic matrix
    D_xy = D_ij.min(dim=2).sqrt() # (B, N), distances from x to y
    D_yx = D_ij.min(dim=1).sqrt() # (B, M), distances from y to x
    return (D_xy.mean(dim=1) + D_yx.mean(dim=1)).view(-1) / 2 # (B,)

@METRIC_REGISTRY.register()
def calculate_specificity(ssm_model, dataloader_train, logger, device, output_path):
    n_samples = 1000
    specificity_mean = []
    specificity_std = []
    logger.info(f'Calculating Specificity...')

    for mode in tqdm(range(1, ssm_model.variances.shape[0] + 1)):
        samples = ssm_model.generate_random_samples(n_samples=n_samples, n_modes=mode)
        samples = samples.reshape(n_samples, -1, 3).to(device)
        samples = samples - samples.mean(dim=1, keepdim=True)

        distances = np.zeros((n_samples, len(dataloader_train)))

        for index, data in enumerate(dataloader_train):
            data = to_device(data, device)
            target = (data['verts'].to(device) * data['face_area'].to(device))
            target = target.repeat(n_samples, 1, 1)

            #loss, _ = chamfer_distance(target.float(), samples.float(), point_reduction=None, batch_reduction=None)
            #distance = 0.5 * (loss[0].sqrt().mean(dim=1) + loss[1].sqrt().mean(dim=1))
            with torch.no_grad():
                distance = chamfer_loss(target.float(), samples.float())
            #distance = to_numpy(distance)
            #print(distance)
            distances[:, index] = to_numpy(distance)

        specificity_mean_value = distances.min(1).mean()
        specificity_std_value = distances.min(1).std()
        specificity_mean.append(specificity_mean_value)
        specificity_std.append(specificity_std_value)
        logger.info(f'Specificity for mode {mode} is {specificity_mean_value:.10f} +/- {specificity_std_value:.10f}')

    result_path = os.path.join(output_path, "specificity.png")
    specificity_mean = np.array(specificity_mean)
    specificity_std = np.array(specificity_std)
    plot_with_std(np.array(list(range(1, ssm_model.variances.shape[0] + 1))),
                  specificity_mean, specificity_std,
                  "Specificity in mm", result_path)
    np.save(os.path.join(output_path, "specificity_mean.npy"), specificity_mean)
    np.save(os.path.join(output_path, "specificity_std.npy"), specificity_std)