import numpy as np
import numbers
import torch
import torch_sparse

def _to_device(x, device):
    if torch.is_tensor(x):
        x = x.to(device=device)
    return x

def transfer_batch_to_device(batch, device):
    """Efficiently transfer a batch to the specified device"""
    device_batch = {}
    for key, value in batch.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch_sparse.SparseTensor):
        #torch.is_tensor(value[0]):
            # Handle list of sparse tensors
            device_batch[key] = [t.to(device=device) for t in value]
        elif isinstance(value, torch_sparse.SparseTensor):
            device_batch[key] = value.to(device)
        elif torch.is_tensor(value):
            # Handle dense tensors
            device_batch[key] = value.to(device)
        else:
            # Handle non-tensor data
            device_batch[key] = value         
    return device_batch

def to_device(x, device):
    if isinstance(x, list):
        x = [to_device(x_i, device) for x_i in x]
        return x
    elif isinstance(x, dict):
        x = {k: to_device(v, device) for (k, v) in x.items()}
        return x
    else:
        return _to_device(x, device)


def to_numpy(tensor, squeeze=True):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        if squeeze:
            tensor = tensor.squeeze()
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, numbers.Number):
        return np.array([tensor])
    else:
        raise NotImplementedError()


def to_tensor(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray
    elif isinstance(ndarray, np.ndarray):
        return torch.from_numpy(ndarray)
    elif isinstance(ndarray, numbers.Number):
        return torch.tensor(ndarray)
    else:
        raise NotImplementedError()


def to_number(ndarray):
    if isinstance(ndarray, torch.Tensor) or isinstance(ndarray, np.ndarray):
        return ndarray.item()
    elif isinstance(ndarray, numbers.Number):
        return ndarray
    else:
        raise NotImplementedError()


def select_points(pts, idx):
    """
    select points based on given indices
    Args:
        pts (tensor): points [B, N, C]
        idx (tensor): indices [B, M]

    Returns:
        (tensor): selected points [B, M, C]
    """
    selected_pts = torch.stack([pts[i, idx[i]] for i in range(pts.shape[0])], dim=0)
    return selected_pts
