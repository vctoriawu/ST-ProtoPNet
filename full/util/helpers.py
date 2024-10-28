import os
import torch
from torch import Tensor
import numpy as np
import math


def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)


def list_of_distances_3d(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=3).unsqueeze(dim=4) - torch.unsqueeze(Y.permute(2, 0, 1), dim=0).unsqueeze(0)) ** 2, dim=2)


def list_of_distances_3d_dot(X, Y):
    return 2 - (torch.sum(torch.unsqueeze(X, dim=3).unsqueeze(dim=4) * torch.unsqueeze(Y.permute(2, 0, 1), dim=0).unsqueeze(0), dim=2) + 1)    ####### [0, 2]


def list_of_similarities_3d_dot(X, Y):
    return torch.sum(torch.unsqueeze(X, dim=3).unsqueeze(dim=4) * torch.unsqueeze(Y.permute(2, 0, 1), dim=0).unsqueeze(0), dim=2)    #######

def list_of_similarities_3d_lorentz(X: Tensor, Y: Tensor, curv: float = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Compute pairwise Lorentzian distances between two batches of points
    in hyperbolic space in a 3D format, similar to list_of_similarities_3d_dot.

    Args:
        X: Tensor of shape `(num_classes, num_protos_per_class, D)`.
        Y: Tensor of shape `(num_classes, num_protos_per_class, D)`.
        curv: Positive scalar denoting the negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(num_classes, num_protos_per_class, num_classes, num_protos_per_class)`
        with Lorentzian distances between points in X and Y.
    """
    # Compute time components for X and Y
    X_time = torch.sqrt(1 / curv + torch.sum(X**2, dim=-1, keepdim=True))
    Y_time = torch.sqrt(1 / curv + torch.sum(Y**2, dim=-1, keepdim=True))

    # Concatenate the time component to form full Lorentz coordinates
    X_lorentz = torch.cat([X_time, X], dim=-1)  # Shape: (num_classes, num_protos_per_class, D+1)
    Y_lorentz = torch.cat([Y_time, Y], dim=-1)  # Shape: (num_classes, num_protos_per_class, D+1)
    

    # Reshape for broadcasting to compute pairwise Lorentz inner product
    X_expanded = X_lorentz.unsqueeze(3).unsqueeze(4)  # Shape: (num_classes, num_protos_per_class, D+1, 1, 1)
    Y_expanded = Y_lorentz.unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3)  # Shape: (1, 1, D+1, num_classes, num_protos_per_class)

    # Compute the Lorentz inner product
    lorentz_inner = -(X_expanded[:, :, :1] @ Y_expanded[:, :, :1].T) + torch.sum(X_expanded[:, :, 1:] @ Y_expanded[:, :, 1:].T, dim=2)
    #print(X_time.shape)
    #print(Y_time.T.shape)
    #lorentz_inner = -torch.matmul(X_time, Y_time.T) + X @ Y.transpose(1,2)
    #print(lorentz_inner.shape)

    # Scale by curvature and ensure numerical stability for arcosh
    #lorentz_inner = torch.clamp(-curv * lorentz_inner, min=1 + eps)

    # Compute Lorentz distance
    lorentz_distance = torch.acosh(lorentz_inner) / curv**0.5  # Shape: (num_classes, num_protos_per_class, num_classes, num_protos_per_class)
    return lorentz_distance

def make_one_hot(target, target_one_hot):
    target = target.view(-1, 1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_and_write(str, file):
    print(str)
    file.write(str + '\n')


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1