"""
Utility functions for probabilistic modeling and math helpers.
"""

import torch
import numpy as np

def reparameterization_trick(mu, logvar):
    """
    Reparameterization trick to sample from a Gaussian distribution
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def KL_div_modified(var1, var2):
    """
    Computes KL divergence between two normal dists. with identical means based only on
    the variances.
    """
    return torch.log(torch.sqrt(var2) / torch.sqrt(var1)) + var1 / (2 * var2) - 0.5

