"""
Core Drifting Field Computations

This module implements the drifting field V(x) computation following Algorithm 2
from the paper "Generative Modeling via Drifting".

The drifting field is defined as:
    V(x) = V+(x) - V-(x)
where:
    V+(x): Attraction from data samples (positive)
    V-(x): Repulsion from generated samples (negative)

Kernel function: K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List


def compute_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute the kernel matrix K(x, y) = exp(-||x - y||^2 / (2 * sigma^2)).
    
    Args:
        x: Source points of shape (N, D)
        y: Target points of shape (M, D)
        sigma: Kernel bandwidth parameter (temperature)
        normalize: Whether to normalize features (feature normalization)
        
    Returns:
        Kernel matrix of shape (N, M)
    """
    if normalize:
        # Feature normalization: scale features so average distance is sqrt(D)
        x = F.normalize(x, p=2, dim=-1) * (x.shape[-1] ** 0.5)
        y = F.normalize(y, p=2, dim=-1) * (y.shape[-1] ** 0.5)
    
    # Compute pairwise squared distances
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y.T
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)  # (N, 1)
    y_sq = (y ** 2).sum(dim=-1, keepdim=True)  # (M, 1)
    
    distances_sq = x_sq + y_sq.T - 2 * (x @ y.T)  # (N, M)
    distances_sq = torch.clamp(distances_sq, min=0.0)  # Numerical stability
    
    # Compute kernel
    kernel = torch.exp(-distances_sq / (2 * sigma ** 2))
    
    return kernel


def softmax_normalize_2d(matrix: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply softmax normalization over both axes (Appendix A.1).
    
    This is the critical detail: normalization happens over both rows and columns
    to ensure proper probability distributions in both directions.
    
    Args:
        matrix: Input matrix of shape (N, M)
        temperature: Temperature for softmax
        
    Returns:
        Normalized matrix of shape (N, M)
    """
    # Apply softmax over rows (dim=-1)
    row_normalized = F.softmax(matrix / temperature, dim=-1)
    
    # Apply softmax over columns (dim=0)
    col_normalized = F.softmax(matrix / temperature, dim=0)
    
    # Combine both normalizations (geometric mean for balanced normalization)
    normalized = torch.sqrt(row_normalized * col_normalized)
    
    # Re-normalize to ensure proper sum
    normalized = normalized / normalized.sum()
    
    return normalized


def compute_V(
    x: torch.Tensor,
    x_data: torch.Tensor,
    x_gen: torch.Tensor,
    sigma: float = 1.0,
    temperature: float = 1.0,
    normalize_features: bool = True,
    normalize_drift: bool = True,
) -> torch.Tensor:
    """
    Compute the drifting field V(x) following Algorithm 2.
    
    V(x) = V+(x) - V-(x)
    where:
        V+(x) = sum_i w_i^+ * (y_i - x)  -- attraction to data samples
        V-(x) = sum_j w_j^- * (z_j - x)  -- repulsion from generated samples
    
    The weights w are computed using softmax normalization over both axes.
    
    Args:
        x: Query points of shape (B, D)
        x_data: Data (positive) samples of shape (N, D)
        x_gen: Generated (negative) samples of shape (M, D)
        sigma: Kernel bandwidth (temperature for kernel)
        temperature: Temperature for softmax normalization
        normalize_features: Whether to apply feature normalization
        normalize_drift: Whether to normalize drift vectors to unit norm
        
    Returns:
        Drifting field V(x) of shape (B, D)
    """
    B, D = x.shape
    N = x_data.shape[0]
    M = x_gen.shape[0]
    
    # Apply feature normalization if requested
    if normalize_features:
        scale = D ** 0.5
        x_norm = F.normalize(x, p=2, dim=-1) * scale
        x_data_norm = F.normalize(x_data, p=2, dim=-1) * scale
        x_gen_norm = F.normalize(x_gen, p=2, dim=-1) * scale
    else:
        x_norm = x
        x_data_norm = x_data
        x_gen_norm = x_gen
    
    # Compute kernel matrices
    K_data = compute_kernel(x_norm, x_data_norm, sigma=sigma, normalize=False)  # (B, N)
    K_gen = compute_kernel(x_norm, x_gen_norm, sigma=sigma, normalize=False)  # (B, M)
    
    # Apply softmax normalization over both axes (Critical: Appendix A.1)
    W_data = softmax_normalize_2d(K_data, temperature=temperature)  # (B, N)
    W_gen = softmax_normalize_2d(K_gen, temperature=temperature)  # (B, M)
    
    # Normalize weights per row to sum to 1
    W_data = W_data / (W_data.sum(dim=-1, keepdim=True) + 1e-8)
    W_gen = W_gen / (W_gen.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Compute attraction field V+ (towards data)
    # V+(x) = sum_i w_i^+ * (y_i - x)
    V_plus = torch.einsum('bn,nd->bd', W_data, x_data) - x  # (B, D)
    
    # Compute repulsion field V- (away from generated)
    # V-(x) = sum_j w_j^- * (z_j - x)
    V_minus = torch.einsum('bm,md->bd', W_gen, x_gen) - x  # (B, D)
    
    # Combine: V = V+ - V-
    V = V_plus - V_minus
    
    # Drift normalization: scale vectors so average norm is 1
    if normalize_drift:
        norms = torch.norm(V, dim=-1, keepdim=True) + 1e-8
        avg_norm = norms.mean()
        V = V / (avg_norm + 1e-8)
    
    return V


def compute_V_multiscale(
    x: torch.Tensor,
    x_data: torch.Tensor,
    x_gen: torch.Tensor,
    sigmas: List[float] = [0.1, 0.5, 1.0, 2.0],
    temperature: float = 1.0,
    normalize_features: bool = True,
    normalize_drift: bool = True,
) -> torch.Tensor:
    """
    Compute the drifting field V(x) at multiple scales (temperatures).
    
    Using multiple temperatures helps capture different granularity of the
    target distribution.
    
    Args:
        x: Query points of shape (B, D)
        x_data: Data (positive) samples of shape (N, D)
        x_gen: Generated (negative) samples of shape (M, D)
        sigmas: List of kernel bandwidths (temperatures)
        temperature: Temperature for softmax normalization
        normalize_features: Whether to apply feature normalization
        normalize_drift: Whether to normalize drift vectors
        
    Returns:
        Averaged drifting field V(x) of shape (B, D)
    """
    V_total = torch.zeros_like(x)
    
    for sigma in sigmas:
        V_scale = compute_V(
            x, x_data, x_gen,
            sigma=sigma,
            temperature=temperature,
            normalize_features=normalize_features,
            normalize_drift=False,  # Normalize after averaging
        )
        V_total = V_total + V_scale
    
    V_total = V_total / len(sigmas)
    
    if normalize_drift:
        norms = torch.norm(V_total, dim=-1, keepdim=True) + 1e-8
        avg_norm = norms.mean()
        V_total = V_total / (avg_norm + 1e-8)
    
    return V_total


def drifting_loss(
    x_gen: torch.Tensor,
    V: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the drifting loss.
    
    Loss = ||G(z) - stopgrad(G(z) + V(G(z)))||^2
    
    The target is computed as stopgrad(x + V(x)), which implements
    fixed-point iteration for learning.
    
    Args:
        x_gen: Generated samples G(z) of shape (B, D)
        V: Drifting field V(x_gen) of shape (B, D)
        target: Stop-gradient target stopgrad(x_gen + V) of shape (B, D)
        
    Returns:
        Scalar loss value
    """
    # L2 loss between generated samples and target
    loss = F.mse_loss(x_gen, target)
    
    return loss
