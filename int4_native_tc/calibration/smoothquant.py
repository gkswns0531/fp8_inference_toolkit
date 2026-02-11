"""SmoothQuant utilities for activation-weight balancing.

Implements the SmoothQuant technique (Xiao et al., 2023) which transfers
quantization difficulty from activations to weights via per-channel scaling:

    Y = X @ W^T = (X / diag(s)) @ (diag(s) @ W^T) = X_smooth @ W_smooth^T

where s_j = (max|X_j|)^alpha / (max|W_j|)^(1-alpha), alpha=0.5

This equalizes the per-channel ranges between activations and weights,
making both easier to quantize with minimal information loss.
"""

from __future__ import annotations

import torch


def compute_smoothing_factors(
    weight: torch.Tensor,
    input_absmax: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute per-channel smoothing factors.

    Args:
        weight: [N, K] weight matrix
        input_absmax: [K] per-channel max |activation| from calibration
        alpha: Balance factor. 0.5 = equal balancing (default).
               Higher alpha shifts more difficulty to weights.

    Returns:
        smooth_scale: [K] per-channel smoothing factors
    """
    w_absmax = weight.detach().float().abs().amax(dim=0)  # [K]
    act_absmax = input_absmax.float()

    # Avoid division by zero
    w_absmax = w_absmax.clamp(min=1e-5)
    act_absmax = act_absmax.clamp(min=1e-5)

    smooth_scale = (
        act_absmax.pow(alpha) / w_absmax.pow(1 - alpha)
    ).clamp(min=1e-5)

    return smooth_scale


def apply_smoothing_to_weight(
    weight: torch.Tensor,
    smooth_scale: torch.Tensor,
) -> torch.Tensor:
    """Apply smoothing to weight: W_smooth = W * diag(s).

    Each column j of W is multiplied by s_j.

    Args:
        weight: [N, K] weight matrix
        smooth_scale: [K] per-channel smoothing factors

    Returns:
        smoothed_weight: [N, K]
    """
    return weight * smooth_scale.unsqueeze(0).to(weight.dtype)
