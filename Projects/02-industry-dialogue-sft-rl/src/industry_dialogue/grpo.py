"""Small GRPO helpers used to explain the group-relative training signal."""

from __future__ import annotations

import torch


def group_relative_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize scalar rewards inside each prompt group.

    Args:
        rewards: Tensor shaped ``[batch_size, group_size]``. Each row contains
            rewards for responses sampled from the same prompt.
        eps: Numerical guard for groups with near-zero reward variance.

    Returns:
        A tensor with the same shape. Zero-variance groups produce zero
        advantages because they contain no relative preference signal.
    """
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [batch_size, group_size]")
    if rewards.shape[1] < 2:
        raise ValueError("GRPO needs at least two responses per prompt")

    values = rewards.float()
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, keepdim=True, unbiased=False)
    normalized = (values - mean) / std.clamp_min(eps)
    return torch.where(std > eps, normalized, torch.zeros_like(normalized))
