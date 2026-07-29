from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification


def pairwise_reward_loss(
    chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor
) -> torch.Tensor:
    """Bradley-Terry style loss used for chosen/rejected reward modeling."""
    chosen = chosen_rewards.float().reshape(-1)
    rejected = rejected_rewards.float().reshape(-1)
    if chosen.shape != rejected.shape:
        raise ValueError("chosen and rejected reward tensors must have the same shape")
    return -F.logsigmoid(chosen - rejected).mean()


@torch.no_grad()
def pairwise_accuracy(
    chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor
) -> torch.Tensor:
    chosen = chosen_rewards.reshape(-1)
    rejected = rejected_rewards.reshape(-1)
    if chosen.shape != rejected.shape:
        raise ValueError("chosen and rejected reward tensors must have the same shape")
    return (chosen > rejected).float().mean()


def load_scalar_reward_model(model_name_or_path: str):
    """Load a Hugging Face sequence-classification model with one scalar output."""
    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=1,
        trust_remote_code=True,
    )
