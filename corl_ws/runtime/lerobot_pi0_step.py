"""One paper-style π0 training step using the local LeRobot layer-10 feature hook.

The caller supplies a real LeRobot batch and matching frozen EgoHOD-Large text
embeddings. This file deliberately does not synthesize either one.
"""

import torch
from torch import Tensor

from semantic_anchor import SemanticAnchor


def train_step(
    policy: torch.nn.Module,
    anchor: SemanticAnchor,
    batch: dict[str, Tensor],
    text_embeddings: Tensor,
    policy_optimizer: torch.optim.Optimizer,
    anchor_optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    policy.train()
    anchor.train()
    policy_optimizer.zero_grad(set_to_none=True)
    anchor_optimizer.zero_grad(set_to_none=True)
    action_loss, _, action_tokens = policy(batch, action_feature_layer=10)
    total_loss, parts = anchor.with_action_loss(action_loss, action_tokens, text_embeddings)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(anchor.parameters(), 1.0)
    policy_optimizer.step()
    anchor_optimizer.step()
    return {"total": float(total_loss.detach()), "action": float(action_loss.detach()),
            **{name: float(value.detach()) for name, value in parts.items() if value.ndim == 0}}
