"""Training-only semantic anchoring module from Xu et al., arXiv:2607.13597.

Inputs are layer-10 action-token features [batch, tokens, feature_dim] and
precomputed frozen EgoHOD-Large text embeddings [batch, 768]. The module is
removed after training; the policy inference graph is unchanged.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AnchorConfig:
    feature_dim: int = 1024  # pi0 action expert; SpatialVLA uses 2304
    text_dim: int = 768
    shared_dim: int = 512
    projection_dim: int = 512
    attention_heads: int = 8
    align_weight: float = 0.1  # pi0; SpatialVLA uses about 0.5
    recon_weight: float = 0.01
    diff_weight: float = 0.075
    initial_temperature: float = 0.07  # paper does not state initial value


def _mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    # Paper specifies two layers, but does not state the hidden width/activation.
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim))


class SemanticAnchor(nn.Module):
    """Shared/private decomposition, attention pooling, and symmetric InfoNCE."""

    def __init__(self, config: AnchorConfig = AnchorConfig()) -> None:
        super().__init__()
        if config.shared_dim % config.attention_heads:
            raise ValueError("shared_dim must be divisible by attention_heads")
        if config.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self.config = config
        self.shared_encoder = _mlp(config.feature_dim, config.shared_dim)
        self.private_encoder = _mlp(config.feature_dim, config.shared_dim)
        self.decoder = _mlp(config.shared_dim, config.feature_dim)
        self.query = nn.Parameter(torch.randn(1, 1, config.shared_dim) * 0.02)
        self.attention = nn.MultiheadAttention(config.shared_dim, config.attention_heads, batch_first=True)
        self.action_head = _mlp(config.shared_dim, config.projection_dim)
        self.text_head = _mlp(config.text_dim, config.projection_dim)
        self.log_temperature = nn.Parameter(torch.tensor(config.initial_temperature).log())

    def forward(self, action_tokens: Tensor, text_embedding: Tensor) -> dict[str, Tensor]:
        cfg = self.config
        if action_tokens.ndim != 3 or action_tokens.shape[-1] != cfg.feature_dim:
            raise ValueError(f"action_tokens must have shape [B, T, {cfg.feature_dim}]")
        if text_embedding.ndim != 2 or text_embedding.shape != (action_tokens.shape[0], cfg.text_dim):
            raise ValueError(f"text_embedding must have shape [B, {cfg.text_dim}]")
        if action_tokens.shape[0] < 2 or action_tokens.shape[1] < 1:
            raise ValueError("InfoNCE requires at least two pairs and one action token")

        # Use fp32 for the small auxiliary heads and stable contrastive logits.
        features = action_tokens.float()
        teacher = text_embedding.detach().float()
        shared = self.shared_encoder(features)
        private = self.private_encoder(features)
        reconstructed = self.decoder(shared + private)

        query = self.query.expand(features.shape[0], -1, -1)
        pooled, _ = self.attention(query, shared, shared, need_weights=False)
        action_projection = F.normalize(self.action_head(pooled[:, 0]), dim=-1)
        text_projection = F.normalize(self.text_head(teacher), dim=-1)
        temperature = self.log_temperature.exp().clamp(min=1e-3, max=1.0)
        logits = action_projection @ text_projection.T / temperature
        labels = torch.arange(features.shape[0], device=features.device)
        align = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        residual = reconstructed - features
        recon = residual.square().mean() + residual.mean(dim=-1).square().mean()
        diff = F.cosine_similarity(shared, private, dim=-1).square().mean()
        aux = cfg.align_weight * align + cfg.recon_weight * recon + cfg.diff_weight * diff
        return {"aux_loss": aux, "align_loss": align, "recon_loss": recon, "diff_loss": diff,
                "temperature": temperature.detach(), "logits": logits.detach()}

    def with_action_loss(self, action_loss: Tensor, action_tokens: Tensor, text_embedding: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Add training-only losses to the backbone's native scalar action loss."""
        if action_loss.ndim != 0:
            raise ValueError("action_loss must be a scalar")
        metrics = self(action_tokens, text_embedding)
        return action_loss + metrics["aux_loss"], metrics
