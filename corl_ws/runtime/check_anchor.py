"""Executable numerical/gradient check for the paper's auxiliary objective."""

import argparse

import torch

from lerobot_pi0_step import train_step
from semantic_anchor import AnchorConfig, SemanticAnchor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--paper-shape", action="store_true", help="Also check full pi0 feature dimensions")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(7)
    config = AnchorConfig(feature_dim=32, text_dim=24, shared_dim=16, projection_dim=16, attention_heads=4)
    anchor = SemanticAnchor(config).to(device)
    tokens = torch.randn(4, 5, 32, device=device, requires_grad=True)
    teacher = torch.randn(4, 24, device=device, requires_grad=True)
    action_loss = tokens.square().mean()
    total, metrics = anchor.with_action_loss(action_loss, tokens, teacher)
    expected = action_loss + 0.1 * metrics["align_loss"] + 0.01 * metrics["recon_loss"] + 0.075 * metrics["diff_loss"]
    torch.testing.assert_close(total, expected)
    assert all(torch.isfinite(value).all() for value in metrics.values())
    total.backward()
    assert tokens.grad is not None and tokens.grad.abs().sum() > 0
    assert teacher.grad is None  # frozen semantic target
    assert anchor.query.grad is not None and anchor.query.grad.abs().sum() > 0
    print("PASS: objective, finite values, backbone gradient, frozen target gradient")
    print({key: round(value.item(), 5) for key, value in metrics.items() if value.ndim == 0})

    class TinyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(32, 32)

        def forward(self, batch, action_feature_layer=None):
            assert action_feature_layer == 10
            features = self.layer(batch["tokens"])
            return features.square().mean(), {}, features

    policy = TinyPolicy().to(device)
    before = policy.layer.weight.detach().clone()
    numbers = train_step(policy, anchor, {"tokens": torch.randn(4, 5, 32, device=device)},
                         torch.randn(4, 24, device=device),
                         torch.optim.AdamW(policy.parameters(), lr=1e-3),
                         torch.optim.AdamW(anchor.parameters(), lr=1e-3))
    assert not torch.equal(before, policy.layer.weight)
    assert all(torch.isfinite(torch.tensor(value)) for value in numbers.values())
    print("PASS: one combined policy/anchor optimizer step")

    if args.paper_shape:
        full_anchor = SemanticAnchor().to(device)
        full_tokens = torch.randn(2, 50, 1024, device=device, requires_grad=True)
        full_teacher = torch.randn(2, 768, device=device)
        full_loss, _ = full_anchor.with_action_loss(full_tokens.square().mean(), full_tokens, full_teacher)
        full_loss.backward()
        assert full_tokens.grad is not None and torch.isfinite(full_tokens.grad).all()
        print(f"PASS: pi0 layer feature dimensions on {device.type}: [2, 50, 1024] + [2, 768]")


if __name__ == "__main__":
    main()
