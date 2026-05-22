"""
src/lora.py
LoRA (Low-Rank Adaptation) for CLIP ViT-B/32.

Novelty: We inject LoRA ONLY into MLP layers (c_fc, c_proj),
not attention projections. This avoids interference with
PyTorch's attention fast-path while achieving effective
domain adaptation with only ~1.5M trainable params (1.7% of CLIP).
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with trainable low-rank matrices A and B.

    Forward pass:
        output = W_frozen(x) + (B @ A)(x) * (alpha / r)

    A: initialised with small random values (Gaussian, std=0.01)
    B: initialised to zero — adapter starts as identity (no-op)

    This zero-B initialisation is critical: the model starts
    fine-tuning from exactly the pretrained CLIP state, not
    a random perturbation of it.

    Args:
        original : frozen nn.Linear to wrap
        r        : LoRA rank (higher = more capacity, more params)
        alpha    : scaling factor — effective scale = alpha/r
        dropout  : applied to input before LoRA branch
    """
    def __init__(self, original: nn.Linear,
                 r: int   = 32,
                 alpha: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.original = original
        self.scale    = alpha / r
        self.dropout  = nn.Dropout(dropout)
        self.lora_A   = nn.Parameter(
            torch.randn(r, original.in_features) * 0.01
        )
        self.lora_B   = nn.Parameter(
            torch.zeros(original.out_features, r)
        )
        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

    # Expose .weight and .bias so CLIP internals that access
    # layer attributes directly still work correctly
    @property
    def weight(self): return self.original.weight

    @property
    def bias(self):   return self.original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + lora * self.scale


def apply_lora(visual: nn.Module,
               r: int       = 32,
               alpha: int   = 64,
               dropout: float = 0.1) -> nn.Module:
    """
    Inject LoRA into MLP layers of every ViT transformer block.

    Targets c_fc and c_proj in each ResidualAttentionBlock.
    Attention projection layers (q, k, v, out) are left frozen
    to preserve CLIP's learned attention patterns.

    Args:
        visual  : CLIP visual encoder (nn.Module)
        r       : LoRA rank
        alpha   : LoRA scaling (effective scale = alpha/r)
        dropout : Dropout rate on LoRA input

    Returns:
        visual with LoRA injected
    """
    for block in visual.transformer.resblocks:
        block.mlp.c_fc   = LoRALinear(block.mlp.c_fc,   r, alpha, dropout)
        block.mlp.c_proj = LoRALinear(block.mlp.c_proj, r, alpha, dropout)
    return visual


def count_parameters(model: nn.Module) -> dict:
    """Return trainable and frozen parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total'     : total,
        'trainable' : trainable,
        'frozen'    : total - trainable,
        'pct'       : trainable / total * 100,
    }
