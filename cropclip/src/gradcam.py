"""
src/gradcam.py
Activation-based saliency maps for CLIP ViT.

Uses L2 norm of patch token activations from the last
ViT transformer block as spatial saliency signal.

More stable than gradient-based GradCAM for LoRA-adapted
CLIP models where gradient flow through frozen layers is
inconsistent. Produces visually meaningful heatmaps that
highlight diseased tissue without requiring backward passes.
"""

import numpy as np
import torch
import cv2


class CLIPGradCAM:
    """
    Activation Saliency for CLIP Vision Transformer.

    Hooks the last ResidualAttentionBlock to capture patch
    token activations. L2 norm of activations serves as a
    spatial saliency map — high norm = high model attention.

    For ViT-B/32 on 224×224 images:
        seq_len = 50 (1 CLS + 49 patch tokens)
        patch grid = 7×7 (upsampled to 224×224)

    Args:
        model : CLIP model with LoRA adapters
    """
    def __init__(self, model):
        self.model = model

    def generate(self,
                 image_tensor: torch.Tensor,
                 class_emb=None,
                 pred_idx: int = None) -> np.ndarray:
        """
        Generate saliency map for one image.

        Args:
            image_tensor : (3, 224, 224) preprocessed tensor
            class_emb    : unused — kept for API compatibility
            pred_idx     : unused — kept for API compatibility

        Returns:
            heatmap : (224, 224) float32 in [0, 1]
                      red=high attention, blue=low attention
        """
        self.model.eval()
        last_block = self.model.visual.transformer.resblocks[-1]
        captured   = [None]

        def hook_fn(module, inp, out):
            captured[0] = out.detach()

        handle = last_block.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                device = next(self.model.parameters()).device
                self.model.encode_image(
                    image_tensor.unsqueeze(0).to(device)
                )
        finally:
            handle.remove()

        if captured[0] is None:
            return np.zeros((224, 224), dtype=np.float32)

        # Layout: [batch, seq_len, embed_dim]
        # Drop CLS token at position 0 — patch tokens only
        acts = captured[0][0, 1:, :]          # [49, 768]
        cam  = acts.norm(dim=-1)              # [49] L2 norm
        n    = int(cam.shape[0] ** 0.5)       # 7
        cam  = cam.cpu().float().numpy().reshape(n, n)
        cam  = cv2.resize(cam, (224, 224))
        cam  = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
