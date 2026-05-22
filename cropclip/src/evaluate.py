"""
src/evaluate.py
Evaluation helpers for LeafNet validation and PlantDoc test.
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


@torch.no_grad()
def build_class_embeddings(model, tokenizer,
                            unique_captions, prompt_fn,
                            device='cuda'):
    """
    Encode all class captions to L2-normalised text embeddings.
    Rebuilt each epoch since LoRA updates affect text encoder slightly.

    Args:
        model           : CLIP model
        tokenizer       : OpenCLIP tokenizer
        unique_captions : sorted list of 89 class captions
        prompt_fn       : prompt strategy function (e.g. make_meta_prompt)
        device          : 'cuda' or 'cpu'

    Returns:
        class_emb : [N_classes, embed_dim] float tensor
    """
    prompts = [prompt_fn(c) for c in unique_captions]
    all_emb = []
    model.eval()
    for i in range(0, len(prompts), 64):
        tokens = tokenizer(prompts[i:i+64]).to(device)
        all_emb.append(F.normalize(model.encode_text(tokens), dim=-1))
    return torch.cat(all_emb, dim=0)


@torch.no_grad()
def evaluate_leafnet(model, loader, class_emb,
                     device='cuda', max_batches=50):
    """
    Evaluate on LeafNet validation set.

    Args:
        model       : CLIP model
        loader      : DataLoader for LeafNet val split
        class_emb   : [N, D] class embeddings
        max_batches : cap for speed during training

    Returns:
        (accuracy_pct, macro_f1_pct)
    """
    preds, labels = [], []
    model.eval()
    for i, (imgs, lbls) in enumerate(loader):
        if i >= max_batches: break
        emb = F.normalize(model.encode_image(imgs.to(device)), dim=-1)
        preds.extend((emb @ class_emb.T).argmax(dim=-1).cpu().tolist())
        labels.extend(lbls.tolist())
    acc = accuracy_score(labels, preds) * 100
    f1  = f1_score(labels, preds, average='macro', zero_division=0) * 100
    return acc, f1


@torch.no_grad()
def evaluate_plantdoc(model, pd_loader, class_emb,
                      caption_to_idx, class_mapping,
                      device='cuda'):
    """
    Evaluate on PlantDoc real-world test set (zero-shot transfer).

    Critical: true_label comes from CLASS_MAPPING (ground truth),
    pred_label comes from model argmax (prediction).
    These are computed independently to avoid data leakage.

    Args:
        model           : CLIP model
        pd_loader       : DataLoader for PlantDoc
        class_emb       : [N, D] class embeddings
        caption_to_idx  : dict mapping caption → class index
        class_mapping   : dict mapping PlantDoc folder → LeafNet caption
        device          : 'cuda' or 'cpu'

    Returns:
        (accuracy_pct, macro_f1_pct, pred_list, label_list)
    """
    preds, labels = [], []
    model.eval()
    for imgs, cls_names in pd_loader:
        emb  = F.normalize(model.encode_image(imgs.to(device)), dim=-1)
        sims = emb @ class_emb.T
        for i, cls in enumerate(cls_names):
            mapped = class_mapping.get(cls)
            if not mapped or mapped not in caption_to_idx: continue
            true_label = caption_to_idx[mapped]
            pred_label = int(sims[i].argmax().item())
            labels.append(true_label)
            preds.append(pred_label)

    acc = accuracy_score(labels, preds) * 100
    f1  = f1_score(labels, preds, average='macro', zero_division=0) * 100
    return acc, f1, preds, labels
