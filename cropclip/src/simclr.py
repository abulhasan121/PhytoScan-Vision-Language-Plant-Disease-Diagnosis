"""
src/simclr.py
SimCLR self-supervised pre-training for domain adaptation.

Novelty: We run SimCLR BEFORE LoRA injection as a domain
adaptation warm-start — teaching the CLIP visual encoder
invariance to agricultural field photography conditions
(lighting, crop, jitter, blur) before any labeled data is used.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


class SimCLRAugment:
    """
    Dual-view augmentation for SimCLR.

    Each image produces two independently augmented views.
    The model must learn to embed them as similar despite
    different augmentations — creating augmentation-invariant
    representations ideal for agricultural field imagery.

    Augmentations chosen to match real field photography variation:
    - RandomResizedCrop: scale variation (drone altitude changes)
    - ColorJitter: lighting/weather variation
    - GaussianBlur: camera focus variation
    - RandomRotation: leaf orientation variation
    - VerticalFlip: flipped field imagery
    """
    def __init__(self, size: int = 224):
        self.transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.2),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5),
            T.RandomApply([T.RandomRotation(30)], p=0.3),
            T.ToTensor(),
            T.Normalize(
                mean=[0.48145466, 0.4578275,  0.40821073],
                std =[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def __call__(self, img):
        return self.transform(img), self.transform(img)


class UnlabeledLeafNet(Dataset):
    """Unlabeled subset of LeafNet for SimCLR pre-training."""
    def __init__(self, hf_dataset, n: int = 60000, seed: int = 42):
        random.seed(seed)
        idx       = random.sample(range(len(hf_dataset)),
                                   min(n, len(hf_dataset)))
        self.data = hf_dataset.select(idx)
        self.aug  = SimCLRAugment()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.aug(self.data[idx]['image'].convert('RGB'))


def nt_xent_loss(z1: torch.Tensor,
                 z2: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    For each image i in the batch:
    - z1[i] and z2[i] are the positive pair (two views of same image)
    - All other 2*(N-1) representations are negatives

    Lower temperature (0.07 vs original 0.5) creates harder
    negatives — the model must be more precise about which
    representations belong together.

    Args:
        z1, z2      : projected representations [B, D]
        temperature : softmax temperature (lower = harder negatives)
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N  = z1.shape[0]
    z  = torch.cat([z1, z2], dim=0)                    # [2N, D]
    sim = torch.mm(z, z.T) / temperature                # [2N, 2N]
    sim.masked_fill_(
        torch.eye(2*N, dtype=torch.bool, device=z.device), -9e15
    )
    labels = torch.cat([
        torch.arange(N, 2*N),
        torch.arange(N)
    ]).to(z.device)
    return F.cross_entropy(sim, labels)


def build_projector(proj_dim: int = 512) -> nn.Module:
    """
    3-layer MLP projection head for SimCLR.

    Deeper than standard 2-layer projector — gives richer
    contrastive space before projection head is discarded
    after SSL pre-training.

    Input dim  : proj_dim (512 for ViT-B/32)
    Output dim : 256
    """
    return nn.Sequential(
        nn.Linear(proj_dim, proj_dim), nn.ReLU(),
        nn.Linear(proj_dim, proj_dim), nn.ReLU(),
        nn.Linear(proj_dim, 256)
    )


def run_simclr(visual: nn.Module,
               leafnet_dataset,
               cfg: dict,
               device: str = 'cuda') -> nn.Module:
    """
    Run SimCLR pre-training and return adapted visual encoder.

    The projection head is discarded after training —
    only the visual encoder weights are saved and used
    as the warm-start for LoRA fine-tuning.

    Args:
        visual          : CLIP visual encoder
        leafnet_dataset : HuggingFace LeafNet dataset
        cfg             : CFG dict with ssl_* keys
        device          : 'cuda' or 'cpu'

    Returns:
        visual encoder with adapted weights
    """
    import os, time
    from tqdm import tqdm

    proj_dim  = visual.output_dim
    projector = build_projector(proj_dim).to(device)
    visual    = visual.to(device)

    ssl_dataset = UnlabeledLeafNet(leafnet_dataset,
                                    n    = cfg['ssl_samples'],
                                    seed = cfg['seed'])
    ssl_loader  = DataLoader(ssl_dataset, batch_size=256,
                             shuffle=True, num_workers=4,
                             pin_memory=True)

    optimizer = torch.optim.AdamW(
        list(visual.parameters()) + list(projector.parameters()),
        lr=cfg['ssl_lr'], weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['ssl_epochs'] * len(ssl_loader)
    )
    scaler    = torch.cuda.amp.GradScaler(enabled=cfg.get('use_amp', True))

    print(f'\nSimCLR Pre-training')
    print(f'  Samples    : {len(ssl_dataset):,}')
    print(f'  Epochs     : {cfg["ssl_epochs"]}')
    print(f'  Temperature: {cfg["ssl_temp"]}')
    print(f'  Projector  : 512→512→512→256 (3-layer)\n')

    best_loss = float('inf')
    ckpt_path = cfg.get('ssl_checkpoint',
                        'results/checkpoints/ssl_pretrained_visual.pt')
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(1, cfg['ssl_epochs'] + 1):
        visual.train(); projector.train()
        epoch_loss = 0
        pbar = tqdm(ssl_loader,
                    desc=f'SSL Epoch {epoch}/{cfg["ssl_epochs"]}')

        for v1, v2 in pbar:
            v1, v2 = v1.to(device), v2.to(device)
            with torch.cuda.amp.autocast(enabled=cfg.get('use_amp', True)):
                h1, h2 = visual(v1), visual(v2)
                z1, z2 = projector(h1), projector(h2)
                loss   = nt_xent_loss(z1, z2, cfg['ssl_temp'])

            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(); scheduler.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        mean_loss = epoch_loss / len(ssl_loader)
        print(f'  Epoch {epoch} — mean loss: {mean_loss:.4f}')

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(visual.state_dict(), ckpt_path)
            print(f'  Saved SSL checkpoint: {ckpt_path}')

    print(f'\nSimCLR complete. Best loss: {best_loss:.4f}')
    return visual
