"""
src/train.py
Full CropCLIP training pipeline.

Stages:
1. SimCLR domain adaptation (warm-start visual encoder)
2. LoRA fine-tuning on LeafNet
3. Evaluation on PlantDoc (zero-shot transfer)

Usage:
    python src/train.py                  # full pipeline
    python src/train.py --skip-ssl       # skip SimCLR
    python src/train.py --demo-only      # load checkpoint + launch demo
"""

import os, time, random, warnings, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import open_clip
from datasets import load_dataset
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

from src.config  import CFG
from src.lora    import apply_lora, count_parameters
from src.simclr  import run_simclr
from src.prompts import make_meta_prompt
from src.dataset import LeafNetDataset, PlantDocDataset, CLASS_MAPPING
from src.evaluate import (build_class_embeddings,
                           evaluate_leafnet,
                           evaluate_plantdoc)

# ── Reproducibility ────────────────────────────────────────
torch.manual_seed(CFG['seed'])
random.seed(CFG['seed'])
np.random.seed(CFG['seed'])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_model(load_ssl: bool = True):
    """Build CropCLIP: CLIP → SSL weights → LoRA injection."""
    print('\nBuilding CropCLIP...')
    model, _, preprocess = open_clip.create_model_and_transforms(
        CFG['clip_model'], pretrained=CFG['pretrained'])
    tokenizer = open_clip.get_tokenizer(CFG['clip_model'])

    if load_ssl and os.path.exists(CFG['ssl_checkpoint']):
        state = torch.load(CFG['ssl_checkpoint'], map_location='cpu')
        model.visual.load_state_dict(state)
        print(f'  SSL weights loaded: {CFG["ssl_checkpoint"]}')
    else:
        print('  Using raw CLIP weights (no SSL)')

    for p in model.parameters():
        p.requires_grad = False

    model.visual = apply_lora(
        model.visual,
        r       = CFG['lora_r'],
        alpha   = CFG['lora_alpha'],
        dropout = CFG['lora_dropout'],
    )
    model = model.to(DEVICE)

    stats = count_parameters(model)
    print(f'  Total params     : {stats["total"]/1e6:.1f}M')
    print(f'  Trainable (LoRA) : {stats["trainable"]/1e3:.1f}K '
          f'({stats["pct"]:.2f}%)')

    lora_params = [p for p in model.parameters() if p.requires_grad]
    return model, preprocess, tokenizer, lora_params


def train(model, preprocess, tokenizer, lora_params,
          leafnet, unique_captions, caption_to_idx):
    """LoRA fine-tuning loop."""

    total   = min(len(leafnet),
                  CFG['max_samples'] if CFG['max_samples'] else len(leafnet))
    indices = np.random.RandomState(CFG['seed']).permutation(total)
    val_n   = int(total * CFG['val_split'])

    train_ds = LeafNetDataset(
        leafnet.select(indices[:total-val_n].tolist()),
        preprocess, caption_to_idx)
    val_ds   = LeafNetDataset(
        leafnet.select(indices[total-val_n:].tolist()),
        preprocess, caption_to_idx)
    pd_ds    = PlantDocDataset(CFG['plantdoc_path'], preprocess)

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'],
                              shuffle=False, num_workers=4, pin_memory=True)
    pd_loader    = DataLoader(pd_ds,    batch_size=64,
                              shuffle=False, num_workers=4)

    print(f'\nTrain: {len(train_ds):,}  Val: {len(val_ds):,}  '
          f'PlantDoc: {len(pd_ds):,}')

    total_steps = CFG['epochs'] * len(train_loader)
    warmup      = CFG['warmup_steps']

    optimizer = torch.optim.AdamW(
        lora_params,
        lr           = CFG['lr'],
        weight_decay = CFG['weight_decay'],
        betas        = (0.9, 0.98),
    )

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler(enabled=CFG.get('use_amp', True))

    history     = {k: [] for k in ['loss','val_acc','val_f1',
                                    'pd_acc','pd_f1']}
    best_pd_acc = 0.0
    no_improve  = 0

    print(f'\nTraining: {CFG["epochs"]} epochs | '
          f'patience={CFG["patience"]} | '
          f'LoRA r={CFG["lora_r"]}\n')
    print(f'{"Epoch":>5} {"Loss":>8} {"ValAcc":>8} {"ValF1":>7} '
          f'{"PDAcc":>7} {"PDF1":>6} {"Time":>7}')
    print('-' * 58)

    for epoch in range(1, CFG['epochs'] + 1):
        t0         = time.time()
        epoch_loss = 0
        model.train()

        class_emb = build_class_embeddings(
            model, tokenizer, unique_captions,
            make_meta_prompt, DEVICE
        ).detach()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=CFG.get('use_amp', True)):
                img_emb = F.normalize(model.encode_image(imgs), dim=-1)
                logits  = img_emb @ class_emb.T * model.logit_scale.exp()
                loss    = F.cross_entropy(
                    logits, labels,
                    label_smoothing=CFG.get('label_smoothing', 0.1)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(lora_params, 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(); scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        class_emb = build_class_embeddings(
            model, tokenizer, unique_captions,
            make_meta_prompt, DEVICE
        )
        val_acc, val_f1     = evaluate_leafnet(
            model, val_loader, class_emb, DEVICE)
        pd_acc, pd_f1, _, _ = evaluate_plantdoc(
            model, pd_loader, class_emb,
            caption_to_idx, CLASS_MAPPING, DEVICE)
        elapsed = time.time() - t0

        for k, v in zip(
            ['loss','val_acc','val_f1','pd_acc','pd_f1'],
            [avg_loss, val_acc, val_f1, pd_acc, pd_f1]
        ):
            history[k].append(v)

        marker = ''
        if pd_acc > best_pd_acc:
            best_pd_acc = pd_acc
            no_improve  = 0
            torch.save({
                'epoch'        : epoch,
                'model_state'  : model.state_dict(),
                'logit_scale'  : model.logit_scale.data,
                'pd_acc'       : pd_acc,
                'pd_f1'        : pd_f1,
                'config'       : CFG,
                'unique_captions': unique_captions,
            }, CFG['best_checkpoint'])
            marker = f'  ★ best'
        else:
            no_improve += 1
            marker = f'  ({no_improve}/{CFG["patience"]})'

        print(f'{epoch:>5} {avg_loss:>8.4f} {val_acc:>7.1f}% '
              f'{val_f1:>6.1f}% {pd_acc:>6.1f}% {pd_f1:>5.1f}% '
              f'{elapsed:>6.0f}s{marker}')

        if no_improve >= CFG['patience']:
            print(f'\nEarly stopping at epoch {epoch}')
            break

    print(f'\n★ Best PlantDoc accuracy: {best_pd_acc:.1f}%')
    return history


def plot_history(history):
    epochs_r = range(1, len(history['loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs_r, history['loss'],
                 color='#3d6b45', linewidth=2, marker='o')
    axes[0].set_title('Training Loss')
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_r, history['val_acc'], label='Accuracy',
                 color='#534AB7', linewidth=2, marker='o')
    axes[1].plot(epochs_r, history['val_f1'],  label='Macro F1',
                 color='#534AB7', linewidth=2, marker='s', linestyle='--')
    axes[1].set_title('LeafNet Validation')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs_r, history['pd_acc'], label='Accuracy',
                 color='#D85A30', linewidth=2, marker='o')
    axes[2].plot(epochs_r, history['pd_f1'],  label='Macro F1',
                 color='#D85A30', linewidth=2, marker='s', linestyle='--')
    axes[2].set_title('PlantDoc Generalization (key metric)')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.set_xlabel('Epoch')
    plt.suptitle('CropCLIP Training History', fontsize=13)
    plt.tight_layout()
    os.makedirs(CFG['figures_dir'], exist_ok=True)
    path = os.path.join(CFG['figures_dir'], 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-ssl',  action='store_true')
    parser.add_argument('--demo-only', action='store_true')
    args = parser.parse_args()

    print('Loading LeafNet...')
    ds      = load_dataset('enalis/LeafNet')
    leafnet = ds['train']

    unique_captions = sorted(set(leafnet['caption']))
    caption_to_idx  = {c: i for i, c in enumerate(unique_captions)}
    print(f'  {len(leafnet):,} images · {len(unique_captions)} classes')

    # Phase 1: SimCLR
    if not args.skip_ssl and not args.demo_only:
        if os.path.exists(CFG['ssl_checkpoint']):
            print(f'SSL checkpoint found — skipping SimCLR')
        else:
            import open_clip as oc
            ssl_clip, _, _ = oc.create_model_and_transforms(
                CFG['clip_model'], pretrained=CFG['pretrained'])
            run_simclr(ssl_clip.visual, leafnet, CFG, DEVICE)

    # Phase 2: Build model
    model, preprocess, tokenizer, lora_params = build_model(
        load_ssl=not args.demo_only
    )

    # Phase 3: Train
    if not args.demo_only:
        history = train(
            model, preprocess, tokenizer, lora_params,
            leafnet, unique_captions, caption_to_idx
        )
        plot_history(history)

    # Load best checkpoint
    if os.path.exists(CFG['best_checkpoint']):
        ckpt = torch.load(CFG['best_checkpoint'], map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        model.logit_scale.data = ckpt['logit_scale']
        print(f'\nLoaded checkpoint — PlantDoc: {ckpt["pd_acc"]:.1f}%')

    # Phase 4: Launch demo
    from app import launch_demo
    launch_demo(model, preprocess, tokenizer,
                unique_captions, caption_to_idx)


if __name__ == '__main__':
    main()
